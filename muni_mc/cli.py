"""Command line interface.

    python -m muni_mc run                              # base case
    python -m muni_mc run --scenario bear_steepener    # layer a scenario
    python -m muni_mc run --set treasury.curve_start.10=4.85 --set macro.recession.prob_12m=0.3
    python -m muni_mc compare base bear_steepener bull_steepener hard_landing
    python -m muni_mc scenarios                        # list scenario files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import report
from . import valuation as val
from .config import PROJECT_ROOT, list_scenarios, load_config, load_scenario
from .model import run_simulation


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="base YAML (default: config/default.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any parameter, e.g. --set treasury.curve_start.10=4.85 (repeatable)")
    p.add_argument("--paths", type=int, help="number of Monte Carlo paths")
    p.add_argument("--horizon", type=int, help="horizon in months")
    p.add_argument("--seed", type=int, help="random seed")
    p.add_argument("--out", help="output directory (default: output/<scenario>)")


def _overrides(args) -> list[str]:
    ov = list(args.overrides)
    if args.paths:
        ov.append(f"simulation.n_paths={args.paths}")
    if args.horizon:
        ov.append(f"simulation.horizon_months={args.horizon}")
        ov.append(f"simulation.report_months={[m for m in (3, 6, 12) if m < args.horizon] + [args.horizon]}")
    if args.seed is not None:
        ov.append(f"simulation.seed={args.seed}")
    return ov


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower() or "run"


def _print_summary(res) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    today = val.valuation_today(res)
    fc = val.forecast_summary(res)
    fc = fc[fc["Month"] == res.horizon].reset_index(drop=True)
    tbl = pd.DataFrame({
        "Tenor": today["Tenor"],
        "Ratio": today["Ratio %"].map("{:.1f}%".format),
        "z(hist)": today["z vs history"].map("{:+.2f}".format),
        "FV": today["Model fair value %"].map("{:.1f}%".format),
        "Signal today": today["Signal"],
        f"{res.horizon}m median": fc["Median %"].map("{:.1f}%".format),
        "5-95%": [f"{a:.0f}-{b:.0f}%" for a, b in zip(fc["P5 %"], fc["P95 %"])],
        "P(cheap+)": fc["P(cheap or better)"].map("{:.0%}".format),
        "P(exp+)": fc["P(expensive or worse)"].map("{:.0%}".format),
    })
    print(f"\n{res.cfg['meta'].get('scenario_name', '')}  |  as of {res.dates[0]:%Y-%m-%d}  |  "
          f"{res.n_paths:,} paths  |  higher ratio = munis cheaper\n")
    print(tbl.to_string(index=False))


def cmd_run(args) -> int:
    cfg = load_config(args.config, scenarios=args.scenario, overrides=_overrides(args))
    res = run_simulation(cfg)
    name = "_".join(args.scenario) if args.scenario else "base"
    out = Path(args.out) if args.out else PROJECT_ROOT / "output" / _slug(name)
    out.mkdir(parents=True, exist_ok=True)
    _print_summary(res)
    figs = report.make_figures(res, out / "charts")
    written = [*figs.values()]
    if not args.no_excel:
        written.append(report.write_excel(res, out / "results.xlsx"))
    if not args.no_html:
        written.append(report.write_html(res, figs, out / "report.html"))
    if args.save_paths:
        p = out / "paths.npz"
        np.savez_compressed(p, ratio=res.ratio, treasury=res.treasury, muni=res.muni,
                            fair_value=res.fair_value, tenors=res.tenors)
        written.append(p)
    print(f"\nWrote {len(written)} files to {out}")
    if not args.no_html:
        print(f"Open: {out / 'report.html'}")
    return 0


def cmd_compare(args) -> int:
    names = args.scenarios or ["base", "bear_steepener", "bull_steepener", "hawkish_fed", "hard_landing", "tax_hike"]
    out = Path(args.out) if args.out else PROJECT_ROOT / "output" / "comparison"
    results = {}
    for n in names:
        cfg = load_config(args.config, scenarios=[n], overrides=_overrides(args))
        results[n] = run_simulation(cfg)
        print(f"  ran {n}")
    tbl = report.scenario_comparison_table(results)
    fig_path = report.save_fig(report.fig_scenarios(results), out / "scenario_comparison.png")
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "scenario_comparison.xlsx", engine="openpyxl") as xw:
        tbl.to_excel(xw, sheet_name="Comparison", index=False)
    if not args.no_html:
        base = results[names[0]]
        figs = report.make_figures(base, out / "charts")
        pcols = [c for c in tbl.columns if "P(" in c]
        extra = "<h2>Scenario comparison</h2>" + report._img(fig_path) + report._table(tbl, pct_cols=pcols)
        report.write_html(base, figs, out / "report.html", extra_sections=extra)
    pd.set_option("display.width", 200)
    show = tbl[[c for c in tbl.columns if "median" in c or c == "Scenario"]]
    print("\nMedian ratio at horizon (%):\n" + show.round(1).to_string(index=False))
    print(f"\nWrote comparison to {out}")
    return 0


def cmd_scenarios(_args) -> int:
    for n in list_scenarios():
        meta = load_scenario(n).get("meta", {})
        print(f"{n:<22} {meta.get('scenario_name', '')}")
        if meta.get("description"):
            print(f"{'':<22} {meta['description']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m muni_mc", description="Muni/Treasury ratio Monte Carlo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one (possibly layered) scenario")
    _common(r)
    r.add_argument("--scenario", action="append", default=[], help="scenario name or YAML path (repeatable, layered)")
    r.add_argument("--no-html", action="store_true")
    r.add_argument("--no-excel", action="store_true")
    r.add_argument("--save-paths", action="store_true", help="also save raw simulated paths (.npz)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="run several scenarios side by side")
    c.add_argument("scenarios", nargs="*", help="scenario names (default: a standard set)")
    _common(c)
    c.add_argument("--no-html", action="store_true")
    c.set_defaults(func=cmd_compare)

    s = sub.add_parser("scenarios", help="list available scenarios")
    s.set_defaults(func=cmd_scenarios)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
