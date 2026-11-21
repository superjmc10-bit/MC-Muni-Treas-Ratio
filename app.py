"""Interactive front end:  streamlit run app.py

Inputs live in a sidebar form and are applied when you press **Run
simulation**, so editing several values triggers one run, not one per
keystroke. Results and charts are cached per set of inputs, and only the
view you are looking at is drawn, so switching views is instant.

The app calls the same engine as the CLI: every control writes into a nested
``updates`` dict merged on top of config/default.yaml and the chosen scenario.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from muni_mc import list_scenarios, load_config, run_simulation
from muni_mc import report
from muni_mc import valuation as val
from muni_mc.config import load_scenario, tenor_values
from muni_mc.curves import interp_curve

st.set_page_config(page_title="Muni/UST Ratio Monte Carlo", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1400px}
h1{font-size:2rem !important;padding-bottom:.25rem !important}
section[data-testid="stSidebar"]{min-width:360px}
section[data-testid="stSidebar"] h3{font-size:.95rem;margin:.6rem 0 .1rem}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:.25rem 0 1rem}
.kpi{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:10px 14px}
.kpi .t{font-size:.8rem;opacity:.75}
.kpi .v{font-size:1.65rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.25}
.kpi .f{font-size:.78rem;opacity:.75;margin-top:4px;font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:1px 9px;border-radius:999px;font-size:.75rem;font-weight:600}
</style>""",
    unsafe_allow_html=True,
)

VIEWS = ["Cheap vs expensive", "Forecast", "Drivers", "Macro", "Tax brackets", "Export"]
PATHS = [2000, 5000, 10000, 25000, 50000]


# --------------------------------------------------------------------------- #
#  cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def scenario_names() -> dict[str, str]:
    return {n: load_scenario(n).get("meta", {}).get("scenario_name", n) for n in list_scenarios()}


@st.cache_data(show_spinner=False)
def scenario_config(scenario: str) -> dict:
    return load_config(scenarios=[scenario])


def _float_keys(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            out[float(k)] = v
        except ValueError:
            out[k] = v
    return out


@st.cache_resource(max_entries=16, show_spinner=False)
def simulate(scenario: str, updates_json: str):
    """Run once per distinct set of inputs; the result object is shared, not copied."""
    t0 = time.perf_counter()
    cfg = load_config(scenarios=[scenario], updates=json.loads(updates_json, object_hook=_float_keys))
    res = run_simulation(cfg)
    return res, time.perf_counter() - t0


@st.cache_data(max_entries=128, show_spinner=False)
def chart(run_key: str, name: str, _res, tenor: float | None = None) -> bytes:
    fn = getattr(report, name)
    fig = fn(_res, tenor) if tenor is not None else fn(_res)
    return report.buffer_png(fig, dpi=120)


@st.cache_data(max_entries=8, show_spinner=False)
def excel_bytes(run_key: str, _res) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        return report.write_excel(_res, Path(d) / "results.xlsx").read_bytes()


# --------------------------------------------------------------------------- #
#  small UI helpers
# --------------------------------------------------------------------------- #
def tenor_label(t: float) -> str:
    return f"{t * 12:g}M" if t < 1 else f"{t:g}Y"


def chip(label: str, labels: list[str]) -> str:
    colors = report._zone_colors(len(labels))
    j = labels.index(label) if label in labels else len(labels) // 2
    fg = report.ZONE_TEXT[j] if len(labels) == 5 else report.INK
    return f'<span class="chip" style="background:{colors[j]};color:{fg}">{label}</span>'


def zone_css(labels: list[str]):
    colors = report._zone_colors(len(labels))

    def css(v):
        if v in labels:
            j = labels.index(v)
            fg = report.ZONE_TEXT[j] if len(labels) == 5 else report.INK
            return f"background-color:{colors[j]};color:{fg};font-weight:600"
        return ""

    return css


def table(df: pd.DataFrame, fmt: dict, labels: list[str] | None = None, zone_cols=()):
    sty = df.style.format(fmt, na_rep="")
    if labels and zone_cols:
        sty = sty.map(zone_css(labels), subset=list(zone_cols))
    st.dataframe(sty, hide_index=True, width="stretch")


def pct(v: float) -> str:
    return f"{100 * v:.0f}%"


# --------------------------------------------------------------------------- #
#  sidebar: inputs (one form, applied on "Run simulation")
# --------------------------------------------------------------------------- #
names = scenario_names()
scenario = st.sidebar.selectbox(
    "Scenario preset", list(names), index=list(names).index("base"), format_func=names.get,
    help="Loads a preset. Everything below starts from it and can be fine-tuned.",
)
cfg0 = scenario_config(scenario)
tenors = [float(t) for t in cfg0["tenors"]]


def k(name: str) -> str:  # widget keys are per scenario, so switching preset resets the inputs
    return f"{scenario}:{name}"


tr0, m0, mu0, val0 = cfg0["treasury"], cfg0["macro"], cfg0["muni"], cfg0["valuation"]

# Treasury table: today's curve and the change you expect by the horizon
ust_tenors = sorted(float(t) for t in tr0["curve_start"])
ust_today = np.array([float(tr0["curve_start"][t]) for t in ust_tenors])
mode0 = tr0.get("path_mode", "unchanged")
if mode0 == "shifts":
    chg0 = tenor_values(tr0["shifts_bp"], ust_tenors)
elif mode0 == "custom":
    chg0 = (interp_curve(tr0["curve_target"], ust_tenors) - ust_today) * 100
else:
    chg0 = np.zeros(len(ust_tenors))
ust_df = pd.DataFrame({"Tenor": [tenor_label(t) for t in ust_tenors], "Today %": ust_today,
                       "Change bp": np.round(chg0, 0)})

with st.sidebar.form("inputs", border=False):
    st.form_submit_button("Run simulation", type="primary", width="stretch", key="run_top")
    st.markdown("### Treasury curve")
    path = st.radio("Expected path to horizon", ["Today + changes", "Forwards"], horizontal=True,
                    index=1 if mode0 == "forwards" else 0, key=k("path"),
                    help="The Monte Carlo adds random shocks around this path. "
                         "'Forwards' rolls to the curve implied by today's forwards and ignores the change column.")
    ust_edit = st.data_editor(
        ust_df, key=k("ust"), hide_index=True, num_rows="fixed", disabled=["Tenor"], width="stretch",
        column_config={"Tenor": st.column_config.TextColumn(width="small"),
                       "Today %": st.column_config.NumberColumn(format="%.2f", step=0.01, width="small"),
                       "Change bp": st.column_config.NumberColumn("Δ bp", format="%d", step=5, width="small",
                                                                  help="Change you expect by the horizon")},
    )

    st.markdown("### Economy")
    rec = st.slider("Recession odds, next 12 months", 0, 100, int(round(100 * m0["recession"]["prob_12m"])), 5,
                    format="%d%%", key=k("rec"))
    ff_shift = st.slider("Fed path vs market pricing (bp)", -200, 200, int(m0["fed"].get("market_path_shift_bp", 0)),
                         25, key=k("ff"))
    cpi_t = st.slider("CPI inflation in 12 months", 1.0, 6.0, float(m0["inflation"]["target_12m"]), 0.1,
                      format="%.1f%%", key=k("cpi_t"))

    st.markdown("### Muni market")
    iss = st.slider("Gross issuance, next 12 months", 300, 800, int(cfg0["supply"]["annual_gross_issuance"]), 10,
                    format="$%dbn", key=k("iss"))
    flows_exp = st.slider("Fund flows ($bn per month)", -15.0, 15.0, float(cfg0["flows"]["expected"]), 0.5,
                          format="%.1f", key=k("flows"))

    st.markdown("### Tax")
    top = st.slider("Top federal rate incl. 3.8% NIIT", 20.0, 60.0, round(100 * cfg0["tax"]["marginal_rate"], 1),
                    0.1, format="%.1f%%", key=k("top"))
    p_tax = st.slider("Chance the top rate changes", 0, 100, int(round(100 * cfg0["tax"]["change_prob_12m"])), 5,
                      format="%d%%", key=k("ptax"))

    with st.expander("Advanced"):
        c1, c2 = st.columns(2)
        paths = c1.select_slider("Paths", PATHS, value=min(PATHS, key=lambda p: abs(p - cfg0["simulation"]["n_paths"])),
                                 key=k("paths"))
        horizon = c2.number_input("Horizon (months)", 3, 36, int(cfg0["simulation"]["horizon_months"]), key=k("h"))
        seed = c1.number_input("Seed", value=int(cfg0["simulation"]["seed"]), step=1, key=k("seed"))
        ff0 = c2.number_input("Fed funds mid today (%)", value=float(m0["fed"]["start_mid"]), step=0.125,
                              format="%.3f", key=k("ff0"))
        cpi0 = c1.number_input("CPI today (%)", value=float(m0["inflation"]["start"]), step=0.1, key=k("cpi0"))
        un0 = c2.number_input("Unemployment today (%)", value=float(m0["unemployment"]["start"]), step=0.1,
                              key=k("un0"))
        un_t = c1.number_input("Unemployment in 12m (%)", value=float(m0["unemployment"]["target_12m"]), step=0.1,
                               key=k("un_t"))
        move0 = c2.number_input("MOVE today", value=float(m0["rate_vol"]["start"]), step=5.0, key=k("mv0"))
        move_lr = c1.number_input("MOVE long run", value=float(m0["rate_vol"]["long_run"]), step=5.0, key=k("mvlr"))
        hy0 = c2.number_input("HY OAS today (bp)", value=float(m0["hy_oas"]["start"]), step=10.0, key=k("hy0"))
        hy_lr = c1.number_input("HY OAS long run (bp)", value=float(m0["hy_oas"]["long_run"]), step=10.0, key=k("hylr"))
        reinv = c2.number_input("Reinvestment ($bn/yr)", value=float(cfg0["supply"]["annual_reinvestment"]), step=10.0,
                                key=k("reinv"))
        flows0 = c1.number_input("Flows, recent pace ($bn/mo)", value=float(cfg0["flows"]["start"]), step=0.5,
                                 key=k("fl0"))
        liq = c2.number_input("Liquidity events / yr", 0.0, 2.0, float(cfg0["liquidity_events"]["annual_intensity"]),
                              0.05, key=k("liq"))
        tax_size = c1.number_input("Size of tax change (pp)", -10.0, 10.0, 100 * float(cfg0["tax"]["change_mean"]), 0.5,
                                   key=k("tsize"))
        threat = c2.number_input("Exemption threat odds (%)", 0, 100,
                                 int(round(100 * cfg0["tax"]["exemption_threat_prob_12m"])), 5, key=k("thr"))

        st.caption("MMD AAA curve today")
        mmd_df = pd.DataFrame({"Tenor": [tenor_label(t) for t in tenors],
                               "MMD AAA %": interp_curve(mu0["curve_start"], tenors)})
        mmd_edit = st.data_editor(mmd_df, key=k("mmd"), hide_index=True, num_rows="fixed", disabled=["Tenor"],
                                  width="stretch",
                                  column_config={"MMD AAA %": st.column_config.NumberColumn(format="%.2f", step=0.01)})

        st.caption("Fair-value model by tenor (ratio points)")
        fv_df = pd.DataFrame({
            "Tenor": [tenor_label(t) for t in tenors],
            "Rate sens.": 100 * tenor_values(mu0["rate_level_sensitivity"], tenors),
            "Your tilt": 100 * tenor_values(mu0.get("view_adjustment") or {}, tenors),
        })
        fv_edit = st.data_editor(
            fv_df, key=k("fv"), hide_index=True, num_rows="fixed", disabled=["Tenor"], width="stretch",
            column_config={
                "Rate sens.": st.column_config.NumberColumn(format="%.1f", step=0.5,
                                                            help="Fair-value ratio drop (pts) per +1pp UST yield"),
                "Your tilt": st.column_config.NumberColumn(format="%.1f", step=0.5,
                                                           help="Add (cheaper) or subtract (richer) ratio points"),
            },
        )

        st.caption("Cheap / expensive yardstick (history)")
        ref = val0["reference"]
        ref_df = pd.DataFrame({
            "Tenor": [tenor_label(t) for t in tenors],
            "Mean %": 100 * tenor_values(ref["mean"], tenors),
            "Sd pts": 100 * tenor_values(ref["sd"], tenors),
            "Avg UST %": tenor_values(ref["avg_treasury"], tenors),
        })
        ref_edit = st.data_editor(
            ref_df, key=k("ref"), hide_index=True, num_rows="fixed", disabled=["Tenor"], width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.1f" if c != "Avg UST %" else "%.2f", step=0.1)
                           for c in ["Mean %", "Sd pts", "Avg UST %"]},
        )
        w_hist = st.slider("Signal weight on history (rest on model)", 0, 100,
                           int(round(100 * val0["weight_history"])), 5, format="%d%%", key=k("wh"))

    st.form_submit_button("Run simulation", type="primary", width="stretch", key="run_bottom")

# --------------------------------------------------------------------------- #
#  translate inputs into config updates
# --------------------------------------------------------------------------- #
def col(edited: pd.DataFrame, name: str, default) -> np.ndarray:
    """Edited column with blank cells falling back to the preset value."""
    return edited[name].astype(float).fillna(pd.Series(np.asarray(default, dtype=float), index=edited.index)).values


changes = {float(t): float(c) for t, c in zip(ust_tenors, col(ust_edit, "Change bp", np.zeros(len(ust_tenors))))}
if path == "Forwards":
    tr_u = {"path_mode": "forwards"}
elif any(abs(c) > 1e-9 for c in changes.values()):
    tr_u = {"path_mode": "shifts", "shifts_bp": changes}
else:
    tr_u = {"path_mode": "unchanged"}
tr_u["curve_start"] = {float(t): float(y) for t, y in zip(ust_tenors, col(ust_edit, "Today %", ust_today))}

horizon = int(horizon)
u = {
    "simulation": {"n_paths": int(paths), "horizon_months": horizon, "seed": int(seed),
                   "report_months": sorted({m for m in (3, 6, 12) if m < horizon} | {horizon})},
    "treasury": tr_u,
    "muni": {
        "curve_start": {t: float(y) for t, y in zip(tenors, col(mmd_edit, "MMD AAA %", mmd_df["MMD AAA %"]))},
        "ratio_start": {},
        "rate_level_sensitivity": {t: float(v) / 100 for t, v in zip(tenors, col(fv_edit, "Rate sens.", fv_df["Rate sens."]))},
        "view_adjustment": {t: float(v) / 100 for t, v in zip(tenors, col(fv_edit, "Your tilt", fv_df["Your tilt"]))},
    },
    "macro": {
        "recession": {"prob_12m": rec / 100},
        "fed": {"start_mid": float(ff0), "market_path_shift_bp": int(ff_shift)},
        "inflation": {"start": float(cpi0), "target_12m": float(cpi_t)},
        "unemployment": {"start": float(un0), "target_12m": float(un_t)},
        "rate_vol": {"start": float(move0), "long_run": float(move_lr)},
        "hy_oas": {"start": float(hy0), "long_run": float(hy_lr)},
    },
    "supply": {"annual_gross_issuance": float(iss), "annual_reinvestment": float(reinv)},
    "flows": {"start": float(flows0), "expected": float(flows_exp)},
    "liquidity_events": {"annual_intensity": float(liq)},
    "tax": {"marginal_rate": top / 100, "change_prob_12m": p_tax / 100, "change_mean": float(tax_size) / 100,
            "exemption_threat_prob_12m": int(threat) / 100},
    "valuation": {
        "reference": {
            "mean": {t: float(v) / 100 for t, v in zip(tenors, col(ref_edit, "Mean %", ref_df["Mean %"]))},
            "sd": {t: float(v) / 100 for t, v in zip(tenors, col(ref_edit, "Sd pts", ref_df["Sd pts"]))},
            "avg_treasury": {t: float(v) for t, v in zip(tenors, col(ref_edit, "Avg UST %", ref_df["Avg UST %"]))},
        },
        "weight_history": w_hist / 100,
        "weight_model": 1 - w_hist / 100,
    },
}
updates_json = json.dumps(u, sort_keys=True, default=float)
run_key = hashlib.md5(f"{scenario}|{updates_json}".encode()).hexdigest()

# --------------------------------------------------------------------------- #
#  run
# --------------------------------------------------------------------------- #
try:
    with st.spinner("Running Monte Carlo..."):
        res, secs = simulate(scenario, updates_json)
except ValueError as exc:
    st.error(f"Input problem: {exc}")
    st.stop()

cfg = res.cfg
labels = list(cfg["valuation"]["labels"])
h = res.horizon
today = val.valuation_today(res)
fc = val.forecast_summary(res)
fc_h = fc[fc["Month"] == h].reset_index(drop=True)

st.title("Muni / Treasury Ratio Monte Carlo")
st.caption(f"{names.get(scenario, scenario)} · as of {res.dates[0]:%d %b %Y} · {res.n_paths:,} paths · "
           f"{h}-month horizon · computed in {secs:.1f}s · **higher ratio = munis cheaper vs Treasuries**")

cards = "".join(
    f'<div class="kpi"><div class="t">{tenor_label(t)} AAA ratio</div>'
    f'<div class="v">{today.loc[i, "Ratio %"]:.1f}%</div>{chip(today.loc[i, "Signal"], labels)}'
    f'<div class="f">In {h}m: {fc_h.loc[i, "Median %"]:.1f}% '
    f'<span title="5th-95th percentile">({fc_h.loc[i, "P5 %"]:.0f}–{fc_h.loc[i, "P95 %"]:.0f}%)</span></div></div>'
    for i, t in enumerate(res.tenors)
)
st.markdown(f'<div class="kpis">{cards}</div>', unsafe_allow_html=True)

view = st.segmented_control("View", VIEWS, default=VIEWS[0], required=True, key="view",
                            label_visibility="collapsed") or VIEWS[0]

# --------------------------------------------------------------------------- #
#  views (only the selected one is drawn)
# --------------------------------------------------------------------------- #
if view == "Cheap vs expensive":
    st.image(chart(run_key, "fig_snapshot", res), width="stretch")
    t = today.rename(columns={"UST %": "UST", "MMD AAA %": "MMD AAA", "Ratio %": "Ratio", "Hist mean %": "Hist mean",
                              "z vs history": "z (history)", "Model fair value %": "Fair value",
                              "Gap to FV (pts)": "Gap to FV", "z vs model": "z (model)"})
    t = t[["Tenor", "UST", "MMD AAA", "Ratio", "Hist mean", "z (history)", "History view", "Fair value",
           "Gap to FV", "Model view", "Signal"]]
    table(t, {"UST": "{:.2f}%", "MMD AAA": "{:.2f}%", "Ratio": "{:.1f}%", "Hist mean": "{:.1f}%",
              "z (history)": "{:+.2f}", "Fair value": "{:.1f}%", "Gap to FV": "{:+.1f} pts"},
          labels, ["History view", "Model view", "Signal"])
    st.caption("History view: ratio vs its historical mean and spread. Model view: ratio vs the fair value implied "
               "by rates, tax, supply, flows, volatility, credit and recession risk. Signal blends the two.")

elif view == "Forecast":
    st.image(chart(run_key, "fig_fans", res), width="stretch")
    f = fc_h[["Tenor", "Start %", "P5 %", "P25 %", "Median %", "P75 %", "P95 %", "P(ratio higher)",
              "P(cheap or better)", "P(expensive or worse)"]].rename(columns={
        "Start %": "Today", "P5 %": "5th", "P25 %": "25th", "Median %": "Median", "P75 %": "75th", "P95 %": "95th",
        "P(ratio higher)": "Chance ratio rises", "P(cheap or better)": "Chance cheap",
        "P(expensive or worse)": "Chance expensive"})
    pfmt = {c: "{:.1f}%" for c in ["Today", "5th", "25th", "Median", "75th", "95th"]}
    pfmt.update({c: pct for c in ["Chance ratio rises", "Chance cheap", "Chance expensive"]})
    st.markdown(f"**Ratio distribution in {h} months**")
    table(f, pfmt)
    st.image(chart(run_key, "fig_zone_probabilities", res), width="stretch")
    st.image(chart(run_key, "fig_distributions", res), width="stretch")

elif view == "Drivers":
    pick = st.pills("Tenor", [float(t) for t in res.tenors], default=float(res.tenors[np.argmin(abs(res.tenors - 10))]),
                    format_func=tenor_label, required=True, key="drv_tenor") or 10.0
    st.image(chart(run_key, "fig_attribution", res, float(pick)), width="stretch")
    a = val.attribution(res)
    st.markdown("**Bridge from today's ratio to the expected ratio (ratio points)**")
    table(a, {c: ("{:.1f}%" if "%" in c else "{:+.2f}") for c in a.columns if c != "Tenor"})
    with st.expander("Fair value today, by component"):
        b = val.fair_value_breakdown(res).reset_index()
        table(b, {c: "{:.2f}" for c in b.columns if c != "Tenor"})

elif view == "Macro":
    ev = val.event_summary(res)
    cols = st.columns(len(ev))
    for c, (_, row) in zip(cols, ev.iterrows()):
        c.metric(row["Event within horizon"], pct(row["Probability"]))
    st.image(chart(run_key, "fig_macro", res), width="stretch")

elif view == "Tax brackets":
    st.image(chart(run_key, "fig_tax_breakeven", res), width="stretch")
    tb = val.tax_breakeven_table(res)
    fmt = {c: ("{:+.1f}" if "cushion" in c else pct) for c in tb.columns if c not in ("Bracket", "Breakeven ratio %")}
    fmt["Breakeven ratio %"] = "{:.1f}%"
    table(tb, fmt)
    st.caption("Cushion = today's ratio minus the bracket's breakeven, in ratio points. Positive means AAA munis "
               "out-yield Treasuries after federal tax for that investor.")

elif view == "Export":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Excel workbook**")
        st.caption("Every table, percentile paths for ratios, yields and macro drivers, and the full input list.")
        st.download_button("Download .xlsx", excel_bytes(run_key, res), "muni_ratio_mc.xlsx", on_click="ignore",
                           width="stretch")
    with c2:
        st.markdown("**HTML report**")
        st.caption("Self-contained page with all charts and tables, ready to share.")
        if st.button("Build report", width="stretch"):
            with st.spinner("Rendering charts..."), tempfile.TemporaryDirectory() as d:
                figs = report.make_figures(res, Path(d) / "charts")
                st.session_state["html_report"] = (run_key, report.write_html(res, figs, Path(d) / "r.html").read_bytes())
        saved = st.session_state.get("html_report")
        if saved and saved[0] == run_key:
            st.download_button("Download .html", saved[1], "muni_ratio_report.html", on_click="ignore", width="stretch")
    with c3:
        st.markdown("**These inputs as a scenario**")
        st.caption("Save into config/scenarios/ and run it from the command line.")
        st.download_button("Download .yaml", yaml.safe_dump(json.loads(updates_json), sort_keys=False),
                           "my_scenario.yaml", on_click="ignore", width="stretch")
