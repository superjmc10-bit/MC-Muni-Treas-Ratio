"""Interactive front end:  streamlit run app.py

Every control writes into a nested ``updates`` dict that is merged on top of
config/default.yaml (and the chosen scenario), so the app and the CLI run the
exact same model.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from muni_mc import list_scenarios, load_config, run_simulation
from muni_mc import report
from muni_mc import valuation as val

st.set_page_config(page_title="Muni/UST Ratio Monte Carlo", layout="wide")


def curve_editor(label: str, curve: dict, key: str, help_text: str = "") -> dict:
    df = pd.DataFrame({"Tenor (yrs)": [float(k) for k in curve], "Value": [float(v) for v in curve.values()]})
    edited = st.data_editor(df, key=f"{scenario}:{key}", num_rows="dynamic", hide_index=True, width="stretch",
                            column_config={"Value": st.column_config.NumberColumn(label, format="%.3f", help=help_text)})
    edited = edited.dropna()
    return {float(t): float(v) for t, v in zip(edited["Tenor (yrs)"], edited["Value"])}


@st.cache_data(max_entries=12, show_spinner=False)
def simulate(scenario: str, updates_json: str):
    cfg = load_config(scenarios=[scenario], updates=json.loads(updates_json, object_hook=_float_keys))
    return run_simulation(cfg)


def _float_keys(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            out[float(k)] = v
        except ValueError:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
#  sidebar: inputs
# --------------------------------------------------------------------------- #
st.sidebar.title("Inputs")
scenarios = list_scenarios()
scenario = st.sidebar.selectbox("Starting scenario", scenarios, index=scenarios.index("base"))
cfg0 = load_config(scenarios=[scenario])
u: dict = {}

with st.sidebar.expander("Simulation", expanded=False):
    u["simulation"] = {
        "n_paths": st.slider("Paths", 1000, 50000, int(cfg0["simulation"]["n_paths"]), 1000),
        "horizon_months": st.slider("Horizon (months)", 3, 36, int(cfg0["simulation"]["horizon_months"])),
        "seed": int(st.number_input("Seed", value=int(cfg0["simulation"]["seed"]), step=1)),
    }
    h = u["simulation"]["horizon_months"]
    u["simulation"]["report_months"] = sorted({m for m in (3, 6, 12) if m < h} | {h})

with st.sidebar.expander("Treasury curve", expanded=True):
    tr0 = cfg0["treasury"]
    st.caption("Today's par curve (%)")
    tr = {"curve_start": curve_editor("UST %", tr0["curve_start"], "ust_start")}
    modes = ["unchanged", "forwards", "shifts", "custom"]
    tr["path_mode"] = st.selectbox("Expected path to horizon", modes, index=modes.index(tr0["path_mode"]),
                                   help="The simulation adds shocks around this path")
    if tr["path_mode"] == "shifts":
        st.caption("Change by horizon (bp)")
        tr["shifts_bp"] = curve_editor("bp", tr0.get("shifts_bp") or {2: 0, 10: 0, 30: 0}, "ust_shifts")
    if tr["path_mode"] == "custom":
        st.caption("Curve at horizon (%)")
        tr["curve_target"] = curve_editor("UST %", tr0.get("curve_target") or tr0["curve_start"], "ust_target")
    tr["term_premium"] = {"change_bp": st.slider("Term premium change (bp, long end)", -100, 150,
                                                 int(tr0["term_premium"]["change_bp"]), 5)}
    fv0 = tr0["factor_vol_bp"]
    tr["factor_vol_bp"] = {"level": st.slider("Level vol (bp/yr)", 20, 150, int(fv0["level"]), 5),
                           "slope": int(fv0["slope"]), "curvature": int(fv0["curvature"])}
    u["treasury"] = tr

with st.sidebar.expander("Muni curve (MMD AAA)", expanded=False):
    st.caption("Today's MMD AAA yields (%)")
    u["muni"] = {"curve_start": curve_editor("MMD %", cfg0["muni"]["curve_start"], "mmd_start"),
                 "ratio_start": {}}
    st.caption("Fair-value rate sensitivity: ratio change per +1pp UST")
    u["muni"]["rate_level_sensitivity"] = curve_editor("ratio / 1pp", cfg0["muni"]["rate_level_sensitivity"], "rls")
    st.caption("Your own fair-value tilt (ratio units)")
    u["muni"]["view_adjustment"] = curve_editor("tilt", cfg0["muni"].get("view_adjustment") or {10: 0.0}, "view")

with st.sidebar.expander("Macro", expanded=True):
    m0 = cfg0["macro"]
    u["macro"] = {
        "recession": {"prob_12m": st.slider("Recession probability, 12m", 0.0, 1.0,
                                            float(m0["recession"]["prob_12m"]), 0.05)},
        "inflation": {"start": st.number_input("CPI y/y today (%)", value=float(m0["inflation"]["start"]), step=0.1),
                      "target_12m": st.number_input("CPI y/y target in 12m (%)", value=float(m0["inflation"]["target_12m"]), step=0.1)},
        "unemployment": {"start": st.number_input("Unemployment today (%)", value=float(m0["unemployment"]["start"]), step=0.1),
                         "target_12m": st.number_input("Unemployment in 12m (%)", value=float(m0["unemployment"]["target_12m"]), step=0.1)},
        "fed": {"start_mid": st.number_input("Fed funds mid today (%)", value=float(m0["fed"]["start_mid"]), step=0.125, format="%.3f"),
                "market_path_shift_bp": st.slider("Shift market-implied Fed path (bp)", -200, 200,
                                                  int(m0["fed"].get("market_path_shift_bp", 0)), 25)},
        "rate_vol": {"start": st.number_input("MOVE today", value=float(m0["rate_vol"]["start"]), step=5.0),
                     "long_run": st.number_input("MOVE long run", value=float(m0["rate_vol"]["long_run"]), step=5.0)},
        "hy_oas": {"start": st.number_input("HY OAS today (bp)", value=float(m0["hy_oas"]["start"]), step=10.0),
                   "long_run": st.number_input("HY OAS long run (bp)", value=float(m0["hy_oas"]["long_run"]), step=10.0)},
    }

with st.sidebar.expander("Muni technicals", expanded=True):
    s0, f0 = cfg0["supply"], cfg0["flows"]
    u["supply"] = {
        "annual_gross_issuance": st.slider("Gross issuance next 12m ($bn)", 300, 800, int(s0["annual_gross_issuance"]), 10),
        "annual_reinvestment": st.slider("Reinvestment: coupons+maturities+calls ($bn)", 250, 650, int(s0["annual_reinvestment"]), 10),
    }
    u["flows"] = {
        "start": st.number_input("Fund flows, recent pace ($bn/month)", value=float(f0["start"]), step=0.5),
        "expected": st.number_input("Fund flows, expected ($bn/month)", value=float(f0["expected"]), step=0.5),
    }
    u["liquidity_events"] = {"annual_intensity": st.slider("Liquidity / outflow events per year", 0.0, 1.0,
                                                           float(cfg0["liquidity_events"]["annual_intensity"]), 0.05)}

with st.sidebar.expander("Tax policy", expanded=False):
    t0 = cfg0["tax"]
    u["tax"] = {
        "marginal_rate": st.slider("Top marginal rate incl. NIIT", 0.20, 0.60, float(t0["marginal_rate"]), 0.001, format="%.3f"),
        "change_prob_12m": st.slider("Probability of a top-rate change", 0.0, 1.0, float(t0["change_prob_12m"]), 0.05),
        "change_mean": st.slider("Average size of that change", -0.05, 0.05, float(t0["change_mean"]), 0.005, format="%.3f"),
        "exemption_threat_prob_12m": st.slider("Probability of a tax-exemption threat", 0.0, 1.0,
                                               float(t0["exemption_threat_prob_12m"]), 0.05),
    }

with st.sidebar.expander("Cheap / expensive yardstick", expanded=False):
    v0 = cfg0["valuation"]
    st.caption("Historical mean ratio (decimal)")
    mean = curve_editor("mean", v0["reference"]["mean"], "hist_mean")
    st.caption("Historical sd (decimal)")
    sd = curve_editor("sd", v0["reference"]["sd"], "hist_sd")
    st.caption("Average UST yield in that window (%)")
    avg_t = curve_editor("UST %", v0["reference"]["avg_treasury"], "hist_ust")
    w_h = st.slider("Weight on history (rest on model)", 0.0, 1.0, float(v0["weight_history"]), 0.05)
    u["valuation"] = {"reference": {"mean": mean, "sd": sd, "avg_treasury": avg_t},
                      "weight_history": w_h, "weight_model": 1.0 - w_h}

# --------------------------------------------------------------------------- #
#  run
# --------------------------------------------------------------------------- #
try:
    with st.spinner("Simulating..."):
        res = simulate(scenario, json.dumps(u, default=float))
except ValueError as exc:
    st.error(f"Input problem: {exc}")
    st.stop()

cfg = res.cfg
labels = cfg["valuation"]["labels"]
h = res.horizon
st.title("Muni / Treasury ratio Monte Carlo")
st.caption(f"{cfg['meta'].get('scenario_name', scenario)} · as of {res.dates[0]:%d %b %Y} · {res.n_paths:,} paths · "
           "**higher ratio = munis cheaper vs Treasuries**")

today = val.valuation_today(res)
fc = val.forecast_summary(res)
fc_h = fc[fc["Month"] == h].reset_index(drop=True)
cols = st.columns(len(res.tenors))
for i, c in enumerate(cols):
    c.metric(f"{res.tenors[i]:g}Y ratio · {today.loc[i, 'Signal']}", f"{today.loc[i, 'Ratio %']:.1f}%",
             f"{fc_h.loc[i, 'Median %'] - today.loc[i, 'Ratio %']:+.1f} pts by {h}m (median)", delta_color="off")

tabs = st.tabs(["Cheap vs expensive", "Forecast", "Drivers", "Macro & technicals", "Tax brackets", "Download"])
with tabs[0]:
    st.image(report.buffer_png(report.fig_snapshot(res)))
    st.dataframe(today.round(2), hide_index=True, width="stretch")
with tabs[1]:
    st.image(report.buffer_png(report.fig_fans(res)))
    st.dataframe(fc_h.round(3), hide_index=True, width="stretch")
    st.image(report.buffer_png(report.fig_zone_probabilities(res)))
    st.image(report.buffer_png(report.fig_distributions(res)))
with tabs[2]:
    tenor = st.selectbox("Tenor", [float(t) for t in res.tenors],
                         index=int(np.argmin(np.abs(res.tenors - 10))), format_func=lambda t: f"{t:g}Y")
    st.image(report.buffer_png(report.fig_attribution(res, tenor)))
    st.subheader("Bridge to expected ratio (ratio points)")
    st.dataframe(val.attribution(res).round(2), hide_index=True, width="stretch")
    st.subheader("Fair value today by component (ratio points)")
    st.dataframe(val.fair_value_breakdown(res).round(2), width="stretch")
with tabs[3]:
    st.image(report.buffer_png(report.fig_macro(res)))
    st.dataframe(val.event_summary(res), hide_index=True)
with tabs[4]:
    st.image(report.buffer_png(report.fig_tax_breakeven(res)))
    st.dataframe(val.tax_breakeven_table(res).round(3), hide_index=True, width="stretch")
with tabs[5]:
    with tempfile.TemporaryDirectory() as d:
        xlsx = report.write_excel(res, Path(d) / "results.xlsx").read_bytes()
    st.download_button("Excel workbook (all tables and percentile paths)", xlsx, "muni_ratio_mc.xlsx")
    st.download_button("These inputs as a scenario YAML", yaml.safe_dump(json.loads(json.dumps(u, default=float)), sort_keys=False),
                       "my_scenario.yaml", help="Save into config/scenarios/ and run it from the CLI")
