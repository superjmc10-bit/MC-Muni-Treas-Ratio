"""Core Monte Carlo engine: muni ratio dynamics on top of the macro block.

For each tenor k and month t the ratio R = MMD AAA / UST evolves as

    1. lag effect      muni yields move b_k x the Treasury move this month
    2. tax repricing   R -= passthrough x (change in top tax rate)
    3. mean reversion  R += (1 - e^{-kappa dt}) (FV - R)
    4. retail floor    pulls R up when it falls below the tax breakeven
    5. crossover cap   pulls R down when banks / insurers / taxable buyers step in
    6. noise           sigma_k (MOVE / ref)^e  sqrt(dt)  eps  (correlated across tenors)
    7. jumps           liquidity events and tax-exemption threats

and the fair value FV is

    FV = (1 - t)                               tax-adjusted Treasury
       + b_k + a_k / UST                        rate level: premium shrinks as UST rises
       + b_sup  * supply_z                      net supply (issuance - reinvestment)
       + b_flow * flows_z                       mutual fund + ETF flows
       + b_move * (MOVE - ref) / 10             rate volatility
       + b_hy   * (HY OAS - ref) / 100          credit / risk appetite
       + b_rec  * recession                     recession regime
       + view                                   your own tilt

(a_k, b_k) are calibrated so that FV equals the historical mean ratio at the
historical average Treasury yield and tax rate, and so that a 1pp rise in
Treasury yields from today's level lowers FV by ``rate_level_sensitivity``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import seasonal_weights, tenor_values
from .curves import interp_curve, interp_weights
from .macro import DT, simulate_macro, step_dates
from .valuation import reference_stats

FV_COMPONENTS: list[tuple[str, str]] = [
    ("tax", "Tax-adjusted Treasury (1 - t)"),
    ("rate_level", "Rate level (premium / UST)"),
    ("supply", "Net supply"),
    ("flows", "Fund flows"),
    ("rate_vol", "Rate volatility (MOVE)"),
    ("risk", "Risk appetite (HY OAS)"),
    ("recession", "Recession regime"),
    ("view", "User view"),
]


@dataclass
class SimulationResult:
    cfg: dict
    tenors: np.ndarray
    dates: list[pd.Timestamp]
    ratio: np.ndarray            # (N, H+1, K) AAA muni / UST
    fair_value: np.ndarray       # (N, H+1, K)
    treasury: np.ndarray         # (N, H+1, K) %
    muni: np.ndarray             # (N, H+1, K) MMD AAA %
    fv_components: dict          # key -> (H+1, K) path means
    macro: dict                  # name -> (N, H+1)
    micro: dict                  # name -> (N, H+1)
    credit_spreads: dict         # rating -> (N, H+1, K) bp over AAA
    tax_rate: np.ndarray         # (N, H+1)
    events: dict                 # name -> (N,) bool
    reference: dict              # mean, sd, avg_treasury, source
    premium: np.ndarray          # (K,) rate-level premium over the tax breakeven, today
    report_months: list = field(default_factory=list)

    @property
    def horizon(self) -> int:
        return len(self.dates) - 1

    @property
    def n_paths(self) -> int:
        return self.ratio.shape[0]

    def rated_ratio(self, rating: str) -> np.ndarray:
        """Ratio for a lower-rated curve: (AAA + spread) / UST."""
        if rating.upper() == "AAA":
            return self.ratio
        return (self.muni + self.credit_spreads[rating] / 100.0) / self.treasury


def _tenor_corr_chol(tenors: np.ndarray, length: float) -> np.ndarray:
    d = np.abs(tenors[:, None] - tenors[None, :])
    return np.linalg.cholesky(np.exp(-d / float(length)) + 1e-12 * np.eye(len(tenors)))


def _one_off_events(
    rng: np.random.Generator, n: int, h: int, p12: float, month: int | None
) -> np.ndarray:
    """Month (1..h) in which a one-off event happens on each path, 0 if never."""
    p_h = 1.0 - (1.0 - p12) ** (h / 12.0) if p12 < 1.0 else 1.0
    happens = rng.random(n) < p_h
    if month:
        when = np.full(n, min(max(int(month), 1), h))
    else:
        when = rng.integers(1, h + 1, size=n)
    return np.where(happens, when, 0)


def run_simulation(cfg: dict) -> SimulationResult:
    sim = cfg["simulation"]
    n, h = int(sim["n_paths"]), int(sim["horizon_months"])
    rng = np.random.default_rng(sim.get("seed"))
    tenors = np.asarray(cfg["tenors"], dtype=float)
    k = len(tenors)
    dates = step_dates(cfg["meta"]["as_of"], h)

    macro = simulate_macro(cfg, rng, tenors, dates)
    T = macro["treasury"]
    rec = macro["recession"]
    move, hy = macro["move"], macro["hy_oas"]

    mu, sup, fl, cr, tax, liq = (
        cfg["muni"], cfg["supply"], cfg["flows"], cfg["credit"], cfg["tax"], cfg["liquidity_events"]
    )
    mv_ref = float(cfg["macro"]["rate_vol"]["reference"])
    hy_ref = float(cfg["macro"]["hy_oas"]["reference"])
    mv_base, hy_base = macro["move_base"], macro["hy_base"]

    def tv(param, name):
        return tenor_values(param, tenors, name)

    # ---------------- starting point ----------------
    if mu.get("ratio_start"):
        r0 = tv(mu["ratio_start"], "muni.ratio_start")
        r0 = np.where(r0 > 2.0, r0 / 100.0, r0)  # accept percent
    else:
        r0 = interp_curve(mu["curve_start"], tenors) / T[0, 0, :]

    # ---------------- parameters ----------------
    ref = reference_stats(cfg, tenors)
    tax_hist = float(tax["marginal_rate_hist"])
    # Rate-level term: ratio(T) = b_k + a_k / T, a hyperbola that
    #   * passes through the historical mean ratio at the historical avg UST, and
    #   * has slope -s_k (ratio per 1pp of UST) at today's Treasury yield.
    # It is reported as a premium over the historical tax breakeven.
    t_ref, t0 = ref["avg_treasury"], T[0, 0, :]
    sens_rate = tv(mu["rate_level_sensitivity"], "muni.rate_level_sensitivity")
    a_rate = sens_rate * t0**2
    b_rate = ref["mean"] - a_rate / t_ref - (1.0 - tax_hist)
    premium = b_rate + a_rate / t0

    kappa = np.log(2.0) / (tv(mu["mean_reversion_half_life_months"], "half life") / 12.0)
    sigma = tv(mu["ratio_vol"], "muni.ratio_vol")
    beta_short = tv(mu["short_run_beta"], "muni.short_run_beta")
    floor_off = tv(mu["floor_offset"], "muni.floor_offset")
    ceiling = tv(mu["crossover_ceiling"], "muni.crossover_ceiling")
    sens = mu["sensitivity"]
    b_sup = tv(sens["supply_per_z"], "supply_per_z")
    b_flow = tv(sens["flows_per_z"], "flows_per_z")
    b_move = tv(sens["move_per_10"], "move_per_10")
    b_hy = tv(sens["hy_per_100bp"], "hy_per_100bp")
    b_rec = tv(sens["recession"], "recession")
    view = tv(mu.get("view_adjustment") or {}, "view_adjustment")
    t_min = float(mu["min_treasury_for_fv"])
    vol_el = float(mu["vol_elasticity_to_move"])
    chol = _tenor_corr_chol(tenors, mu["tenor_corr_length"])
    a_mr = 1.0 - np.exp(-kappa * DT)
    a_fl = 1.0 - np.exp(-float(mu["floor_strength"]) * DT)
    a_ce = 1.0 - np.exp(-float(mu["crossover_strength"]) * DT)
    w10 = interp_weights(tenors, 10.0)

    sw_iss = seasonal_weights(sup["issuance_seasonality"])
    sw_re = seasonal_weights(sup["reinvestment_seasonality"])
    g_ann, re_ann = float(sup["annual_gross_issuance"]), float(sup["annual_reinvestment"])
    ref_share = float(sup["refunding_share"])
    sig_iss = float(sup["issuance_vol"])

    liq_mult = tv(liq["tenor_multiplier"], "liquidity tenor_multiplier")
    jm, js = float(liq["jump_mean"]), float(liq["jump_sd"])
    ln_sd = np.sqrt(np.log(1.0 + (js / jm) ** 2)) if jm > 0 else 0.0
    ln_mu = np.log(jm) - 0.5 * ln_sd**2 if jm > 0 else 0.0

    ratings = list(cr["spreads_10y_bp"].keys())
    cr_mult = tv(cr["tenor_multiplier"], "credit tenor_multiplier")

    # ---------------- one-off events ----------------
    tax_when = _one_off_events(rng, n, h, float(tax["change_prob_12m"]), tax.get("change_month"))
    tax_size = rng.normal(float(tax["change_mean"]), float(tax["change_sd"]), size=n)
    thr_when = _one_off_events(rng, n, h, float(tax["exemption_threat_prob_12m"]), None)

    # ---------------- storage ----------------
    R = np.empty((n, h + 1, k))
    FV = np.empty((n, h + 1, k))
    M = np.empty((n, h + 1, k))
    comps = {key: np.empty((h + 1, k)) for key, _ in FV_COMPONENTS}
    flows = np.empty((n, h + 1))
    net_supply = np.empty((n, h + 1))
    gross = np.empty((n, h + 1))
    tax_rate = np.empty((n, h + 1))
    spreads = {r: np.empty((n, h + 1, k)) for r in ratings}
    liq_any = np.zeros(n, dtype=bool)

    # ---------------- state at t = 0 ----------------
    m0 = dates[0].month - 1
    net0 = g_ann * sw_iss[m0] - re_ann * sw_re[m0]
    supply_z = np.full(n, (net0 - sup["reference_net_monthly"]) / sup["z_scale"])
    flows_prev = np.full(n, float(fl["start"]))
    flows_z = (flows_prev - fl["reference"]) / fl["z_scale"]
    tau = np.full(n, float(tax["marginal_rate"]))
    credit_f = np.zeros(n)
    ret_excess_prev = np.zeros(n)

    def fair_value(t_idx: int, tau_now, sz, fz) -> dict:
        parts = {
            "tax": np.broadcast_to((1.0 - tau_now)[:, None], (n, k)),
            "rate_level": b_rate[None, :] + a_rate[None, :] / np.maximum(T[:, t_idx, :], t_min),
            "supply": b_sup[None, :] * sz[:, None],
            "flows": b_flow[None, :] * fz[:, None],
            "rate_vol": b_move[None, :] * ((move[:, t_idx] - mv_ref) / 10.0)[:, None],
            "risk": b_hy[None, :] * ((hy[:, t_idx] - hy_ref) / 100.0)[:, None],
            "recession": b_rec[None, :] * rec[:, t_idx][:, None].astype(float),
            "view": np.broadcast_to(view[None, :], (n, k)),
        }
        return parts

    parts = fair_value(0, tau, supply_z, flows_z)
    FV[:, 0, :] = sum(parts.values())
    for key in comps:
        comps[key][0] = parts[key].mean(axis=0)
    R[:, 0, :] = r0[None, :]
    M[:, 0, :] = R[:, 0, :] * T[:, 0, :]
    m10_0 = float(M[0, 0, :] @ w10)
    m10_prev = np.full(n, m10_0)
    flows[:, 0] = flows_prev
    net_supply[:, 0] = net0
    gross[:, 0] = g_ann * sw_iss[m0]
    tax_rate[:, 0] = tau
    for r in ratings:
        spreads[r][:, 0, :] = float(cr["spreads_10y_bp"][r]) * cr_mult[None, :]

    # ---------------- monthly loop ----------------
    for t in range(h):
        s = t + 1
        mth = dates[s].month - 1
        now_rec = rec[:, s].astype(float)

        # tax events
        d_tau = np.where(tax_when == s, tax_size, 0.0)
        tau = np.clip(tau + d_tau, 0.0, 0.7)
        threat = thr_when == s

        # liquidity events (state-dependent Poisson intensity)
        lam = (
            float(liq["annual_intensity"])
            * (move[:, s] / mv_ref) ** float(liq["move_elasticity"])
            * (1.0 + (float(liq["recession_multiplier"]) - 1.0) * now_rec)
        )
        liq_hit = rng.random(n) < 1.0 - np.exp(-lam * DT)
        liq_any |= liq_hit
        jump = np.where(liq_hit, np.exp(ln_mu + ln_sd * rng.standard_normal(n)), 0.0) if jm > 0 else 0.0

        # supply: new money + rate-sensitive refundings - reinvestment
        base_g = g_ann * sw_iss[mth]
        refund_base = base_g * ref_share
        refund = refund_base * np.exp(-float(sup["refunding_rate_elasticity"]) * (m10_prev - m10_0))
        noise = np.exp(sig_iss * rng.standard_normal(n) - 0.5 * sig_iss**2)
        g = (base_g * (1.0 - ref_share) + refund) * noise
        redemptions = re_ann * sw_re[mth] + float(sup["refunding_redemption_offset"]) * (refund - refund_base)
        net = g - redemptions
        a_s = float(sup["ema_alpha"])
        supply_z = (1 - a_s) * supply_z + a_s * (net - sup["reference_net_monthly"]) / sup["z_scale"]

        # fund flows: persistence, performance chasing, yield buyers, risk-off
        f_new = (
            fl["expected"]
            + fl["persistence"] * (flows_prev - fl["expected"])
            + fl["performance_chasing"] * ret_excess_prev
            + fl["yield_level"] * (m10_prev - m10_0)
            + fl["move_per_10"] * (move[:, s] - mv_base[s]) / 10.0
            + fl["hy_per_100bp"] * (hy[:, s] - hy_base[s]) / 100.0
            + fl["recession"] * now_rec
            + fl["vol"] * rng.standard_normal(n)
            + liq["flow_hit"] * liq_hit
            + tax["exemption_threat_flow_hit"] * threat
        )
        a_f = float(fl["ema_alpha"])
        flows_z = (1 - a_f) * flows_z + a_f * (f_new - fl["reference"]) / fl["z_scale"]
        flows_prev = f_new

        # fair value
        parts = fair_value(s, tau, supply_z, flows_z)
        fv = sum(parts.values())
        for key in comps:
            comps[key][s] = parts[key].mean(axis=0)

        # ratio dynamics
        t_prev, t_now = T[:, t, :], T[:, s, :]
        r = R[:, t, :]
        r = (r * t_prev + beta_short[None, :] * (t_now - t_prev)) / t_now      # 1 lag
        r = r - float(tax["immediate_passthrough"]) * d_tau[:, None]             # 2 tax
        r = r + a_mr[None, :] * (fv - r)                                         # 3 mean reversion
        floor = (1.0 - tau)[:, None] + floor_off[None, :]
        r = r + a_fl * np.maximum(floor - r, 0.0)                                # 4 retail floor
        r = r - a_ce * np.maximum(r - ceiling[None, :], 0.0)                     # 5 crossover cap
        vs = (move[:, s] / mv_ref) ** vol_el
        eta = rng.standard_normal((n, k)) @ chol.T
        r = r + sigma[None, :] * vs[:, None] * np.sqrt(DT) * eta                 # 6 noise
        r = r + jump[:, None] * liq_mult[None, :] if jm > 0 else r               # 7 jumps
        r = r + float(tax["exemption_threat_shock"]) * threat[:, None]
        r = np.clip(r, 0.15, 2.5)

        R[:, s, :] = r
        FV[:, s, :] = fv
        M[:, s, :] = r * t_now
        m10 = M[:, s, :] @ w10
        ret_excess_prev = -float(fl["index_duration"]) * (m10 - m10_prev)
        m10_prev = m10

        # muni credit spreads (for rated ratios)
        d_lhy = np.log(hy[:, s]) - np.log(hy[:, t])
        credit_f = (
            credit_f
            - float(cr["mean_reversion"]) * credit_f * DT
            + float(cr["fiscal_stress_drift"]) * DT
            + float(cr["hy_elasticity"]) * d_lhy
            + float(cr["recession_shock"]) * macro["recession_entry"][:, s]
            + float(cr["vol"]) * np.sqrt(DT) * rng.standard_normal(n)
        )
        for rt in ratings:
            spreads[rt][:, s, :] = (
                float(cr["spreads_10y_bp"][rt])
                * cr_mult[None, :]
                * np.exp(float(cr["beta"][rt]) * credit_f)[:, None]
            )

        flows[:, s] = f_new
        net_supply[:, s] = net
        gross[:, s] = g
        tax_rate[:, s] = tau

    report_months = sorted({int(m) for m in sim.get("report_months", [])} | {h})
    return SimulationResult(
        cfg=cfg,
        tenors=tenors,
        dates=dates,
        ratio=R,
        fair_value=FV,
        treasury=T,
        muni=M,
        fv_components=comps,
        macro={
            key: v
            for key, v in macro.items()
            if key not in {"treasury", "treasury_base", "hy_base", "move_base"}
        },
        micro={"fund_flows": flows, "net_supply": net_supply, "gross_issuance": gross},
        credit_spreads=spreads,
        tax_rate=tax_rate,
        events={
            "recession": macro["recession"].max(axis=1) > 0,
            "liquidity": liq_any,
            "tax_change": tax_when > 0,
            "exemption_threat": thr_when > 0,
        },
        reference=ref,
        premium=premium,
        report_months=[m for m in report_months if m <= h],
    )
