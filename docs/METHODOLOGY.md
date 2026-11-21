# Methodology

The model simulates `N` monthly paths over a 12-month horizon (configurable)
for every tenor in `tenors` (default 2, 5, 10, 20, 30 years). A path has two
blocks. The **macro block** (rates, economy, risk) does not depend on munis, so
it is simulated first. The **muni block** (ratios, supply, flows, credit) runs
on top of it. All parameters live in `config/default.yaml`; the names below
match the YAML keys.

Notation: `R` = AAA muni / UST ratio, `T` = Treasury yield, `M = R·T` = MMD AAA
yield, `dt = 1/12`, `t` = top marginal tax rate.

---

## 1. Macro block (`muni_mc/macro.py`)

### Recession regime
Two-state Markov chain. The monthly entry hazard is set so that the chance of a
recession starting within 12 months equals `recession.prob_12m` at today's
conditions:

```
h_t = [1 - (1 - P12)^(1/12)] · exp( a_ff (FF_t - FF_mkt,t) + a_hy (HY_t - HY_base,t)/100 )
```

The hazard rises when the Fed is tighter than priced or credit is wider than
its expected path. Recessions end with probability `1/avg_duration_months`
each month.

### Inflation and unemployment
Mean-reverting (OU) processes toward your 12-month targets, with a recession
drift and correlated shocks:

```
π_{t+1} = π_t + κπ(π̄ - π_t)dt + d_rec·rec·dt + σπ √dt ε
```

*Surprises* are measured against the shock-free path: `π_dev = π - π_base`
and `u_dev = u - u_base`.

### Fed funds
```
desired_t = market_path_t + 0.75·π_dev - 1.0·u_dev - recession_cuts_t
```
At FOMC months (and in an emergency during recessions) the Fed moves toward
`desired_t` in 25bp steps, using unbiased stochastic rounding. Steps are
capped at ±50bp per meeting, or ±100bp in a recession. The market path comes
from futures pricing (`fed.market_path`). `market_path_shift_bp` moves the
whole path.

### Rate volatility (MOVE) and risk appetite (HY OAS)
Log-OU processes. MOVE jumps on recession entry, and the HY OAS target moves
to `recession_level` while a recession lasts. MOVE scales every Treasury and
muni shock volatility.

### Treasury curve
```
T_t(τ) = Base_t(τ)                                     expected path (your view)
       + NS(τ)·z_t                                     level / slope / curvature shocks
       + w(τ)·(FF_t - FF_mkt,t)                        Fed surprise, fading with maturity
       + (1 - w(τ))·TP_t                               term premium, long end
```
* `Base_t` glides linearly from today's curve to the horizon curve chosen by
  `path_mode`: `unchanged`, `forwards`, `shifts` (bp by tenor) or `custom`
  (a full curve).
* `z` follows OU with Nelson-Siegel loadings (λ = 0.7308) and vols in
  `factor_vol_bp`, scaled by MOVE/reference. Inflation surprises add to the
  level (`inflation_to_level`). Recession entry adds a flight-to-quality
  level shock.
* `w(τ) = (1 - e^{-τ/τp}) / (τ/τp)` with `τp = ff_passthrough_tau`.
* **Centring** (`center: mean`): after simulation, each tenor and month is
  shifted so the cross-path mean equals `Base_t`. The yield curve you enter is
  therefore the *expected* curve. The macro shocks (recession tail, Fed
  surprises) shape the distribution around it without biasing it.

---

## 2. Muni block (`muni_mc/model.py`)

### Fair value
```
FV = (1 - t)                                       tax-adjusted Treasury
   + b_k + a_k / T                                  rate level
   + β_sup  · supply_z                              net supply
   + β_flow · flows_z                               fund flows      (β_flow < 0)
   + β_move · (MOVE - MOVE_ref)/10                  rate volatility
   + β_hy   · (HY - HY_ref)/100                     risk appetite
   + β_rec  · recession                             recession regime
   + view_k                                         your tilt
```

**Rate level.** Munis carry a semi-sticky *yield* premium over tax-adjusted
Treasuries, so ratios compress when Treasury yields are high and widen when
yields are low. The hyperbola `b_k + a_k/T` is calibrated so that:

1. at the historical average UST yield and historical tax rate, FV equals
   the historical mean ratio; and
2. at today's yield, a +1pp move in Treasuries lowers FV by
   `rate_level_sensitivity` (ratio units).

**Tax.** Raising the top rate by Δt lowers FV by Δt, one for one, because the
breakeven ratio `1 - t` falls. A share `immediate_passthrough` is priced in
on the day of enactment; mean reversion handles the rest.

**Supply.** Gross issuance has three parts: new money × seasonality,
refundings that grow as 10Y MMD falls (`refunding_rate_elasticity`), and
lognormal noise. Reinvestment from coupons, maturities and calls is
seasonal, heaviest in Jun-Aug and Dec-Jan. `supply_z` is an EMA of
`(net supply - reference_net_monthly) / z_scale`. Heavy net supply cheapens
munis, and summer reinvestment richens them.

**Flows.** Mutual fund and ETF flows, in $bn per month:
```
F_t = F̄ + φ(F_{t-1} - F̄) + c_ret·excess_return_{t-1} + c_yld·(M10_t - M10_0)
      + c_move·ΔMOVE + c_hy·ΔHY + c_rec·rec + σ ε + liquidity hits
```
`excess_return = -duration · ΔM10` creates performance chasing. A selloff
causes outflows, outflows cheapen munis, and cheaper munis mean a further
selloff: the Dec-2018, Mar-2020 and 2022 dynamic. `flows_z` is an EMA of
`(F - reference)/z_scale`.

### Ratio dynamics, each month
```
1. lag            R ← (R·T_prev + b_k·ΔT) / T_now     munis move b_k × the UST move
2. tax            R ← R - passthrough·Δt
3. mean reversion R ← R + (1 - e^{-κ dt})(FV - R)     κ = ln2 / half-life
4. retail floor   R ← R + (1 - e^{-k_f dt})·max(0, (1 - t) + offset_k - R)
5. crossover cap  R ← R - (1 - e^{-k_c dt})·max(0, R - ceiling_k)
6. noise          R ← R + σ_k (MOVE/ref)^0.75 √dt η    η correlated across tenors
7. jumps          liquidity events (Poisson, intensity ↑ with MOVE and recession)
                  and tax-exemption headline threats
```
* **Lag.** `short_run_beta` < ratio, so ratios rise in rallies and fall in
  selloffs before mean-reverting. This is a well-documented feature of munis.
* **Retail floor.** When the ratio falls below the tax breakeven, munis stop
  making sense even for top-bracket buyers, so demand fades and the ratio is
  pushed back up.
* **Crossover cap.** At high ratios, banks, insurers and taxable
  (crossover) buyers step in.

### Credit (rated ratios)
Spreads to AAA for AA/A/BBB are `base × tenor_mult × exp(β_rating · c_t)`.
The common factor `c_t` mean-reverts and responds to HY moves, recession
entry and a fiscal-stress drift (federal aid / Medicaid changes). Rated
ratios are `(MMD AAA + spread) / UST`.

---

## 3. Cheap vs expensive (`muni_mc/valuation.py`)

Higher ratio means munis are **cheap**; lower means **expensive**.

| lens | z-score |
|---|---|
| History | `(R - mean_hist) / sd_hist` |
| Model | `(R - FV) / sd_hist` |
| Composite (headline "Signal") | `w_h·z_hist + w_m·z_model` (default 50/50) |

Zones use `bands_z = [-1, -0.5, 0.5, 1]`:

| z | zone |
|---|---|
| < -1 | Very Expensive |
| -1 to -0.5 | Expensive |
| -0.5 to +0.5 | Fair |
| +0.5 to +1 | Cheap |
| > +1 | Very Cheap |

Forecast tables report the share of paths in each zone at 3, 6 and 12
months.

**Tax-bracket view.** Breakeven ratio = `1 - federal rate`. Treasuries are
state-tax exempt, so state tax drops out of the muni-vs-Treasury comparison.
It *does* matter versus corporates. For each bracket the report shows
today's cushion and the probability that the ratio stays above breakeven.

**Attribution.** `E[R_H] - R_0 = Σ_j (E[FV_j,H] - FV_j,0) + (E[gap_H] - gap_0)`
with `gap = R - FV`. This is an exact identity, because expectations add.

---

## 4. Calibration and the November 2026 backdrop

Starting levels are approximations from public commentary, collected
through web search on 2026-11-19. The source pages themselves could not be
opened from the build environment, so treat every starred `(*)` input in
`default.yaml` as a placeholder to refresh from your own data.

| input | value | basis |
|---|---|---|
| Fed funds | 3.75-4.00% | Held 28 Oct 2026 after a Sept hike; two dissents favoured a hike; ~80% odds of a Dec hike priced |
| UST 2Y / 10Y / 30Y | 4.80 / 5.07 / 5.41% | Close around 17-18 Nov 2026 |
| CPI | 3.6% y/y start | Sept 3.7% y/y; Oct +0.4% m/m headline, core +0.3% for the 3rd straight month. No Oct y/y print (missing Oct-2025 base) |
| Unemployment | 4.1% | Oct payrolls +165k |
| GDP | strong | Q3 advance 3.4% annualised |
| MMD AAA 10Y / 30Y | 3.25 / 4.70% | ~18 Nov 2026 |
| Ratios 10Y / 30Y | ~64 / ~87% | Computed from the levels above. Early-Nov prints were 67 / 89% (Nuveen); end-Oct 62/69/90% for 5/10/30Y, "near fair value" (Goldman) |
| Issuance next 12m | $590bn | 2026 is a third straight record year (>$600bn); Barclays 2027 forecast $580-600bn |
| Fund flows | $5bn/month expected | 2026 YTD through Q3: $74bn ($28bn mutual funds, $46bn ETFs); HY funds saw 10 straight weeks of outflows in Oct-Nov |
| Credit | tight | Muni credit spreads below 5-year averages (Goldman 3Q26) |
| Tax | 40.8% | 37% top rate (made permanent July 2025) + 3.8% NIIT |
| History stats | 5y mean/sd | Approximate. **Replace via `valuation.history_csv`** |

Sensitivities and volatilities are judgement-based. They are set so that:

* one-year 10Y ratio changes have a standard deviation of roughly 6-8 pts;
* front-end ratios are the most volatile;
* a hard landing pushes 2Y ratios into the 80s; and
* a 2.6pp tax hike richens ratios by about 2-3 pts.

If you have MMD history, re-estimate the half-lives, vols and betas by
regression. The structure of the model does not change.

## 5. Limitations

* Monthly time steps. Intra-month spikes are smoothed except through jumps.
* Treasury yields are par yields treated as zero yields when computing
  forwards.
* Fed path, recession, MOVE and HY dynamics are reduced-form, not a
  structural macro model.
* The AAA ratio is a benchmark-curve concept. Individual bonds also carry
  coupon, call, state and liquidity effects that this model does not cover.
* History statistics drive the "cheap/expensive" labels. They are only as
  good as the reference window you choose.
