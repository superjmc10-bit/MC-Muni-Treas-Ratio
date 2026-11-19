"""Charts, Excel workbook and a self-contained HTML report."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import valuation as val  # noqa: E402
from .model import SimulationResult  # noqa: E402

# --------------------------------------------------------------------------- #
#  palette (validated default instance; swap for your brand's values)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE = SERIES[0], SERIES[1]
FAN_OUTER, FAN_INNER = "#cde2fb", "#86b6ef"
# Ratio forecasts use violet so they never read as the blue "cheap" zone colour.
FORECAST, FORECAST_TINT = "#6250d6", "#e0dcf7"
# diverging: expensive (red arm) <- fair (gray) -> cheap (blue arm)
ZONE_COLORS = ["#a8302f", "#ee8f8d", "#d3d2cc", "#86b6ef", "#1c5cab"]
ZONE_TEXT = ["#ffffff", INK, INK, INK, "#ffffff"]
UP, DOWN, NEUTRAL = "#2a78d6", "#e34948", "#898781"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "sans-serif",
        "font.size": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
    }
)


def _zone_colors(n_labels: int) -> list[str]:
    if n_labels == len(ZONE_COLORS):
        return ZONE_COLORS
    # fallback for custom band counts: interpolate along the diverging scale
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("zones", ZONE_COLORS)
    return [matplotlib.colors.to_hex(cmap(x)) for x in np.linspace(0, 1, n_labels)]


def _tl(t: float) -> str:
    return f"{t:g}Y"


def _pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))


def _subtitle(fig, text: str, y: float = 0.955):
    fig.text(0.012, y, text, color=INK2, fontsize=9, ha="left", va="top")


def _title(fig, text: str, y: float = 0.995):
    fig.text(0.012, y, text, color=INK, fontsize=13, weight="bold", ha="left", va="top")


def _zone_edges(res: SimulationResult) -> np.ndarray:
    """(n_bands, K) ratio levels (%) of each band edge, from history stats."""
    bands = np.asarray(res.cfg["valuation"]["bands_z"], dtype=float)
    return 100 * (res.reference["mean"][None, :] + bands[:, None] * res.reference["sd"][None, :])


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def fig_snapshot(res: SimulationResult) -> plt.Figure:
    """Hero chart: today's ratio curve vs history zones, model fair value and the forecast."""
    cfg = res.cfg
    labels = cfg["valuation"]["labels"]
    colors = _zone_colors(len(labels))
    x = np.arange(len(res.tenors))
    edges = _zone_edges(res)
    h = res.horizon
    today = 100 * res.ratio[0, 0]
    fv = 100 * res.fair_value[0, 0]
    q = 100 * np.percentile(res.ratio[:, h, :], [5, 25, 50, 75, 95], axis=0)
    zc = val.z_scores(res, "composite")[0, 0]

    lo = min(today.min(), q[0].min(), edges[0].min()) - 6
    hi = max(today.max(), q[4].max(), edges[-1].max()) + 8
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.98)
    # extend the bands half a step past the end tenors so the zone labels sit beside them
    xe = np.concatenate([[x[0] - 0.45], x, [x[-1] + 0.45]])
    pad = lambda a: np.concatenate([[a[0]], a, [a[-1]]])  # noqa: E731
    bounds = [np.full(len(xe), lo)] + [pad(e) for e in edges] + [np.full(len(xe), hi)]
    for j in range(len(labels)):
        ax.fill_between(xe, bounds[j], bounds[j + 1], color=colors[j], alpha=0.16, lw=0, zorder=0)
        ax.text(
            xe[-1] + 0.06, (max(bounds[j][-1], lo) + min(bounds[j + 1][-1], hi)) / 2,
            labels[j], color=INK2, fontsize=8, va="center", ha="left",
        )
    ax.plot(x, 100 * res.reference["mean"], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Historical mean", zorder=2)
    ax.plot(x, fv, color=ORANGE, lw=2, marker="o", ms=5, label="Model fair value (today)", zorder=3)
    off = 0.14
    ax.vlines(x + off, q[0], q[4], color=FORECAST, lw=1.0, alpha=0.7, zorder=3)
    ax.vlines(x + off, q[1], q[3], color=FORECAST, lw=4.0, zorder=3, label=f"{h}m forecast: 25-75% (line 5-95%)")
    ax.plot(x + off, q[2], ls="none", marker="D", ms=7, color=FORECAST, mec=SURFACE, mew=2, zorder=4, label=f"{h}m median")
    ax.plot(x, today, ls="none", marker="o", ms=10, color=INK, mec=SURFACE, mew=2, zorder=5, label="Today")
    for i in range(len(x)):
        lab = val.label_for(zc[i], cfg)
        ax.annotate(
            f"{today[i]:.1f}%\n{lab}", (x[i], today[i]), xytext=(-12, 0), textcoords="offset points",
            ha="right", va="center", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85),
        )
    ax.set_xticks(x, [_tl(t) for t in res.tenors])
    ax.set_xlim(-0.75, x[-1] + 1.35)
    ax.set_ylim(lo, hi)
    _pct_axis(ax)
    ax.set_ylabel("AAA muni yield / UST yield")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.13), ncol=3, handlelength=1.6, columnspacing=1.2)
    _title(fig, "Cheap vs expensive: muni/Treasury ratio curve")
    _subtitle(
        fig,
        f"As of {res.dates[0]:%d %b %Y}. Higher ratio = munis cheaper vs Treasuries. Zones are history bands "
        f"(mean ± {abs(cfg['valuation']['bands_z'][1]):g}/{abs(cfg['valuation']['bands_z'][0]):g} sd); labels use the composite signal.",
    )
    return fig


def fig_fans(res: SimulationResult) -> plt.Figure:
    k = len(res.tenors)
    ncols = 3
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    fig.subplots_adjust(top=0.86 if nrows > 1 else 0.78, hspace=0.42, wspace=0.28, left=0.06, right=0.96, bottom=0.08)
    labels = res.cfg["valuation"]["labels"]
    colors = _zone_colors(len(labels))
    edges = _zone_edges(res)
    d = res.dates
    for i, t in enumerate(res.tenors):
        ax = axes[i // ncols][i % ncols]
        r = 100 * res.ratio[:, :, i]
        p = np.percentile(r, [5, 25, 50, 75, 95], axis=0)
        fvm = 100 * np.median(res.fair_value[:, :, i], axis=0)
        lo, hi = min(p[0].min(), edges[0, i]) - 3, max(p[4].max(), edges[-1, i]) + 3
        b = [lo] + list(edges[:, i]) + [hi]
        for j in range(len(labels)):
            ax.axhspan(b[j], b[j + 1], color=colors[j], alpha=0.16, lw=0, zorder=0)
        ax.fill_between(d, p[0], p[4], color=INK, alpha=0.08, lw=0, zorder=1)
        ax.fill_between(d, p[1], p[3], color=INK, alpha=0.16, lw=0, zorder=2)
        ax.plot(d, p[2], color=FORECAST, lw=2, zorder=3)
        ax.plot(d, fvm, color=ORANGE, lw=1.5, ls=(0, (4, 2)), zorder=3)
        ax.plot(d[0], r[0, 0], marker="o", ms=8, color=INK, mec=SURFACE, mew=2, zorder=4)
        ax.annotate(f"{p[2, -1]:.1f}%", (d[-1], p[2, -1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK, annotation_clip=False)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{_tl(t)} ratio", pad=4)
        _pct_axis(ax)
        _date_axis(ax)
    for j in range(k, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles = [
        Patch(color=INK, alpha=0.10, label="5-95%"),
        Patch(color=INK, alpha=0.22, label="25-75%"),
        plt.Line2D([], [], color=FORECAST, lw=2, label="Median"),
        plt.Line2D([], [], color=ORANGE, lw=1.5, ls=(0, (4, 2)), label="Median fair value"),
        plt.Line2D([], [], color=INK, marker="o", ls="none", ms=7, label="Today"),
    ] + [Patch(color=c, alpha=0.35, label=lab) for c, lab in zip(colors, labels)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955 if nrows > 1 else 0.90),
               ncol=10, handlelength=1.4, columnspacing=1.0, fontsize=8)
    _title(fig, f"Ratio forecast fans, next {res.horizon} months")
    return fig


def fig_zone_probabilities(res: SimulationResult, basis: str = "composite") -> plt.Figure:
    labels = res.cfg["valuation"]["labels"]
    colors = _zone_colors(len(labels))
    months = res.report_months
    fig, axes = plt.subplots(1, len(months), figsize=(4.0 * len(months) + 0.5, 3.4), sharey=True, squeeze=False)
    fig.subplots_adjust(top=0.70, bottom=0.12, left=0.06, right=0.975, wspace=0.10)
    for a, mth in zip(axes[0], months):
        df = val.zone_probabilities(res, mth, basis)
        y = np.arange(len(df))[::-1]
        left = np.zeros(len(df))
        for j, lab in enumerate(labels):
            w = 100 * df[lab].values
            a.barh(y, w, left=left, color=colors[j], edgecolor=SURFACE, linewidth=2, height=0.68)
            for yi, wi, li in zip(y, w, left):
                if wi >= 9:
                    a.text(li + wi / 2, yi, f"{wi:.0f}", ha="center", va="center", fontsize=8, color=ZONE_TEXT[j % 5])
            left += w
        a.set_yticks(y, df.index)
        a.set_xlim(0, 100)
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        a.set_title(f"In {mth} months ({res.dates[mth]:%b-%y})", pad=4)
        a.grid(axis="y", visible=False)
    fig.legend(handles=[Patch(color=c, label=lab) for c, lab in zip(colors, labels)], loc="upper left",
               bbox_to_anchor=(0.01, 0.86), ncol=len(labels), fontsize=8.5)
    _title(fig, "Probability of each valuation zone")
    _subtitle(fig, f"Share of {res.n_paths:,} simulated paths, {basis} signal. Numbers are % of paths.", y=0.93)
    return fig


def fig_distributions(res: SimulationResult) -> plt.Figure:
    """Horizon histograms with bins coloured by the history zone they fall in."""
    k = len(res.tenors)
    labels = res.cfg["valuation"]["labels"]
    colors = _zone_colors(len(labels))
    edges = _zone_edges(res)
    h = res.horizon
    fig, axes = plt.subplots(1, k, figsize=(2.6 * k + 0.6, 3.3), squeeze=False)
    fig.subplots_adjust(top=0.70, bottom=0.14, left=0.04, right=0.99, wspace=0.18)
    for i, a in enumerate(axes[0]):
        r = 100 * res.ratio[:, h, i]
        lo, hi = np.percentile(r, [0.5, 99.5])
        bins = np.linspace(lo, hi, 36)
        counts, b = np.histogram(r, bins=bins)
        mids = 0.5 * (b[1:] + b[:-1])
        zone = np.digitize(mids, edges[:, i])
        a.bar(mids, 100 * counts / counts.sum(), width=(b[1] - b[0]) * 0.92,
              color=[colors[z] for z in zone], edgecolor="none")
        a.axvline(100 * res.ratio[0, 0, i], color=INK, lw=1.5)
        a.axvline(np.median(r), color=FORECAST, lw=1.5, ls=(0, (3, 2)))
        a.set_title(_tl(res.tenors[i]), pad=4)
        a.set_yticks([])
        a.grid(False)
        a.spines["left"].set_visible(False)
        _pct_axis(a, "x")
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    handles = [plt.Line2D([], [], color=INK, lw=1.5, label="Today"),
               plt.Line2D([], [], color=FORECAST, lw=1.5, ls=(0, (3, 2)), label=f"{h}m median")]
    handles += [Patch(color=c, label=lab) for c, lab in zip(colors, labels)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.86), ncol=len(handles), fontsize=8)
    _title(fig, f"Where ratios land in {h} months")
    _subtitle(fig, "Distribution of simulated ratios, bars coloured by history zone.", y=0.93)
    return fig


def fig_attribution(res: SimulationResult, tenor: float | None = None) -> plt.Figure:
    tenor = float(tenor if tenor is not None else res.cfg["valuation"].get("attribution_tenor", 10))
    i = int(np.argmin(np.abs(res.tenors - tenor)))
    row = val.attribution(res).iloc[i]
    steps = [(c, row[c]) for c in row.index if c not in {"Tenor", "Start ratio %", "Expected ratio %"}]
    steps = [(c, v) for c, v in steps if abs(v) >= 0.005]
    start, end = row["Start ratio %"], row["Expected ratio %"]
    names = ["Today"] + [c for c, _ in steps] + [f"Expected in {res.horizon}m"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    fig.subplots_adjust(top=0.82, bottom=0.30, left=0.07, right=0.98)
    level = start
    xs = np.arange(len(names))
    ax.bar(0, start, color=NEUTRAL, width=0.62)
    ax.text(0, start, f"{start:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    for j, (c, v) in enumerate(steps, start=1):
        bottom = level if v >= 0 else level + v
        ax.bar(j, abs(v), bottom=bottom, color=UP if v >= 0 else DOWN, width=0.62)
        ax.text(j, bottom + abs(v), f"{v:+.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
        level += v
    ax.bar(len(names) - 1, end, color=NEUTRAL, width=0.62)
    ax.text(len(names) - 1, end, f"{end:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    lo = min(start, end, level) - 4
    hi = max(start, end) + 3 + max(0, max((v for _, v in steps), default=0))
    ax.set_ylim(max(lo, 0), hi)
    ax.set_xticks(xs, [n.replace(" (", "\n(") for n in names], rotation=30, ha="right")
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=UP, label="Cheapens (ratio up)"), Patch(color=DOWN, label="Richens (ratio down)")],
              loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2)
    _title(fig, f"What moves the {_tl(res.tenors[i])} ratio over the next {res.horizon} months")
    _subtitle(fig, "Expected change in each fair-value driver, plus closing of today's rich/cheap gap (path means).", y=0.93)
    return fig


def _fan(ax, dates, arr, title, unit="", color=BLUE):
    p = np.percentile(arr, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(dates, p[0], p[4], color=FAN_OUTER, lw=0)
    ax.fill_between(dates, p[1], p[3], color=FAN_INNER, lw=0)
    ax.plot(dates, p[2], color=color, lw=2)
    ax.annotate(f"{p[2, -1]:.2f}{unit}" if abs(p[2, -1]) < 20 else f"{p[2, -1]:.0f}{unit}",
                (dates[-1], p[2, -1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK,
                annotation_clip=False)
    ax.set_title(title, pad=4, fontsize=10)
    _date_axis(ax)


def fig_macro(res: SimulationResult) -> plt.Figure:
    d = res.dates
    m, mi = res.macro, res.micro
    t = list(res.tenors)

    def tenor_series(arr, target):
        j = int(np.argmin(np.abs(np.asarray(t) - target)))
        return arr[:, :, j], _tl(t[j])

    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2))
    fig.subplots_adjust(top=0.89, bottom=0.05, left=0.05, right=0.97, hspace=0.45, wspace=0.28)
    ax = axes.ravel()
    _fan(ax[0], d, m["fed_funds"], "Fed funds (mid, %)")
    ax[0].plot(d, m["fed_funds_market"], color=MUTED, lw=1.2, ls=(0, (4, 3)))
    _fan(ax[1], d, m["inflation"], "CPI inflation (y/y, %)")
    _fan(ax[2], d, m["unemployment"], "Unemployment (%)")
    ax[3].plot(d, 100 * m["recession"].mean(axis=0), color=BLUE, lw=2)
    ax[3].set_title("Share of paths in recession (%)", pad=4, fontsize=10)
    _date_axis(ax[3])
    for a, target in zip(ax[4:7], (2, 10, 30)):
        s, lab = tenor_series(res.treasury, target)
        _fan(a, d, s, f"UST {lab} (%)")
    s, lab = tenor_series(res.muni, 10)
    _fan(ax[7], d, s, f"MMD AAA {lab} (%)")
    _fan(ax[8], d, m["move"], "MOVE index")
    _fan(ax[9], d, m["hy_oas"], "HY corporate OAS (bp)")
    _fan(ax[10], d[1:], mi["fund_flows"][:, 1:], "Fund flows ($bn / month)")
    _fan(ax[11], d[1:], mi["net_supply"][:, 1:], "Net supply ($bn / month)")
    for a in ax:
        a.set_xlim(d[0] - pd.Timedelta(days=10), d[-1] + pd.Timedelta(days=10))
    handles = [Patch(color=FAN_OUTER, label="5-95%"), Patch(color=FAN_INNER, label="25-75%"),
               plt.Line2D([], [], color=BLUE, lw=2, label="Median"),
               plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Market-implied Fed path")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=4)
    _title(fig, "Macro and muni-market drivers (simulated)")
    return fig


def fig_tax_breakeven(res: SimulationResult) -> plt.Figure:
    x = np.arange(len(res.tenors))
    h = res.horizon
    q = 100 * np.percentile(res.ratio[:, h, :], [5, 50, 95], axis=0)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.07, right=0.80)
    ax.fill_between(x, q[0], q[2], color=FORECAST_TINT, lw=0, label=f"{h}m range (5-95%)")
    ax.plot(x, q[1], color=FORECAST, lw=2, marker="D", ms=6, mec=SURFACE, mew=1.5, label=f"{h}m median")
    ax.plot(x, 100 * res.ratio[0, 0], color=INK, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="Today")
    for name, rate in res.cfg["tax"]["brackets"].items():
        be = 100 * (1 - float(rate))
        ax.axhline(be, color=MUTED, lw=1, ls=(0, (2, 2)))
        ax.text(x[-1] + 0.15, be, f"{name} bracket: {be:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_xticks(x, [_tl(t) for t in res.tenors])
    ax.set_xlim(-0.3, x[-1] + 0.1)
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3)
    _title(fig, "Who should own munis vs Treasuries? Ratio vs tax breakeven")
    _subtitle(fig, "Above a bracket's line, AAA munis out-yield Treasuries after federal tax for that investor "
                   "(Treasuries are state-tax exempt, so state tax drops out).", y=0.93)
    return fig


def fig_scenarios(results: dict[str, SimulationResult]) -> plt.Figure:
    first = next(iter(results.values()))
    x = np.arange(len(first.tenors))
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.10, left=0.07, right=0.98)
    ax.plot(x, 100 * first.ratio[0, 0], color=INK, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="Today")
    for j, (name, r) in enumerate(results.items()):
        med = 100 * np.median(r.ratio[:, r.horizon, :], axis=0)
        ax.plot(x, med, color=SERIES[j % len(SERIES)], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
                label=r.cfg["meta"].get("scenario_name", name))
    ax.set_xticks(x, [_tl(t) for t in first.tenors])
    _pct_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.2), ncol=3)
    _title(fig, f"Scenario comparison: median ratio in {first.horizon} months")
    return fig


def save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figures(res: SimulationResult, outdir: Path) -> dict[str, Path]:
    outdir = Path(outdir)
    figs = {
        "snapshot": fig_snapshot(res),
        "fans": fig_fans(res),
        "zones": fig_zone_probabilities(res),
        "distributions": fig_distributions(res),
        "attribution": fig_attribution(res),
        "macro": fig_macro(res),
        "tax": fig_tax_breakeven(res),
    }
    return {k: save_fig(f, outdir / f"{i + 1:02d}_{k}.png") for i, (k, f) in enumerate(figs.items())}


# --------------------------------------------------------------------------- #
#  tables for Excel
# --------------------------------------------------------------------------- #
def percentile_table(res: SimulationResult, arr: np.ndarray, name: str, scale: float = 1.0) -> pd.DataFrame:
    """Long table of percentiles by month (and tenor for 3-D arrays)."""
    rows = []
    qs = [5, 25, 50, 75, 95]
    if arr.ndim == 3:
        p = np.percentile(arr, qs, axis=0) * scale
        mean = arr.mean(axis=0) * scale
        for t in range(arr.shape[1]):
            for i, ten in enumerate(res.tenors):
                rows.append({"Series": name, "Month": t, "Date": res.dates[t].date(), "Tenor": _tl(ten),
                             "Mean": mean[t, i], **{f"P{q}": p[j, t, i] for j, q in enumerate(qs)}})
    else:
        p = np.percentile(arr, qs, axis=0) * scale
        mean = arr.mean(axis=0) * scale
        for t in range(arr.shape[1]):
            rows.append({"Series": name, "Month": t, "Date": res.dates[t].date(), "Tenor": "",
                         "Mean": mean[t], **{f"P{q}": p[j, t] for j, q in enumerate(qs)}})
    return pd.DataFrame(rows)


def flatten_config(cfg: dict, prefix: str = "") -> list[tuple[str, str]]:
    out = []
    for k, v in cfg.items():
        key = f"{prefix}.{k:g}" if isinstance(k, float) else f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v and not all(isinstance(x, float) for x in v):
            out += flatten_config(v, key)
        elif isinstance(v, dict):
            out.append((key, ", ".join(f"{kk:g}: {vv}" for kk, vv in v.items()) if v else "{}"))
        else:
            out.append((key, str(v)))
    return out


def rated_table(res: SimulationResult) -> pd.DataFrame:
    rows = []
    h = res.horizon
    for rating in ["AAA"] + list(res.cfg["report"].get("ratings", [])):
        rr = 100 * res.rated_ratio(rating)
        for i, t in enumerate(res.tenors):
            rows.append({"Rating": rating, "Tenor": _tl(t), "Ratio today %": rr[0, 0, i],
                         f"Median {h}m %": np.median(rr[:, h, i]), f"P5 {h}m %": np.percentile(rr[:, h, i], 5),
                         f"P95 {h}m %": np.percentile(rr[:, h, i], 95)})
    return pd.DataFrame(rows)


def write_excel(res: SimulationResult, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    zones = []
    for basis in ("composite", "history", "model"):
        for mth in res.report_months:
            z = val.zone_probabilities(res, mth, basis).reset_index()
            z.insert(0, "Month", mth)
            z.insert(0, "Basis", basis)
            zones.append(z)
    sheets = {
        "Valuation today": val.valuation_today(res),
        "Forecast": val.forecast_summary(res),
        "Zone probabilities": pd.concat(zones, ignore_index=True),
        "Attribution": val.attribution(res),
        "Fair value today": val.fair_value_breakdown(res, 0).reset_index(),
        "Tax breakeven": val.tax_breakeven_table(res),
        "Rated ratios": rated_table(res),
        "Events": val.event_summary(res),
        "Ratio paths": percentile_table(res, res.ratio, "AAA ratio %", 100),
        "Treasury paths": percentile_table(res, res.treasury, "UST %"),
        "MMD paths": percentile_table(res, res.muni, "MMD AAA %"),
        "Macro paths": pd.concat(
            [percentile_table(res, res.macro[k], k) for k in ("fed_funds", "inflation", "unemployment", "move", "hy_oas")]
            + [percentile_table(res, res.micro[k], k) for k in ("fund_flows", "net_supply", "gross_issuance")],
            ignore_index=True,
        ),
        "Inputs": pd.DataFrame(flatten_config(res.cfg), columns=["Parameter", "Value"]),
    }
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
                for c in col[1:]:
                    if isinstance(c.value, float):
                        c.number_format = "0.000" if abs(c.value) < 1.5 else "0.00"
            ws.freeze_panes = "A2"
    return path


# --------------------------------------------------------------------------- #
#  HTML report
# --------------------------------------------------------------------------- #
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--ring:rgba(11,11,11,.10)}
*{box-sizing:border-box}body{margin:0;background:var(--plane);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}h3{font-size:15px;margin:20px 0 6px}
p.lead{color:var(--ink2);margin:0 0 16px;max-width:860px}.muted{color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.card .k{font-size:12px;color:var(--ink2)}.card .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.card .s{font-size:12px;margin-top:2px}
figure{margin:12px 0;background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:8px}
figure img{width:100%;height:auto;display:block}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--ring);border-radius:10px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 9px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;background:var(--plane)}td:first-child,th:first-child{text-align:left}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.note{background:var(--surface);border:1px solid var(--ring);border-left:3px solid #eb6834;border-radius:8px;padding:10px 14px;color:var(--ink2)}
ul.tight li{margin:2px 0}
"""


def _chip(label: str, labels: Sequence[str]) -> str:
    colors = _zone_colors(len(labels))
    j = list(labels).index(label) if label in labels else len(labels) // 2
    fg = ZONE_TEXT[j] if len(labels) == 5 else INK
    return f'<span class="chip" style="background:{colors[j]};color:{fg}">{html.escape(label)}</span>'


def _img(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{data}"></figure>'


def _table(df: pd.DataFrame, labels: Sequence[str] = (), pct_cols: Sequence[str] = (), digits: int = 2) -> str:
    out = df.copy()
    for c in out.columns:
        if c in pct_cols:
            out[c] = out[c].map(lambda v: f"{100 * v:.0f}%")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:,.{digits}f}")
        elif labels and out[c].isin(labels).all():
            out[c] = out[c].map(lambda v: _chip(v, labels))
    return '<div class="tbl">' + out.to_html(index=False, escape=False, border=0) + "</div>"


def write_html(res: SimulationResult, figs: dict[str, Path], path: Path, extra_sections: str = "") -> Path:
    cfg = res.cfg
    labels = cfg["valuation"]["labels"]
    h = res.horizon
    today = val.valuation_today(res)
    fc = val.forecast_summary(res)
    fc_h = fc[fc["Month"] == h]
    meta = cfg["meta"]
    name = meta.get("scenario_name", "Scenario")

    cards = []
    for i, t in enumerate(res.tenors):
        row = today.iloc[i]
        frow = fc_h.iloc[i]
        cards.append(
            f'<div class="card"><div class="k">{_tl(t)} AAA ratio</div><div class="v">{row["Ratio %"]:.1f}%</div>'
            f'<div class="s">{_chip(row["Signal"], labels)}</div>'
            f'<div class="s muted">{h}m median {frow["Median %"]:.1f}% '
            f'({frow["P5 %"]:.0f}-{frow["P95 %"]:.0f}%)</div></div>'
        )
    ev = val.event_summary(res)

    today_cols = ["Tenor", "UST %", "MMD AAA %", "Ratio %", "Hist mean %", "z vs history", "History view",
                  "Model fair value %", "Gap to FV (pts)", "Model view", "Composite z", "Signal"]
    fc_cols = ["Tenor", "Start %", "P5 %", "P25 %", "Median %", "P75 %", "P95 %", "Mean %",
               "P(ratio higher)", "P(cheap or better)", "P(expensive or worse)"]
    att = val.attribution(res)
    fvb = val.fair_value_breakdown(res, 0).reset_index()
    tax_tbl = val.tax_breakeven_table(res)
    pcols = [c for c in tax_tbl.columns if c.startswith(tuple(_tl(t) for t in res.tenors)) and "P(" in c]

    key_inputs = [
        ("UST 2Y / 10Y / 30Y today", " / ".join(f"{res.treasury[0, 0, int(np.argmin(abs(res.tenors - x)))]:.2f}%" for x in (2, 10, 30))),
        ("Curve path", f"{cfg['treasury']['path_mode']}" + (f" {cfg['treasury'].get('shifts_bp')}" if cfg['treasury']['path_mode'] == 'shifts' else "")),
        ("Fed funds today / market path end", f"{cfg['macro']['fed']['start_mid']:.3f}% / {res.macro['fed_funds_market'][-1]:.3f}%"),
        ("Recession probability (12m input)", f"{100 * cfg['macro']['recession']['prob_12m']:.0f}%"),
        ("Inflation start → target", f"{cfg['macro']['inflation']['start']}% → {cfg['macro']['inflation']['target_12m']}%"),
        ("Gross issuance / reinvestment ($bn)", f"{cfg['supply']['annual_gross_issuance']:.0f} / {cfg['supply']['annual_reinvestment']:.0f}"),
        ("Fund flows expected ($bn / month)", f"{cfg['flows']['expected']}"),
        ("Top marginal tax rate", f"{100 * cfg['tax']['marginal_rate']:.1f}%"),
        ("Valuation reference", res.reference["source"]),
        ("Paths / horizon / seed", f"{res.n_paths:,} / {h} months / {cfg['simulation'].get('seed')}"),
    ]
    inputs_html = "<div class='tbl'><table>" + "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in key_inputs
    ) + "</table></div>"

    body = f"""
<main>
<h1>Muni / Treasury ratio Monte Carlo: {html.escape(name)}</h1>
<p class="lead">As of {res.dates[0]:%d %B %Y}. {html.escape(str(meta.get('description', '')).strip())}</p>
<div class="note"><b>How to read this.</b> Ratio = AAA muni yield ÷ Treasury yield at the same maturity.
<b>Higher ratio = munis CHEAP</b> (more yield vs Treasuries); <b>lower ratio = munis EXPENSIVE</b> (rich).
"History" compares with the reference window; "Model" compares with a fair value built from rates, tax,
supply, flows, volatility, credit and recession risk; the <b>Signal</b> blends both.</div>
<div class="cards">{''.join(cards)}</div>

<h2>Cheap vs expensive today</h2>
{_img(figs['snapshot'])}
{_table(today[today_cols], labels)}

<h2>{h}-month forecast</h2>
{_img(figs['fans'])}
{_table(fc_h[fc_cols], labels, pct_cols=["P(ratio higher)", "P(cheap or better)", "P(expensive or worse)"])}
{_img(figs['zones'])}
{_img(figs['distributions'])}

<h2>What drives the forecast</h2>
{_img(figs['attribution'])}
<h3>Bridge from today's ratio to the expected ratio (ratio points)</h3>
{_table(att)}
<h3>Fair value today, by component (ratio points)</h3>
{_table(fvb)}

<h2>Macro and market technicals</h2>
{_img(figs['macro'])}
{_table(ev, pct_cols=["Probability"])}

<h2>Who should own munis?</h2>
{_img(figs['tax'])}
{_table(tax_tbl, pct_cols=pcols)}

{extra_sections}

<h2>Key inputs</h2>
{inputs_html}
<p class="muted">Full parameter list: the "Inputs" sheet of the Excel workbook. Model equations: docs/METHODOLOGY.md.
Starting levels are approximations from public mid-November 2026 commentary. Replace them with your own
MMD / Bloomberg / Treasury data before relying on the output. This is a scenario tool, not investment advice.</p>
</main>"""
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Muni Ratio Monte Carlo</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


def scenario_comparison_table(results: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        fc = val.forecast_summary(r)
        fc = fc[fc["Month"] == r.horizon]
        row = {"Scenario": r.cfg["meta"].get("scenario_name", name)}
        for _, f in fc.iterrows():
            row[f"{f['Tenor']} median %"] = f["Median %"]
        for _, f in fc.iterrows():
            row[f"{f['Tenor']} P(cheap+)"] = f["P(cheap or better)"]
        i10 = int(np.argmin(np.abs(r.tenors - 10)))
        row["UST 10Y median %"] = np.median(r.treasury[:, r.horizon, i10])
        rows.append(row)
    return pd.DataFrame(rows)


def buffer_png(fig: plt.Figure) -> bytes:
    """PNG bytes of a figure (used by the Streamlit app)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
