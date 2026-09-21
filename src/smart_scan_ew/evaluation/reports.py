"""Report generator: Matplotlib figures + HTML report."""

from __future__ import annotations

import base64
import io
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from smart_scan_ew.evaluation.metrics import EpisodeMetrics, aggregate_metrics

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def plot_pd_pfa_bar(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    """Bar chart of mean Pd and Pfa per scheduler."""
    schedulers = list(metrics_by_scheduler.keys())
    pds = [np.nanmean([m.pd for m in v]) for v in metrics_by_scheduler.values()]
    pfas = [np.nanmean([m.pfa for m in v]) for v in metrics_by_scheduler.values()]

    x = np.arange(len(schedulers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, pds, w, label="P_D", color="#2196F3")
    ax.bar(x + w / 2, pfas, w, label="P_FA", color="#F44336")
    ax.set_xticks(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ax.set_xticklabels(schedulers, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_title("Detection and False-Alarm Rate by Scheduler")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_intercept_time_cdf(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    """CDF of interception times per scheduler."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, (name, metrics_list) in enumerate(metrics_by_scheduler.items()):
        times: list[float] = []
        for m in metrics_list:
            times.extend(m.intercept_times)
        if not times:
            continue
        t_sorted = np.sort(times)
        cdf = np.arange(1, len(t_sorted) + 1) / len(t_sorted)
        ax.plot(t_sorted, cdf, label=name, color=colors[i % 10], lw=2)
    ax.set_xlabel("Intercept time (slots)")
    ax.set_ylabel("CDF")
    ax.set_title("Interception Time CDF")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_interception_rate_bar(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    """Bar chart of interception rate per scheduler (mean only)."""
    schedulers = list(metrics_by_scheduler.keys())
    rates = [np.nanmean([m.interception_rate for m in v]) for v in metrics_by_scheduler.values()]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(schedulers)), rates, color="#4CAF50", edgecolor="white")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("Interception Rate")
    ax.set_title("Fraction of Emitters Intercepted per Scheduler")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(len(schedulers)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ax.set_xticklabels(schedulers, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_interception_rate_with_std(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    """Bar chart of avg interception rate with ±1 std error bars (across seeds)."""
    schedulers = list(metrics_by_scheduler.keys())
    means, stds = [], []
    for v in metrics_by_scheduler.values():
        vals = [m.interception_rate for m in v]
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals)) if len(vals) > 1 else 0.0)

    x = np.arange(len(schedulers))
    colors = ["#4CAF50" if m == max(means) else "#388E3C" for m in means]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="white",
        error_kw={"ecolor": "#FFEB3B", "lw": 1.5},
    )
    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.3f}\n±{s:.3f}" if s > 0 else f"{m:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05 + 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#E0E0E0",
        )
    ax.set_ylabel("Avg Interception Rate (mean ± std across seeds)")
    ax.set_title("Interception Rate — Consistency Across Seeds")
    ax.set_ylim(0, min(1.15, max(means) + max(stds) + 0.15))
    ax.set_xticks(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ax.set_xticklabels(schedulers, rotation=30, ha="right")
    ax.axhline(
        np.nanmean(means),
        color="#FF5722",
        linestyle="--",
        lw=1,
        label=f"Overall mean = {np.nanmean(means):.3f}",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_reward_comparison(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    """Box plot of cumulative rewards per scheduler."""
    fig, ax = plt.subplots(figsize=(10, 4))
    data = [[m.cumulative_reward for m in v] for v in metrics_by_scheduler.values()]
    labels = list(metrics_by_scheduler.keys())
    # 'labels' kwarg was renamed to 'tick_labels' in Matplotlib ≥ 3.9;
    # fall back to setting ticks manually for compatibility with both.
    bp = ax.boxplot(data, patch_artist=True)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Scheduler Reward Distribution")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_revisit_interval(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    """Mean revisit interval per scheduler."""
    schedulers = list(metrics_by_scheduler.keys())
    means = [
        np.nanmean([m.mean_revisit_interval for m in v]) for v in metrics_by_scheduler.values()
    ]
    maxes = [np.nanmean([m.max_revisit_interval for m in v]) for v in metrics_by_scheduler.values()]
    x = np.arange(len(schedulers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, means, w, label="Mean", color="#9C27B0")
    ax.bar(x + w / 2, maxes, w, label="Max", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Revisit interval (slots)")
    ax.set_title("Band Revisit Intervals")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Scan EW — Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; margin: 0; padding: 0; }}
  header {{ background: linear-gradient(135deg,#1f6feb,#388bfd);
            padding: 2rem; text-align: center; }}
  header h1 {{ margin: 0; font-size: 2rem; color: #fff; }}
  header p  {{ margin: 0.5rem 0 0; color: rgba(255,255,255,0.8); }}
  section  {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
  h2 {{ border-bottom: 1px solid #30363d; padding-bottom: 0.4rem; color: #58a6ff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin: 1rem 0; }}
  th {{ background: #161b22; color: #58a6ff; padding: 8px 10px; text-align: left;
        border: 1px solid #30363d; }}
  td {{ padding: 6px 10px; border: 1px solid #21262d; }}
  tr:nth-child(even) {{ background: #161b22; }}
  img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0;
         box-shadow: 0 4px 20px rgba(0,0,0,0.5); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; }}
  .best {{ background: #1a7f37; color: #fff; }}
  .note {{ font-size: 0.8rem; color: #8b949e; margin: 0.25rem 0 1rem; }}
  footer {{ text-align: center; color: #484f58; padding: 2rem; font-size: 0.8rem; }}
</style>
</head>
<body>
<header>
  <h1>⚡ Smart Scan EW — Benchmark Report</h1>
  <p>{subtitle}</p>
</header>
<section>
  <h2>Summary Table</h2>
  <p class="note">Metrics shown as mean ± std across {n_seeds} independent seeds. <span class="badge best">best</span> highlights the top scheduler per column.</p>
  {summary_table}
  <h2>Detection &amp; False-Alarm Rate</h2>
  <img src="data:image/png;base64,{fig_pd_pfa}" alt="P_D / P_FA chart">
  <h2>Interception Rate (mean ± std across seeds)</h2>
  <img src="data:image/png;base64,{fig_ir_std}" alt="Interception rate with std error bars">
  <h2>Interception Rate (per scheduler)</h2>
  <img src="data:image/png;base64,{fig_ir}" alt="Interception rate chart">
  <h2>Intercept Time CDF</h2>
  <img src="data:image/png;base64,{fig_cdf}" alt="Intercept time CDF">
  <h2>Cumulative Reward Distribution</h2>
  <img src="data:image/png;base64,{fig_reward}" alt="Reward boxplot">
  <h2>Band Revisit Intervals</h2>
  <img src="data:image/png;base64,{fig_revisit}" alt="Revisit interval chart">
</section>
<footer>
  Generated by Smart Scan EW evaluation pipeline &mdash;
  simulation only, authorised defensive research.
</footer>
</body>
</html>
"""


def _build_summary_table(metrics_by_scheduler: dict[str, list[EpisodeMetrics]]) -> str:
    headers = [
        "Scheduler",
        "Pd (mean±std)",
        "Pfa",
        "Interception Rate (mean±std)",
        "Mean Intercept Time (mean±std)",
        "P95 Intercept Time",
        "Coverage",
        "Mean Revisit",
        "Cum. Reward",
    ]
    rows_data: list[tuple[str, dict]] = []
    for name, mlist in metrics_by_scheduler.items():
        agg = aggregate_metrics(mlist)
        rows_data.append((name, agg))

    # Find best values for highlighting
    best_pd = max(d.get("pd_mean", 0) for _, d in rows_data)
    best_ir = max(d.get("interception_rate_mean", 0) for _, d in rows_data)
    best_ti = min(
        (d.get("mean_intercept_time_mean", float("inf")) for _, d in rows_data),
        default=float("inf"),
    )

    def mean_std_cell(mean: float, std: float, best: float, lower_is_better: bool = False) -> str:
        """Format a cell as 'mean ± std' with optional best badge."""
        if mean != mean:  # nan
            return "—"
        badge = ""
        is_best = (lower_is_better and abs(mean - best) < 1e-6) or (
            not lower_is_better and abs(mean - best) < 1e-6
        )
        if is_best:
            badge = ' <span class="badge best">best</span>'
        std_str = f" ±{std:.3f}" if std == std and std > 0 else ""
        return f"{mean:.3f}{std_str}{badge}"

    def plain_cell(val: float, fmt: str = ".3f") -> str:
        return "—" if val != val else format(val, fmt)

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for name, agg in rows_data:
        ti_mean = agg.get("mean_intercept_time_mean", float("nan"))
        ti_std = agg.get("mean_intercept_time_std", float("nan"))
        pd_mean = agg.get("pd_mean", float("nan"))
        pd_std = agg.get("pd_std", float("nan"))
        ir_mean = agg.get("interception_rate_mean", float("nan"))
        ir_std = agg.get("interception_rate_std", float("nan"))
        rows_html += (
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{mean_std_cell(pd_mean, pd_std, best_pd)}</td>"
            f"<td>{plain_cell(agg.get('pfa_mean', float('nan')))}</td>"
            f"<td>{mean_std_cell(ir_mean, ir_std, best_ir)}</td>"
            f"<td>{mean_std_cell(ti_mean, ti_std, best_ti, lower_is_better=True)}</td>"
            f"<td>{plain_cell(agg.get('p95_intercept_time_mean', float('nan')), '.1f')}</td>"
            f"<td>{plain_cell(agg.get('coverage_ratio_mean', float('nan')))}</td>"
            f"<td>{plain_cell(agg.get('mean_revisit_interval_mean', float('nan')), '.1f')}</td>"
            f"<td>{plain_cell(agg.get('cumulative_reward_mean', float('nan')), '.1f')}</td>"
            f"</tr>\n"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"


def generate_html_report(
    all_metrics: list[EpisodeMetrics],
    output_path: str | Path,
    subtitle: str = "Adaptive ES Receiver Scheduler Evaluation",
    n_seeds: int | None = None,
) -> Path:
    """Build and save the full HTML report.

    Parameters
    ----------
    all_metrics:
        Flat list of EpisodeMetrics from ``run_benchmark``.
    output_path:
        Path to write the HTML file.
    subtitle:
        Subtitle shown in the header.
    n_seeds:
        Number of seeds used; displayed in the report header note.
        Auto-inferred from data if not provided.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by scheduler
    by_sched: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_sched.setdefault(m.scheduler_name, []).append(m)

    # Auto-infer n_seeds if not provided
    if n_seeds is None:
        seeds_seen = {m.seed for m in all_metrics}
        n_seeds = len(seeds_seen)

    # Generate figures
    fig_pd_pfa = plot_pd_pfa_bar(by_sched)
    fig_ir = plot_interception_rate_bar(by_sched)
    fig_ir_std = plot_interception_rate_with_std(by_sched)
    fig_cdf = plot_intercept_time_cdf(by_sched)
    fig_reward = plot_reward_comparison(by_sched)
    fig_revisit = plot_revisit_interval(by_sched)
    summary_table = _build_summary_table(by_sched)

    import datetime

    html = _HTML_TEMPLATE.format(
        subtitle=f"{subtitle} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        n_seeds=n_seeds,
        summary_table=summary_table,
        fig_pd_pfa=fig_pd_pfa,
        fig_ir=fig_ir,
        fig_ir_std=fig_ir_std,
        fig_cdf=fig_cdf,
        fig_reward=fig_reward,
        fig_revisit=fig_revisit,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
