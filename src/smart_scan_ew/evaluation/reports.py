"""Report generator: Matplotlib figures + HTML report."""

from __future__ import annotations

import base64
import warnings
import io
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
    """Bar chart of interception rate per scheduler."""
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
    means = [np.nanmean([m.mean_revisit_interval for m in v]) for v in metrics_by_scheduler.values()]
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
  {summary_table}
  <h2>Detection &amp; False-Alarm Rate</h2>
  <img src="data:image/png;base64,{fig_pd_pfa}" alt="P_D / P_FA chart">
  <h2>Interception Rate</h2>
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


def _build_summary_table(
    metrics_by_scheduler: dict[str, list[EpisodeMetrics]]
) -> str:
    headers = [
        "Scheduler", "Pd", "Pfa", "Interception Rate",
        "Mean Intercept Time", "P95 Intercept Time",
        "Coverage", "Mean Revisit", "Cum. Reward",
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

    def cell(val: float, best: float, lower_is_better: bool = False) -> str:
        if val != val:  # nan
            return "—"
        badge = ""
        if lower_is_better and abs(val - best) < 1e-6:
            badge = ' <span class="badge best">best</span>'
        elif not lower_is_better and abs(val - best) < 1e-6:
            badge = ' <span class="badge best">best</span>'
        return f"{val:.3f}{badge}"

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for name, agg in rows_data:
        ti = agg.get("mean_intercept_time_mean", float("nan"))
        rows_html += (
            f"<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{cell(agg.get('pd_mean', float('nan')), best_pd)}</td>"
            f"<td>{agg.get('pfa_mean', float('nan')):.3f}</td>"
            f"<td>{cell(agg.get('interception_rate_mean', float('nan')), best_ir)}</td>"
            f"<td>{cell(ti, best_ti, lower_is_better=True)}</td>"
            f"<td>{agg.get('p95_intercept_time_mean', float('nan')):.1f}</td>"
            f"<td>{agg.get('coverage_ratio_mean', float('nan')):.3f}</td>"
            f"<td>{agg.get('mean_revisit_interval_mean', float('nan')):.1f}</td>"
            f"<td>{agg.get('cumulative_reward_mean', float('nan')):.1f}</td>"
            f"</tr>\n"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table>"


def generate_html_report(
    all_metrics: list[EpisodeMetrics],
    output_path: str | Path,
    subtitle: str = "Adaptive ES Receiver Scheduler Evaluation",
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
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by scheduler
    by_sched: dict[str, list[EpisodeMetrics]] = {}
    for m in all_metrics:
        by_sched.setdefault(m.scheduler_name, []).append(m)

    # Generate figures
    fig_pd_pfa = plot_pd_pfa_bar(by_sched)
    fig_ir = plot_interception_rate_bar(by_sched)
    fig_cdf = plot_intercept_time_cdf(by_sched)
    fig_reward = plot_reward_comparison(by_sched)
    fig_revisit = plot_revisit_interval(by_sched)
    summary_table = _build_summary_table(by_sched)

    import datetime
    html = _HTML_TEMPLATE.format(
        subtitle=f"{subtitle} — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        summary_table=summary_table,
        fig_pd_pfa=fig_pd_pfa,
        fig_ir=fig_ir,
        fig_cdf=fig_cdf,
        fig_reward=fig_reward,
        fig_revisit=fig_revisit,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
