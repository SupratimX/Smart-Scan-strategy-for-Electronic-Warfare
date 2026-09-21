"""train.py — Multi-episode offline training for the Thompson Sampling scheduler.

Simulates hundreds of episodes across all scenarios, accumulates the
Bayesian belief (alpha/beta counts) from each episode into a persistent
model, then saves the trained weights for deployment.

The "training" loop works as follows:
  - For each training episode, the scheduler runs an full episode on a
    randomly sampled scenario with a unique seed.
  - After each episode, its learned alpha/beta posteriors are merged into
    a running global model using a weighted average (new episodes count less
    so the model stabilises over time).
  - After all training episodes the final weights are saved to
    artifacts/models/trained_thompson.npz

Usage
-----
    python scripts/train.py                          # default 200 episodes
    python scripts/train.py --episodes 500 --verbose
    python scripts/train.py --episodes 200 --eval    # train then evaluate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smart_scan_ew.config import ReceiverConfig, load_scenario_config
from smart_scan_ew.evaluation.reports import generate_html_report
from smart_scan_ew.evaluation.runner import EpisodeRunner
from smart_scan_ew.schedulers.bandit import ThompsonSamplingScheduler
from smart_scan_ew.schedulers.registry import SCHEDULER_REGISTRY

# ---------------------------------------------------------------------------
# Trainable scheduler — extends Thompson Sampling with load/save weights
# ---------------------------------------------------------------------------


class TrainableThompsonScheduler(ThompsonSamplingScheduler):
    """Thompson Sampling scheduler with persistent weight save/load."""

    name = "trained_thompson"

    def save_weights(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, alpha=self._alpha, beta=self._beta)
        print(f"  Weights saved -> {path}")

    def load_weights(self, path: Path) -> None:
        data = np.load(path)
        self._alpha = data["alpha"].copy()
        self._beta = data["beta"].copy()
        print(f"  Weights loaded <- {path}")

    def merge_weights(
        self, other_alpha: np.ndarray, other_beta: np.ndarray, decay: float = 0.95
    ) -> None:
        """Merge weights from a finished episode into the global model.

        Uses exponential moving average so early episodes count less
        than recent ones, helping the model converge.
        """
        self._alpha = decay * self._alpha + (1 - decay) * other_alpha
        self._beta = decay * self._beta + (1 - decay) * other_beta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    n_episodes: int,
    scenarios: list[Path],
    receiver_cfg: ReceiverConfig,
    weights_path: Path,
    verbose: bool = True,
) -> TrainableThompsonScheduler:
    """Run multi-episode training and return the trained scheduler."""

    global_model = TrainableThompsonScheduler(receiver_cfg, seed=0)

    # Load existing weights if available (resume training)
    if weights_path.exists():
        global_model.load_weights(weights_path)
        print(f"  Resuming from existing weights at {weights_path}")

    rng = np.random.default_rng(42)
    t0 = time.time()

    print(f"\n  Training over {n_episodes} episodes across {len(scenarios)} scenarios...")
    print(
        f"  {'Episode':>8} | {'Scenario':30} | {'IR':>6} | {'Pd':>6} | {'T_int':>8} | {'alpha_mean':>10}"
    )
    print(f"  {'-' * 80}")

    all_ir, all_ti = [], []

    for ep in range(n_episodes):
        # Pick a random scenario and seed each episode
        scenario_path = scenarios[rng.integers(len(scenarios))]
        seed = int(rng.integers(0, 100_000))

        scenario_cfg = load_scenario_config(scenario_path)

        # Episode scheduler — starts from the global model's current weights
        ep_scheduler = TrainableThompsonScheduler(receiver_cfg, seed=seed)
        ep_scheduler._alpha = global_model._alpha.copy()
        ep_scheduler._beta = global_model._beta.copy()

        runner = EpisodeRunner(
            scenario_cfg=scenario_cfg,
            receiver_cfg=receiver_cfg,
            scheduler=ep_scheduler,
            seed=seed,
        )
        _, metrics = runner.run()

        # Merge episode weights back into global model
        # Decay rate decreases as training progresses (later episodes matter more)
        decay = 0.98 - 0.3 * (ep / n_episodes)  # 0.98 → 0.68 over training
        global_model.merge_weights(ep_scheduler._alpha, ep_scheduler._beta, decay=decay)

        all_ir.append(metrics.interception_rate)
        if metrics.mean_intercept_time == metrics.mean_intercept_time:
            all_ti.append(metrics.mean_intercept_time)

        if verbose and (ep % max(1, n_episodes // 20) == 0 or ep == n_episodes - 1):
            print(
                f"  {ep + 1:>8} | {scenario_cfg.scenario_id:30} | "
                f"{metrics.interception_rate:>6.3f} | {metrics.pd:>6.3f} | "
                f"{metrics.mean_intercept_time:>8.1f} | "
                f"{global_model._alpha.mean():>10.2f}"
            )

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(
        f"  Avg IR  : {np.mean(all_ir):.3f}  (final 20%: {np.mean(all_ir[-n_episodes // 5 :]):.3f})"
    )
    if all_ti:
        print(
            f"  Avg T_int: {np.mean(all_ti):.1f}  (final 20%: {np.mean(all_ti[-n_episodes // 5 :]):.1f}"
        )

    global_model.save_weights(weights_path)
    return global_model


# ---------------------------------------------------------------------------
# Evaluation of trained vs untrained
# ---------------------------------------------------------------------------


def evaluate_trained_vs_baseline(
    trained: TrainableThompsonScheduler,
    weights_path: Path,
    scenarios: list[Path],
    receiver_cfg: ReceiverConfig,
    n_seeds: int = 5,
    output_dir: Path = ROOT / "artifacts" / "results" / "training",
) -> None:
    """Compare trained Thompson vs all baseline schedulers."""
    from smart_scan_ew.evaluation.metrics import EpisodeMetrics

    print(f"\n{'=' * 60}")
    print("  Evaluation: Trained vs Baseline Schedulers")
    print(f"{'=' * 60}\n")

    all_metrics: list[EpisodeMetrics] = []

    # Evaluate trained scheduler
    for scenario_path in scenarios:
        scenario_cfg = load_scenario_config(scenario_path)
        for seed_offset in range(n_seeds):
            seed = 200 + seed_offset
            sched = TrainableThompsonScheduler(receiver_cfg, seed=seed)
            sched.load_weights(weights_path)
            runner = EpisodeRunner(scenario_cfg, receiver_cfg, sched, seed=seed)
            _, m = runner.run()
            all_metrics.append(m)
            print(
                f"  trained_thompson          | {scenario_cfg.scenario_id:25} | "
                f"seed={seed} | IR={m.interception_rate:.3f} T_int={m.mean_intercept_time:.1f}"
            )

    # Evaluate baselines
    for sched_name, SchedulerClass in SCHEDULER_REGISTRY.items():
        for scenario_path in scenarios:
            scenario_cfg = load_scenario_config(scenario_path)
            for seed_offset in range(n_seeds):
                seed = 200 + seed_offset
                sched = SchedulerClass(receiver_cfg, seed=seed)
                runner = EpisodeRunner(scenario_cfg, receiver_cfg, sched, seed=seed)
                _, m = runner.run()
                all_metrics.append(m)

    # Print comparison table
    by_sched: dict[str, list] = {}
    for m in all_metrics:
        by_sched.setdefault(m.scheduler_name, []).append(m)

    print(f"\n  {'Scheduler':30} | {'avg IR':>8} | {'avg T_int':>10} | {'avg Pd':>8}")
    print(f"  {'-' * 65}")
    for name, mlist in sorted(
        by_sched.items(), key=lambda x: -np.mean([m.interception_rate for m in x[1]])
    ):
        ir = np.mean([m.interception_rate for m in mlist])
        pd = np.mean([m.pd for m in mlist])
        ti_v = [
            m.mean_intercept_time for m in mlist if m.mean_intercept_time == m.mean_intercept_time
        ]
        ti = np.mean(ti_v) if ti_v else float("nan")
        marker = "  <-- TRAINED" if name == "trained_thompson" else ""
        print(f"  {name:30} | {ir:>8.3f} | {ti:>10.1f} | {pd:>8.3f}{marker}")

    # Save HTML report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "training_eval.html"
    generate_html_report(
        all_metrics,
        output_path=report_path,
        subtitle="Trained Thompson vs Baselines",
        n_seeds=n_seeds,
    )
    print(f"\n  Report -> {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Thompson Sampling EW scheduler")
    p.add_argument(
        "--episodes", type=int, default=200, help="Number of training episodes (default: 200)"
    )
    p.add_argument("--eval", action="store_true", help="Run evaluation after training")
    p.add_argument("--eval-seeds", type=int, default=5)
    p.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "artifacts" / "models" / "trained_thompson.npz",
        help="Path to save/load model weights",
    )
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # All available scenarios for training
    scenario_dir = ROOT / "configs" / "scenarios"
    scenarios = [
        p
        for p in scenario_dir.glob("*.yaml")
        if "empty" not in p.name  # skip empty spectrum — nothing to learn
    ]
    print(f"\n  Scenarios for training ({len(scenarios)}):")
    for s in scenarios:
        print(f"    - {s.name}")

    receiver_cfg = ReceiverConfig()

    print(f"\n{'=' * 60}")
    print(f"  PHASE 1: Training ({args.episodes} episodes)")
    print(f"{'=' * 60}")

    trained = train(
        n_episodes=args.episodes,
        scenarios=scenarios,
        receiver_cfg=receiver_cfg,
        weights_path=args.weights,
        verbose=args.verbose,
    )

    if args.eval:
        report_path = evaluate_trained_vs_baseline(
            trained=trained,
            weights_path=args.weights,
            scenarios=scenarios,
            receiver_cfg=receiver_cfg,
            n_seeds=args.eval_seeds,
        )
        import subprocess

        subprocess.Popen(["start", "", str(report_path)], shell=True)

    print(f"\n{'=' * 60}")
    print(f"  Done! Weights at: {args.weights}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
