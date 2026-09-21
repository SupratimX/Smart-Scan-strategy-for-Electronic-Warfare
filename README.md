# Smart Scan EW

**ML-based Electronic Support (ES) receiver scheduler** for adaptively scanning frequency bands
to minimise interception time while maintaining high detection and low false-alarm rates.

> **Scope boundary:** Simulation and authorised defensive ES research only.
> Must not be used to control offensive RF effects.

---

## Quick Start

### 1. Install Backend

```bash
cd smart-scan-ew/backend
pip install -e ".[dev]"
```

### 2. Run the Web Frontend

```bash
# Serve the frontend locally:
cd ../frontend
npx serve .
```

### 3. One-command acceptance test (generates data → evaluates all schedulers → plots → report)

```bash
cd ../backend
python scripts/run_all.py
```

Report is written to `artifacts/reports/benchmark.html`.

### 4. Live terminal scanner demo

```bash
# Watch Thompson sampling scan bands and track occupancy beliefs live in the terminal
python scripts/demo.py --scheduler thompson_sampling --steps 30

# Or test the periodicity-aware scheduler on periodic emitters
python scripts/demo.py --scheduler periodicity_aware --scenario configs/scenarios/periodic_emitters.yaml
```

### 5. Individual pipeline scripts

```bash
# Generate synthetic dataset
python scripts/generate_dataset.py --config configs/scenarios/periodic_emitters.yaml --seed 42

# Run all baseline schedulers on a benchmark suite
python scripts/run_baselines.py --suite configs/evaluation/benchmark_suite.yaml --seed 42

# Evaluate and save results
python scripts/evaluate.py --suite configs/evaluation/benchmark_suite.yaml --seed 42

# Produce HTML report
python scripts/make_report.py --results-dir artifacts/results --output artifacts/reports/benchmark.html
```

### 6. Run tests

```bash
pytest tests/ -v --cov=smart_scan_ew --cov-report=term-missing
```

---

## Repository Structure

```
smart-scan-ew/
├── backend/           Python backend (Simulation & Machine Learning)
│   ├── configs/       YAML scenario, receiver, training, and evaluation configs
│   ├── src/smart_scan_ew/ Main library
│   │   ├── environment/   RF simulator, emitters, propagation, receiver, noise
│   │   ├── detection/     Energy detector and feature extractor
│   │   ├── state_estimation/ Occupancy, periodicity, and belief-state estimators
│   │   ├── schedulers/    Uniform, random, round-robin, greedy, bandit, periodic
│   │   └── evaluation/    Metrics, episode runner, report generator
│   ├── scripts/       CLI entry-point scripts
│   ├── tests/         Unit, property, and integration tests
│   ├── artifacts/     Generated models, reports, and figures
│   └── data/          Raw, interim, and processed datasets
└── frontend/          Web user interface
    ├── app.js         Frontend application logic
    ├── index.html     Main UI structure
    └── style.css      Styling
```

---

## Schedulers

| Scheduler | Strategy |
|---|---|
| `UniformSweep` | Sequential fixed-dwell sweep over all bands |
| `RandomScan` | Uniform random band selection |
| `RoundRobin` | Circular scan with max-revisit enforcement |
| `GreedyOccupancy` | Picks band with highest estimated occupancy |
| `ThompsonSampling` | Bayesian bandit with Beta posteriors per band |
| `PeriodicityAware` | Phase-prediction + Thompson sampling + forced exploration |

---

## Metrics

- **P_D** — probability of detection  
- **P_FA** — false-alarm rate  
- **Interception rate** — fraction of emitters intercepted  
- **Mean/median/p95 intercept time** — primary optimisation target  
- **Coverage ratio** — useful band-time fraction  
- **Revisit interval** — mean & max per band  
- **Cumulative reward** — composite scheduler objective  

---

## Design Notes

- Simulator truth logs are **never** exposed to any scheduler during execution.
- Deterministic replay guaranteed by per-episode seeded `numpy.random.Generator`.
- All receiver constraints (dwell bounds, retuning delay, band validity) are enforced in `BaseScheduler`.
- Thompson Sampling uses 10 % per-step discount on counts for non-stationarity.
- Periodicity estimator uses autocorrelation over a sliding window of detection times.
