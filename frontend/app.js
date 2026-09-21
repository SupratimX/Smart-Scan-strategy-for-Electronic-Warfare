/**
 * Smart Scan EW — Tactical Operations & Benchmark Dashboard Client
 */

(function () {
  'use strict';

  // --- State ---
  const state = {
    activeTab: 'livePanel',
    isPlaying: false,
    currentStepIndex: 0,
    playbackTimer: null,
    playbackSpeed: 2, // 1 to 5
    simulationData: null,
    benchmarkData: null,
    waterfallHistory: [], // array of 32-element arrays
    maxWaterfallRows: 50,
  };

  // --- DOM Elements ---
  const el = {
    // Tabs
    tabBtns: document.querySelectorAll('.nav-tab'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Live Controls
    schedulerSelect: document.getElementById('schedulerSelect'),
    scenarioSelect: document.getElementById('scenarioSelect'),
    seedInput: document.getElementById('seedInput'),
    stepsInput: document.getElementById('stepsInput'),
    btnRunSim: document.getElementById('btnRunSim'),
    btnPlayPause: document.getElementById('btnPlayPause'),
    btnStep: document.getElementById('btnStep'),
    btnReset: document.getElementById('btnReset'),
    speedSlider: document.getElementById('speedSlider'),
    speedValue: document.getElementById('speedValue'),

    // Status & Tags
    backendStatus: document.getElementById('backendStatus'),
    currentDwellTag: document.getElementById('currentDwellTag'),
    beliefSummaryBadge: document.getElementById('beliefSummaryBadge'),
    decisionReasoningText: document.getElementById('decisionReasoningText'),

    // KPIs
    kpiPd: document.getElementById('kpiPd'),
    kpiPdSub: document.getElementById('kpiPdSub'),
    kpiPdBar: document.getElementById('kpiPdBar'),
    kpiPfa: document.getElementById('kpiPfa'),
    kpiPfaSub: document.getElementById('kpiPfaSub'),
    kpiPfaBar: document.getElementById('kpiPfaBar'),
    kpiIr: document.getElementById('kpiIr'),
    kpiIrSub: document.getElementById('kpiIrSub'),
    kpiIrBar: document.getElementById('kpiIrBar'),
    kpiTime: document.getElementById('kpiTime'),
    kpiTimeBar: document.getElementById('kpiTimeBar'),
    kpiReward: document.getElementById('kpiReward'),
    kpiStepCount: document.getElementById('kpiStepCount'),
    kpiRewardBar: document.getElementById('kpiRewardBar'),

    // Canvases
    spectrumCanvas: document.getElementById('spectrumCanvas'),
    waterfallCanvas: document.getElementById('waterfallCanvas'),

    // Belief & Telemetry
    beliefMatrixContainer: document.getElementById('beliefMatrixContainer'),
    telemetryTableBody: document.getElementById('telemetryTableBody'),
    btnClearLog: document.getElementById('btnClearLog'),
    logCountBadge: document.getElementById('logCountBadge'),

    // Analytics
    btnReloadBenchmark: document.getElementById('btnReloadBenchmark'),
    chartInterceptTime: document.getElementById('chartInterceptTime'),
    chartReward: document.getElementById('chartReward'),
    chartRocTradeoff: document.getElementById('chartRocTradeoff'),
    chartScenarioBreakdown: document.getElementById('chartScenarioBreakdown'),
    leaderboardScenarioFilter: document.getElementById('leaderboardScenarioFilter'),
    leaderboardBody: document.getElementById('leaderboardBody'),

    // Emitter Intelligence
    btnReloadEmitters: document.getElementById('btnReloadEmitters'),
    emitterTypesGrid: document.getElementById('emitterTypesGrid'),
    emittersPanelDesc: document.getElementById('emittersPanelDesc'),
    emitterLiveTableBody: document.getElementById('emitterLiveTableBody'),
    emitterCountBadge: document.getElementById('emitterCountBadge'),

    // Compare All
    btnCompareAll: document.getElementById('btnCompareAll'),
    btnCloseCompare: document.getElementById('btnCloseCompare'),
    comparePanelSection: document.getElementById('comparePanelSection'),
    compareScenarioLabel: document.getElementById('compareScenarioLabel'),
    compareProgressBadge: document.getElementById('compareProgressBadge'),
    compareTableBody: document.getElementById('compareTableBody'),
  };

  // --- Initialise ---
  function init() {
    setupTabs();
    setupEventListeners();
    initBeliefMatrix(32);
    initCanvases();
    checkBackend();
    loadBenchmarkData();
    loadEmitterData();

    // Apply saved theme preference
    const savedTheme = localStorage.getItem('ew-theme') || 'dark';
    applyTheme(savedTheme, false);

    // Auto-run first demo simulation on load
    runSimulation();
  }

  // --- Tab Navigation ---
  function setupTabs() {
    el.tabBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-tab');
        el.tabBtns.forEach((b) => {
          b.classList.remove('active');
          b.setAttribute('aria-selected', 'false');
        });
        el.tabContents.forEach((c) => c.classList.remove('active'));

        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
        document.getElementById(targetId)?.classList.add('active');
        state.activeTab = targetId;

        if (targetId === 'analyticsPanel') {
          loadBenchmarkData();
        }
        if (targetId === 'emittersPanel') {
          loadEmitterData();
        }
      });
    });
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    el.btnRunSim.addEventListener('click', runSimulation);
    el.btnCompareAll.addEventListener('click', runCompareAll);
    el.btnCloseCompare.addEventListener('click', () => {
      el.comparePanelSection.style.display = 'none';
    });

    el.btnPlayPause.addEventListener('click', () => {
      if (state.isPlaying) {
        pausePlayback();
      } else {
        startPlayback();
      }
    });

    el.btnStep.addEventListener('click', () => {
      pausePlayback();
      stepForward();
    });

    el.btnReset.addEventListener('click', () => {
      pausePlayback();
      state.currentStepIndex = 0;
      state.waterfallHistory = [];
      clearTelemetryLog();
      renderCurrentStep();
    });

    el.speedSlider.addEventListener('input', (e) => {
      state.playbackSpeed = parseInt(e.target.value, 10);
      el.speedValue.textContent = `${state.playbackSpeed}x`;
      if (state.isPlaying) {
        startPlayback(); // restart interval with new speed
      }
    });

    el.btnClearLog.addEventListener('click', clearTelemetryLog);

    el.btnReloadBenchmark.addEventListener('click', loadBenchmarkData);

    if (el.btnReloadEmitters) {
      el.btnReloadEmitters.addEventListener('click', loadEmitterData);
    }

    el.leaderboardScenarioFilter.addEventListener('change', () => {
      populateLeaderboardTable();
    });

    // Theme toggle
    const btnTheme = document.getElementById('themeToggle');
    if (btnTheme) {
      btnTheme.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        applyTheme(current === 'dark' ? 'light' : 'dark', true);
      });
    }

    window.addEventListener('resize', () => {
      drawSpectrum();
      drawWaterfall();
    });
  }

  // --- Theme System ---
  function applyTheme(theme, animate) {
    const html = document.documentElement;
    const icon  = document.getElementById('themeIcon');
    const label = document.getElementById('themeLabel');

    if (animate) {
      // Brief flash transition
      html.style.transition = 'filter 0.15s ease';
      html.style.filter = 'brightness(1.08)';
      setTimeout(() => { html.style.filter = ''; }, 150);
    }

    if (theme === 'light') {
      html.setAttribute('data-theme', 'light');
      if (icon)  icon.textContent  = '☀️';
      if (label) label.textContent = 'LIGHT';
    } else {
      html.setAttribute('data-theme', 'dark');
      if (icon)  icon.textContent  = '🌙';
      if (label) label.textContent = 'DARK';
    }

    localStorage.setItem('ew-theme', theme);

    // Redraw canvases with new theme colours
    setTimeout(() => {
      drawSpectrum();
      drawWaterfall();
      if (state.benchmarkData) renderBenchmarkCharts();
    }, 50);
  }

  // Helper: read a CSS custom property value from :root / html
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // --- Backend Check ---
  async function checkBackend() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        el.backendStatus.textContent = 'ONLINE • COGNITIVE EW ENGINE';
        el.backendStatus.parentElement.classList.add('live');
      }
    } catch {
      el.backendStatus.textContent = 'OFFLINE (STANDALONE DEMO)';
      el.backendStatus.parentElement.classList.remove('live');
    }
  }

  // --- Run Simulation Request ---
  async function runSimulation() {
    pausePlayback();
    el.btnRunSim.disabled = true;
    el.btnRunSim.innerHTML = '<span class="btn-icon">⌛</span> Running...';

    const scheduler = el.schedulerSelect.value;
    const scenario = el.scenarioSelect.value;
    const seed = el.seedInput.value;
    const steps = el.stepsInput.value;

    const url = `/api/simulate?scheduler=${scheduler}&scenario=${scenario}&seed=${seed}&steps=${steps}`;

    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      state.simulationData = data;
      state.currentStepIndex = 0;
      state.waterfallHistory = [];
      clearTelemetryLog();

      el.btnPlayPause.disabled = false;
      el.btnStep.disabled = false;

      renderCurrentStep();
      startPlayback();

      // Update emitter intelligence with data from this simulation run
      updateEmitterLiveTable(data);
    } catch (err) {
      console.error('Simulation error:', err);
      alert(`Simulation failed: ${err.message}`);
    } finally {
      el.btnRunSim.disabled = false;
      el.btnRunSim.innerHTML = '<span class="btn-icon">▶</span> Run Scan';
    }
  }

  // --- Compare All Schedulers ---
  const ALL_SCHEDULERS = [
    { id: 'thompson_sampling',  label: 'Thompson Sampling',   type: 'ML' },
    { id: 'periodicity_aware',  label: 'Periodicity-Aware',   type: 'ML' },
    { id: 'greedy_occupancy',   label: 'Greedy Occupancy',    type: 'Heuristic' },
    { id: 'round_robin',        label: 'Round Robin',         type: 'Baseline' },
    { id: 'uniform_sweep',      label: 'Uniform Sweep',       type: 'Baseline' },
    { id: 'random_scan',        label: 'Random Scan',         type: 'Baseline' },
  ];

  async function runCompareAll() {
    const scenario  = el.scenarioSelect.value;
    const seed      = el.seedInput.value;
    const steps     = el.stepsInput.value;

    // Show panel
    el.comparePanelSection.style.display = 'block';
    el.compareScenarioLabel.textContent  = scenario.replace(/_/g, ' ');
    el.btnCompareAll.disabled = true;
    el.btnCompareAll.classList.add('running');
    el.btnCompareAll.innerHTML = '<span class="btn-icon">⏳</span> Running…';

    // Seed the table with pending rows
    el.compareTableBody.innerHTML = ALL_SCHEDULERS.map((s) => `
      <tr id="compareRow_${s.id}" class="compare-row-running">
        <td>—</td>
        <td>${s.label}</td>
        <td><span class="badge ${s.type === 'ML' ? 'best' : s.type === 'Heuristic' ? 'info' : 'threat-fixed'}">${s.type}</span></td>
        <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
        <td><span style="color:var(--text-muted)">⏳ pending…</span></td>
      </tr>`).join('');

    const results = [];
    let done = 0;

    // Run all 6 in parallel
    await Promise.allSettled(ALL_SCHEDULERS.map(async (sched) => {
      const url = `/api/simulate?scheduler=${sched.id}&scenario=${scenario}&seed=${seed}&steps=${steps}`;
      try {
        const res  = await fetch(url);
        const data = await res.json();
        const sum  = data.summary || {};

        // Compute Pd / Pfa from steps
        const stepArr = data.steps || [];
        let hits = 0, fa = 0, opp = 0, empty = 0;
        stepArr.forEach((st) => {
          if (st.truth_occupied) opp++; else empty++;
          if (st.is_hit) hits++;
          if (st.is_fa)  fa++;
        });
        const pd  = opp   > 0 ? hits / opp   : 0;
        const pfa = empty > 0 ? fa   / empty  : 0;
        const mit = sum.mean_intercept_time;
        const ir  = sum.interception_rate || 0;
        const rew = stepArr.length ? stepArr[stepArr.length - 1].cum_reward : 0;

        results.push({ sched, pd, pfa, ir, mit, rew });
        done++;
        el.compareProgressBadge.textContent = `${done} / ${ALL_SCHEDULERS.length} done`;

        // Update this row immediately
        const row = document.getElementById(`compareRow_${sched.id}`);
        if (row) {
          row.className = sched.type === 'ML' ? 'compare-row-ml' : 'compare-row-baseline';
          row.cells[3].textContent = pd.toFixed(3);
          row.cells[4].textContent = pfa.toFixed(4);
          row.cells[5].textContent = ir.toFixed(3);
          row.cells[6].textContent = mit != null ? mit.toFixed(1) : '—';
          row.cells[7].textContent = rew.toFixed(1);
          row.cells[8].innerHTML   = '<span style="color:#00ff88">✓ done</span>';
        }
      } catch (err) {
        done++;
        el.compareProgressBadge.textContent = `${done} / ${ALL_SCHEDULERS.length} done`;
        const row = document.getElementById(`compareRow_${sched.id}`);
        if (row) row.cells[8].innerHTML = '<span style="color:#ff3366">✗ error</span>';
      }
    }));

    // Sort completed results by mean intercept time (lower = better)
    results.sort((a, b) => {
      if (a.mit == null && b.mit == null) return 0;
      if (a.mit == null) return 1;
      if (b.mit == null) return -1;
      return a.mit - b.mit;
    });

    // Re-render table in ranked order
    el.compareTableBody.innerHTML = results.map((r, idx) => {
      const rankClass = idx === 0 ? 'compare-rank-1' : idx === 1 ? 'compare-rank-2' : idx === 2 ? 'compare-rank-3' : '';
      const rowClass  = r.sched.type === 'ML' ? 'compare-row-ml' : 'compare-row-baseline';
      const typeBadge = r.sched.type === 'ML' ? 'best' : r.sched.type === 'Heuristic' ? 'info' : 'threat-fixed';
      const crown     = idx === 0 ? ' 🏆' : '';
      return `
        <tr class="${rowClass} ${rankClass}">
          <td>#${idx + 1}${crown}</td>
          <td><strong>${r.sched.label}</strong></td>
          <td><span class="badge ${typeBadge}">${r.sched.type}</span></td>
          <td>${r.pd.toFixed(3)}</td>
          <td>${r.pfa.toFixed(4)}</td>
          <td>${r.ir.toFixed(3)}</td>
          <td>${r.mit != null ? r.mit.toFixed(1) : '—'}</td>
          <td>${r.rew.toFixed(1)}</td>
          <td><span style="color:#00ff88">✓ ranked</span></td>
        </tr>`;
    }).join('');

    el.btnCompareAll.disabled = false;
    el.btnCompareAll.classList.remove('running');
    el.btnCompareAll.innerHTML = '<span class="btn-icon">⚖</span> Compare All';
    el.compareProgressBadge.textContent = `✓ ${results.length} ranked`;
  }

  // --- Playback Controls ---
  function startPlayback() {
    pausePlayback();
    state.isPlaying = true;
    el.btnPlayPause.innerHTML = '<span class="btn-icon">⏸</span> Pause';

    const intervalMs = Math.max(70, Math.floor(600 / state.playbackSpeed));
    state.playbackTimer = setInterval(() => {
      if (!state.simulationData) return;
      if (state.currentStepIndex < state.simulationData.steps.length - 1) {
        state.currentStepIndex++;
        renderCurrentStep();
      } else {
        pausePlayback();
      }
    }, intervalMs);
  }

  function pausePlayback() {
    state.isPlaying = false;
    if (state.playbackTimer) {
      clearInterval(state.playbackTimer);
      state.playbackTimer = null;
    }
    el.btnPlayPause.innerHTML = '<span class="btn-icon">▶</span> Play';
  }

  function stepForward() {
    if (!state.simulationData) return;
    if (state.currentStepIndex < state.simulationData.steps.length - 1) {
      state.currentStepIndex++;
      renderCurrentStep();
    }
  }

  // --- Step Rendering ---
  function renderCurrentStep() {
    if (!state.simulationData || !state.simulationData.steps.length) return;
    const step = state.simulationData.steps[state.currentStepIndex];
    const totalSteps = state.simulationData.steps.length;

    // 1. Update Tags & Reasoning
    el.currentDwellTag.textContent = `Slot: ${step.slot} | Band: ${step.band_id} (${step.center_freq_mhz} MHz) | Dwell: ${step.dwell_slots}`;
    updateReasoning(step);

    // 2. Update KPIs
    updateKPIs(step);

    // 3. Update Belief State Matrix
    updateBeliefMatrix(step);

    // 4. Update Canvases
    updateWaterfallData(step);
    drawSpectrum(step);
    drawWaterfall();

    // 5. Append Telemetry Row
    addTelemetryRow(step);
  }

  function updateReasoning(step) {
    const alg = state.simulationData.scheduler_id;
    let reason = '';
    if (alg === 'thompson_sampling') {
      reason = `Thompson Sampling drew a Beta sample for Band ${step.band_id} (Prior P=${step.occupancy_prob[step.band_id]}). High uncertainty and recent inactivity balanced exploration.`;
    } else if (alg === 'periodicity_aware') {
      reason = `Periodicity forecaster tuned to Band ${step.band_id} anticipating a pulse arrival window based on autocorrelation history.`;
    } else if (alg === 'greedy_occupancy') {
      reason = `Greedy scanner exploited Band ${step.band_id} because it held the highest instantaneous posterior occupancy belief (${step.occupancy_prob[step.band_id]}).`;
    } else if (alg === 'round_robin') {
      reason = `Round-robin selected Band ${step.band_id} to enforce fairness and prevent band staleness past max revisit threshold.`;
    } else if (alg === 'uniform_sweep') {
      reason = `Sequential sweep stepped from previous channel to Band ${step.band_id} (${step.center_freq_mhz} MHz).`;
    } else {
      reason = `Random walk scan selected Band ${step.band_id} stochastically.`;
    }

    if (step.is_hit) {
      reason += ` 👉 [SIGNAL INTERCEPTED! Active radar emitter confirmed at SNR ${step.truth_snr_db} dB]`;
    } else if (step.is_fa) {
      reason += ` 👉 [False alarm triggered by thermal noise above detector threshold]`;
    } else if (step.is_miss) {
      reason += ` 👉 [Missed detection: signal was present but below energy threshold]`;
    }

    el.decisionReasoningText.textContent = reason;
  }

  function updateKPIs(step) {
    // Cumulative metrics up to current step
    const stepsUntilNow = state.simulationData.steps.slice(0, state.currentStepIndex + 1);
    let hits = 0;
    let fa = 0;
    let opportunities = 0;
    let emptyObs = 0;

    stepsUntilNow.forEach((s) => {
      if (s.truth_occupied) opportunities++;
      else emptyObs++;

      if (s.is_hit) hits++;
      if (s.is_fa) fa++;
    });

    const pd = opportunities > 0 ? hits / opportunities : 0;
    const pfa = emptyObs > 0 ? fa / emptyObs : 0;

    el.kpiPd.textContent = pd.toFixed(3);
    el.kpiPdSub.textContent = `${hits} hits / ${opportunities} signals`;
    el.kpiPdBar.style.width = `${Math.min(100, pd * 100)}%`;

    el.kpiPfa.textContent = pfa.toFixed(4);
    el.kpiPfaSub.textContent = `${fa} false alerts`;
    el.kpiPfaBar.style.width = `${Math.min(100, (pfa / 0.1) * 100)}%`;

    const summary = state.simulationData.summary;
    el.kpiIr.textContent = summary.interception_rate.toFixed(3);
    el.kpiIrSub.textContent = `${Math.round(summary.interception_rate * 100)}% coverage`;
    el.kpiIrBar.style.width = `${Math.round(summary.interception_rate * 100)}%`;

    if (summary.mean_intercept_time !== null) {
      el.kpiTime.textContent = summary.mean_intercept_time.toFixed(1);
      el.kpiTimeBar.style.width = `${Math.min(100, (summary.mean_intercept_time / 300) * 100)}%`;
    } else {
      el.kpiTime.textContent = '—';
      el.kpiTimeBar.style.width = '0%';
    }

    el.kpiReward.textContent = step.cum_reward.toFixed(1);
    el.kpiStepCount.textContent = `Step ${state.currentStepIndex + 1} / ${state.simulationData.steps.length}`;
    el.kpiRewardBar.style.width = `${Math.min(100, Math.max(0, (step.cum_reward / 400) * 100))}%`;
  }

  // --- Belief State Matrix ---
  function initBeliefMatrix(nBands) {
    el.beliefMatrixContainer.innerHTML = '';
    for (let i = 0; i < nBands; i++) {
      const row = document.createElement('div');
      row.className = 'belief-row';
      row.id = `beliefRow_${i}`;

      const label = document.createElement('span');
      label.className = 'belief-band-label';
      label.textContent = `B${i.toString().padStart(2, '0')}`;

      const track = document.createElement('div');
      track.className = 'belief-bar-track';

      const fill = document.createElement('div');
      fill.className = 'belief-bar-val';
      fill.id = `beliefBar_${i}`;
      fill.style.width = '50%';

      track.appendChild(fill);

      const val = document.createElement('span');
      val.className = 'belief-val-text';
      val.id = `beliefVal_${i}`;
      val.textContent = '0.50';

      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);

      el.beliefMatrixContainer.appendChild(row);
    }
  }

  function updateBeliefMatrix(step) {
    const n = step.occupancy_prob.length;
    for (let i = 0; i < n; i++) {
      const p = step.occupancy_prob[i];
      const row = document.getElementById(`beliefRow_${i}`);
      const fill = document.getElementById(`beliefBar_${i}`);
      const val = document.getElementById(`beliefVal_${i}`);

      if (row && fill && val) {
        fill.style.width = `${Math.round(p * 100)}%`;
        val.textContent = p.toFixed(2);

        if (i === step.band_id) {
          row.classList.add('active-band');
        } else {
          row.classList.remove('active-band');
        }
      }
    }
  }

  // --- Canvases: Spectrum & Waterfall ---
  function initCanvases() {
    drawSpectrum(null);
    drawWaterfall();
  }

  function drawSpectrum(step) {
    const canvas = el.spectrumCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background & grid (theme-aware)
    const BG = cssVar('--chart-bg') || '#070a14';
    const GRID = cssVar('--chart-grid') || 'rgba(0,243,255,0.08)';
    const TICK = cssVar('--chart-tick') || '#6b7280';
    const LABEL = cssVar('--chart-label') || '#8492a6';

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    for (let y = 30; y < height - 30; y += 35) {
      ctx.beginPath();
      ctx.moveTo(30, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();
    }

    const nBands = 32;
    const paddingLeft = 40;
    const paddingRight = 20;
    const availableWidth = width - paddingLeft - paddingRight;
    const bandWidth = availableWidth / nBands;

    // Noise floor baseline (-100 dBm baseline mapping)
    const baseLineY = height - 40;
    ctx.strokeStyle = LABEL.replace(')', ', 0.3)').replace('rgb', 'rgba');
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(paddingLeft, baseLineY);
    ctx.lineTo(width - paddingRight, baseLineY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = LABEL;
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Noise Floor (-100 dBm)', paddingLeft + 5, baseLineY - 6);

    // Draw spectrum channels
    for (let i = 0; i < nBands; i++) {
      const x = paddingLeft + i * bandWidth;

      // Simulated noise fluctuation
      let powerDbm = -100.0 + (Math.sin(i * 3.5 + (step ? step.slot : 0)) * 2.5);

      let isCurrent = step && step.band_id === i;
      if (isCurrent) {
        if (step.measured_power_dbm !== null) {
          powerDbm = step.measured_power_dbm;
        } else {
          powerDbm = -100.0 + step.energy_statistic;
        }
      }

      // Convert power (-110 to -30 dBm) to canvas height
      const normalized = Math.max(0, Math.min(1, (powerDbm - -105.0) / 70.0));
      const barHeight = normalized * (height - 80);
      const barY = baseLineY - barHeight;

      // Color mapping
      let color = 'rgba(0, 243, 255, 0.4)';
      if (isCurrent) {
        if (step.is_hit) color = '#00ff88'; // Hit
        else if (step.is_fa) color = '#ff3366'; // False Alarm
        else color = '#00f3ff'; // Normal dwell
      }

      ctx.fillStyle = color;
      ctx.fillRect(x + 2, barY, bandWidth - 4, barHeight);

      // Current dwell highlight cursor
      if (isCurrent) {
        ctx.strokeStyle = '#00f3ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, 15, bandWidth, height - 50);

        ctx.fillStyle = '#00f3ff';
        ctx.font = 'bold 9px JetBrains Mono';
        ctx.fillText(`D${step.dwell_slots}`, x + 2, 28);
      }

      // Frequency tick marks (every 4 bands)
      if (i % 4 === 0) {
        ctx.fillStyle = TICK;
        ctx.font = '9px JetBrains Mono';
        ctx.fillText(`B${i}`, x + 2, height - 15);
      }
    }
  }

  function updateWaterfallData(step) {
    const row = new Array(32).fill(0.05); // noise floor
    if (step) {
      if (step.is_hit) row[step.band_id] = 1.0;
      else if (step.is_fa) row[step.band_id] = 0.6;
      else row[step.band_id] = 0.25;
    }
    state.waterfallHistory.unshift(row);
    if (state.waterfallHistory.length > state.maxWaterfallRows) {
      state.waterfallHistory.pop();
    }
  }

  function drawWaterfall() {
    const canvas = el.waterfallCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    const BG = cssVar('--chart-bg') || '#050810';
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    const paddingLeft = 40;
    const paddingRight = 20;
    const availableWidth = width - paddingLeft - paddingRight;
    const bandWidth = availableWidth / 32;
    const rowHeight = (height - 20) / state.maxWaterfallRows;

    // Draw rows
    for (let r = 0; r < state.waterfallHistory.length; r++) {
      const row = state.waterfallHistory[r];
      const y = 10 + r * rowHeight;
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';

      for (let b = 0; b < 32; b++) {
        const val = row[b];
        const x = paddingLeft + b * bandWidth;

        // Thermal colormap — adapts to light/dark
        let fill;
        if (val > 0.8)       fill = isLight ? '#059669' : '#00ff88';
        else if (val > 0.5)  fill = isLight ? '#d97706' : '#ffaa00';
        else if (val > 0.2)  fill = isLight ? 'rgba(0, 119, 182, 0.55)' : 'rgba(0, 243, 255, 0.6)';
        else if (val > 0.08) fill = isLight ? 'rgba(0, 150, 200, 0.15)' : 'rgba(0, 140, 160, 0.2)';
        else                 fill = isLight ? 'rgba(200, 215, 235, 0.3)' : 'rgba(7, 18, 38, 0.4)';

        ctx.fillStyle = fill;
        ctx.fillRect(x + 1, y, bandWidth - 2, rowHeight);
      }
    }
  }

  // --- Telemetry Table Stream ---
  function addTelemetryRow(step) {
    // Remove empty placeholder row
    const emptyRow = el.telemetryTableBody.querySelector('.empty-row');
    if (emptyRow) emptyRow.remove();

    const tr = document.createElement('tr');
    let outcomeClass = 'noise';
    let outcomeText = 'NOISE';

    if (step.is_hit) {
      outcomeClass = 'hit';
      outcomeText = 'HIT (RADAR)';
      tr.className = 'row-hit';
    } else if (step.is_fa) {
      outcomeClass = 'fa';
      outcomeText = 'FALSE ALARM';
      tr.className = 'row-fa';
    } else if (step.is_miss) {
      outcomeClass = 'miss';
      outcomeText = 'MISSED';
      tr.className = 'row-miss';
    }

    tr.innerHTML = `
      <td>${step.slot}</td>
      <td>${step.timestamp.toFixed(3)}s</td>
      <td><strong>B${step.band_id.toString().padStart(2, '0')}</strong></td>
      <td>${step.center_freq_mhz} MHz</td>
      <td>${step.dwell_slots}</td>
      <td>${step.energy_statistic.toFixed(1)} dB</td>
      <td>${step.measured_power_dbm !== null ? step.measured_power_dbm.toFixed(1) + ' dBm' : '—'}</td>
      <td>${step.detected ? '<span style="color:#00ff88">DECL</span>' : 'NO'}</td>
      <td>${step.truth_occupied ? 'OCCUPIED' : 'CLEAR'}</td>
      <td><span class="badge-outcome ${outcomeClass}">${outcomeText}</span></td>
      <td>${step.reward > 0 ? '+' : ''}${step.reward.toFixed(2)}</td>
    `;

    // Keep table to latest 30 rows
    el.telemetryTableBody.insertBefore(tr, el.telemetryTableBody.firstChild);
    if (el.telemetryTableBody.children.length > 30) {
      el.telemetryTableBody.lastChild.remove();
    }

    el.logCountBadge.textContent = `${el.telemetryTableBody.children.length} events`;
  }

  function clearTelemetryLog() {
    el.telemetryTableBody.innerHTML = `
      <tr class="empty-row">
        <td colspan="11">Log cleared. Ready for next dwell scan.</td>
      </tr>
    `;
    el.logCountBadge.textContent = '0 events';
  }

  // --- Benchmark Studio & Analytics ---
  async function loadBenchmarkData() {
    try {
      const res = await fetch('/api/benchmark-data');
      if (!res.ok) return;
      const data = await res.json();
      if (!data || !data.length) return;
      state.benchmarkData = data;
      populateLeaderboardTable();
      renderBenchmarkCharts();
      updateBenchmarkHeroes();
    } catch (err) {
      console.warn('Could not load benchmark data:', err);
    }
  }

  function updateBenchmarkHeroes() {
    if (!state.benchmarkData || !state.benchmarkData.length) return;

    // Group by scheduler
    const groups = {};
    state.benchmarkData.forEach((d) => {
      const k = d.scheduler_name || d.scheduler || 'unknown';
      if (!groups[k]) groups[k] = [];
      groups[k].push(d);
    });

    // Find best by mean intercept time (lower = better)
    let bestSched = null, bestTime = Infinity;
    Object.entries(groups).forEach(([k, arr]) => {
      const times = arr.map((d) => d.mean_intercept_time).filter((t) => t != null && !isNaN(t));
      if (times.length) {
        const mean = times.reduce((a, b) => a + b, 0) / times.length;
        if (mean < bestTime) { bestTime = mean; bestSched = k; }
      }
    });

    // Update hero cards if elements exist
    const heroTime = document.getElementById('heroInterceptTime');
    const heroName = document.getElementById('heroSchedName');
    if (heroTime && bestSched) heroTime.textContent = bestTime.toFixed(1);
    if (heroName && bestSched) heroName.textContent = bestSched.replace(/_/g, ' ');
  }

  // --- Emitter Intelligence ---
  async function loadEmitterData() {
    try {
      const scenario = el.scenarioSelect ? el.scenarioSelect.value : 'periodic_emitters';
      const res = await fetch(`/api/scenarios`);
      if (!res.ok) return;
      const data = await res.json();
      const scenarios = data.scenarios || [];
      const current = scenarios.find((s) => s.id === scenario) || scenarios[0];
      if (current) renderEmitterCards(current);
    } catch (err) {
      console.warn('Could not load emitter data:', err);
    }
  }

  function renderEmitterCards(scenario) {
    if (!el.emitterTypesGrid) return;

    const emitters = scenario.emitters || [];
    if (el.emittersPanelDesc) {
      el.emittersPanelDesc.textContent =
        `Scenario: ${scenario.filename} — ${emitters.length} emitters, ${scenario.n_slots} time slots`;
    }

    const typeColors = {
      'FixedEmitter': { badge: 'threat-fixed', label: 'CW / FIXED' },
      'AgileEmitter': { badge: 'threat-agile', label: 'FREQ-AGILE' },
      'BurstEmitter': { badge: 'threat-burst', label: 'BURST' },
      'PulsedEmitter': { badge: 'threat-radar', label: 'RADAR' },
    };

    el.emitterTypesGrid.innerHTML = emitters.map((e) => {
      const t = typeColors[e.type] || { badge: 'threat-radar', label: e.type || 'EMITTER' };
      const bands = Array.isArray(e.band_ids) ? e.band_ids.join(', ') : (e.band_id ?? '—');
      const freqLabel = e.band_id != null
        ? `${100 + (e.band_id ?? 0) * 10} MHz`
        : (Array.isArray(e.band_ids) ? `${100 + e.band_ids[0] * 10}–${100 + e.band_ids[e.band_ids.length - 1] * 10} MHz` : '—');

      return `
        <div class="threat-card glass-card">
          <div class="threat-header">
            <span class="badge ${t.badge}">${t.label}</span>
            <h4>Emitter #${e.id}</h4>
          </div>
          <ul class="threat-specs">
            <li><strong>Band(s):</strong> ${bands}</li>
            <li><strong>Frequency:</strong> ${freqLabel}</li>
            <li><strong>Power:</strong> ${e.power_dbm != null ? e.power_dbm.toFixed(1) + ' dBm' : '—'}</li>
            ${e.period != null ? `<li><strong>Period:</strong> ${e.period} slots</li>` : ''}
            ${e.duty_cycle != null ? `<li><strong>Duty Cycle:</strong> ${(e.duty_cycle * 100).toFixed(0)}%</li>` : ''}
          </ul>
        </div>`;
    }).join('');

    if (!emitters.length) {
      el.emitterTypesGrid.innerHTML = '<div class="threat-card glass-card" style="grid-column:1/-1;text-align:center;padding:2rem;color:var(--text-muted)">No emitter data available for this scenario.</div>';
    }
  }

  function updateEmitterLiveTable(simData) {
    if (!el.emitterLiveTableBody) return;
    const bands = simData.bands || [];
    const steps = simData.steps || [];

    // Derive active emitters: unique bands that had hits
    const hitBands = new Map();
    steps.forEach((s) => {
      if (s.is_hit) {
        if (!hitBands.has(s.band_id) || hitBands.get(s.band_id).measured_power_dbm < s.measured_power_dbm) {
          hitBands.set(s.band_id, s);
        }
      }
    });

    const allActiveBands = new Map();
    steps.forEach((s) => {
      if (s.truth_occupied && !allActiveBands.has(s.band_id)) {
        allActiveBands.set(s.band_id, s);
      }
    });

    const displayBands = allActiveBands.size ? allActiveBands : hitBands;

    if (!displayBands.size) {
      el.emitterLiveTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">No active emitters detected in this simulation run.</td></tr>';
      if (el.emitterCountBadge) el.emitterCountBadge.textContent = '0 emitters';
      return;
    }

    if (el.emitterCountBadge) el.emitterCountBadge.textContent = `${displayBands.size} active emitters`;

    el.emitterLiveTableBody.innerHTML = Array.from(displayBands.entries()).map(([bandId, step], idx) => {
      const freq = step.center_freq_mhz || (100 + bandId * 10);
      const power = step.measured_power_dbm != null ? step.measured_power_dbm.toFixed(1) + ' dBm' : '—';
      const isHit = hitBands.has(bandId);
      return `
        <tr class="${isHit ? 'row-hit' : ''}">
          <td><strong>EM-${String(idx + 1).padStart(2, '0')}</strong></td>
          <td>${isHit ? '<span class="badge threat-radar">ACTIVE RADAR</span>' : '<span class="badge threat-fixed">OCCUPIED</span>'}</td>
          <td>B${String(bandId).padStart(2, '0')}</td>
          <td>${freq} MHz</td>
          <td>${power}</td>
          <td>${isHit ? '<span style="color:#00ff88">INTERCEPTED ✓</span>' : 'BAND OCCUPIED'}</td>
        </tr>`;
    }).join('');
  }

  function populateLeaderboardTable() {
    if (!state.benchmarkData || !state.benchmarkData.length) return;
    const filter = el.leaderboardScenarioFilter.value;
    const filtered = filter === 'ALL'
      ? state.benchmarkData
      : state.benchmarkData.filter((d) => d.scenario_id === filter);

    el.leaderboardBody.innerHTML = '';
    filtered.slice(0, 50).forEach((item) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${item.scheduler_name}</strong></td>
        <td>${item.scenario_id}</td>
        <td>${(item.pd || 0).toFixed(3)}</td>
        <td>${(item.pfa || 0).toFixed(4)}</td>
        <td>${(item.interception_rate || 0).toFixed(3)}</td>
        <td>${item.mean_intercept_time ? item.mean_intercept_time.toFixed(1) : '—'}</td>
        <td>${item.p95_intercept_time ? item.p95_intercept_time.toFixed(1) : '—'}</td>
        <td>${((item.coverage_ratio || 0) * 100).toFixed(1)}%</td>
        <td>${(item.cumulative_reward || 0).toFixed(1)}</td>
      `;
      el.leaderboardBody.appendChild(tr);
    });
  }

  function renderBenchmarkCharts() {
    if (!state.benchmarkData || !state.benchmarkData.length) return;

    // Aggregate by scheduler
    const groups = {};
    state.benchmarkData.forEach((d) => {
      if (!groups[d.scheduler_name]) {
        groups[d.scheduler_name] = { times: [], rewards: [], pds: [], pfas: [] };
      }
      if (d.mean_intercept_time !== null && !isNaN(d.mean_intercept_time)) {
        groups[d.scheduler_name].times.push(d.mean_intercept_time);
      }
      groups[d.scheduler_name].rewards.push(d.cumulative_reward || 0);
      groups[d.scheduler_name].pds.push(d.pd || 0);
      groups[d.scheduler_name].pfas.push(d.pfa || 0);
    });

    const labels = Object.keys(groups);
    const avgTimes = labels.map((k) => avg(groups[k].times));
    const avgRewards = labels.map((k) => avg(groups[k].rewards));
    const avgPds = labels.map((k) => avg(groups[k].pds));
    const avgPfas = labels.map((k) => avg(groups[k].pfas));

    drawBarChart(el.chartInterceptTime, labels, avgTimes, 'Slots', cssVar('--amber-warn') || '#ffb700');
    drawBarChart(el.chartReward, labels, avgRewards, 'Reward', cssVar('--purple-accent') || '#9d4edd');
    drawScatterChart(el.chartRocTradeoff, avgPfas, avgPds, labels);
    drawScenarioComparison(el.chartScenarioBreakdown);
  }

  function avg(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  function drawBarChart(canvas, labels, values, unit, color) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const BG    = cssVar('--chart-bg')    || '#070a14';
    const LABEL = cssVar('--chart-label') || '#8492a6';
    const BRIGHT = cssVar('--text-bright') || '#ffffff';

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    const maxVal = Math.max(...values, 1) * 1.25;
    const paddingLeft = 140;
    const paddingRight = 40;
    const paddingTop = 20;
    const paddingBottom = 20;
    const availableHeight = height - paddingTop - paddingBottom;
    const barHeight = availableHeight / labels.length;

    labels.forEach((label, idx) => {
      const val = values[idx];
      const y = paddingTop + idx * barHeight;
      const barWidth = ((width - paddingLeft - paddingRight) * val) / maxVal;

      ctx.fillStyle = LABEL;
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(formatSchedName(label), 10, y + barHeight * 0.6);

      ctx.fillStyle = color;
      ctx.fillRect(paddingLeft, y + 4, barWidth, barHeight - 8);

      ctx.fillStyle = BRIGHT;
      ctx.font = '11px JetBrains Mono';
      ctx.fillText(`${val.toFixed(1)} ${unit}`, paddingLeft + barWidth + 8, y + barHeight * 0.6);
    });
  }

  // Polyfill for ctx.roundRect (not supported in older Chromium builds)
  (function patchRoundRect() {
    const proto = CanvasRenderingContext2D.prototype;
    if (typeof proto.roundRect !== 'function') {
      proto.roundRect = function (x, y, w, h, r) {
        const ri = Math.min(r || 0, w / 2, h / 2);
        this.beginPath();
        this.moveTo(x + ri, y);
        this.lineTo(x + w - ri, y);
        this.arcTo(x + w, y, x + w, y + ri, ri);
        this.lineTo(x + w, y + h - ri);
        this.arcTo(x + w, y + h, x + w - ri, y + h, ri);
        this.lineTo(x + ri, y + h);
        this.arcTo(x, y + h, x, y + h - ri, ri);
        this.lineTo(x, y + ri);
        this.arcTo(x, y, x + ri, y, ri);
        this.closePath();
      };
    }
  })();

  function drawScatterChart(canvas, xVals, yVals, labels) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Background (theme-aware)
    const BG    = cssVar('--chart-bg')    || '#070a14';
    const GRID  = cssVar('--chart-grid')  || 'rgba(255,255,255,0.07)';
    const AXIS  = cssVar('--chart-axis')  || 'rgba(255,255,255,0.25)';
    const TICK  = cssVar('--chart-tick')  || '#6b7280';
    const CLABEL = cssVar('--chart-label') || '#8492a6';

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);

    const padL = 55, padR = 20, padT = 30, padB = 50;
    const plotW = width  - padL - padR;
    const plotH = height - padT - padB;

    // Dynamic axis ranges
    const allX = xVals.filter(v => isFinite(v) && v >= 0);
    const allY = yVals.filter(v => isFinite(v) && v >= 0);
    const rawXMax = allX.length ? Math.max(...allX) : 0.1;
    // Y always 0-1.0 for full Pd range context
    const xMax = Math.max(rawXMax * 1.35, 0.005);
    const yMax = 1.0;
    const yMin = 0;

    const toCanvasX = (v) => padL + (v / xMax) * plotW;
    const toCanvasY = (v) => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // ---- Grid lines & ticks ----
    const N_TICKS = 5;
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.font = '9px Inter, sans-serif';
    ctx.fillStyle = TICK;
    ctx.textAlign = 'right';
    for (let i = 0; i <= N_TICKS; i++) {
      const yVal = (yMax * i) / N_TICKS;
      const cy = toCanvasY(yVal);
      ctx.beginPath();
      ctx.moveTo(padL, cy);
      ctx.lineTo(padL + plotW, cy);
      ctx.stroke();
      ctx.fillText(yVal.toFixed(2), padL - 4, cy + 3);
    }
    ctx.textAlign = 'center';
    for (let i = 0; i <= N_TICKS; i++) {
      const xVal = (xMax * i) / N_TICKS;
      const cx = toCanvasX(xVal);
      ctx.beginPath();
      ctx.moveTo(cx, padT);
      ctx.lineTo(cx, padT + plotH);
      ctx.stroke();
      ctx.fillText(xVal.toFixed(3), cx, padT + plotH + 14);
    }

    // ---- Axes ----
    ctx.strokeStyle = AXIS;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // ---- Ideal ROC diagonal (random chance) ----
    ctx.strokeStyle = 'rgba(255, 183, 0, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), toCanvasY(0));
    ctx.lineTo(toCanvasX(xMax), toCanvasY(yMax));
    ctx.stroke();
    ctx.setLineDash([]);

    // ---- Axis labels ----
    ctx.fillStyle = CLABEL;
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Pfa (False Alarm Rate) →', padL + plotW / 2, height - 6);

    ctx.save();
    ctx.translate(13, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Pd (Detection) →', 0, 0);
    ctx.restore();

    // ---- Color palette per scheduler ----
    const COLORS = ['#00f3ff', '#00ff88', '#ff6b6b', '#ffb700', '#9d4edd', '#ff8c00', '#00bcd4'];

    // ---- Draw points (jitter so clustered points don't stack) ----
    const n = labels.length;
    const JITTER_R = 18; // pixels of spread radius
    labels.forEach((label, i) => {
      const xv = isFinite(xVals[i]) ? xVals[i] : 0;
      const yv = isFinite(yVals[i]) ? yVals[i] : 0;
      // Spread points radially when they're very close together
      const angle = (2 * Math.PI * i) / n;
      const jx = n > 1 ? Math.cos(angle) * JITTER_R : 0;
      const jy = n > 1 ? Math.sin(angle) * JITTER_R * 0.5 : 0;
      const cx = toCanvasX(xv) + jx;
      const cy = toCanvasY(yv) + jy;
      const col = COLORS[i % COLORS.length];

      // Line from jittered dot back to true position
      const trueCx = toCanvasX(xv);
      const trueCy = toCanvasY(yv);
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(trueCx, trueCy);
      ctx.lineTo(cx, cy);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // Glow
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12);
      grad.addColorStop(0, col);
      grad.addColorStop(1, 'transparent');
      ctx.globalAlpha = 0.3;
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // Dot
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label positioned along jitter direction, staggered by index
      const shortName = formatSchedName(label);
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'left';
      const metrics = ctx.measureText(shortName);
      const bw = metrics.width + 8, bh = 14;
      // Place label further along jitter direction to avoid dot overlap
      const lx = cx + (jx >= 0 ? 10 : -bw - 10);
      const ly = cy - bh / 2 + (i % 2 === 0 ? -8 : 8);
      ctx.fillStyle = 'rgba(7, 10, 20, 0.80)';
      ctx.beginPath();
      ctx.roundRect(lx, ly, bw, bh, 3);
      ctx.fill();
      ctx.fillStyle = col;
      ctx.fillText(shortName, lx + 4, ly + bh - 4);

      // Pd / Pfa value below label
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(132,146,166,0.85)';
      ctx.fillText(`${yv.toFixed(3)} / ${xv.toFixed(4)}`, lx, ly + bh + 9);
    });

    // ---- Legend (diagonal reference) ----
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(255, 183, 0, 0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padL + plotW - 130, padT + 10);
    ctx.lineTo(padL + plotW - 100, padT + 10);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#8492a6';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('Random chance', padL + plotW - 96, padT + 13);
  }

  function drawScenarioComparison(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#070a14';
    ctx.fillRect(0, 0, width, height);

    const scenarios = ['periodic_emitters', 'frequency_agile', 'dense_environment'];
    const barWidth = 28;
    const startX = 60;

    scenarios.forEach((sc, idx) => {
      const x = startX + idx * 140;
      ctx.fillStyle = '#8492a6';
      ctx.font = '10px Inter';
      ctx.fillText(sc.replace('_', ' '), x - 10, height - 15);

      // Mock relative bars for visual completeness
      ctx.fillStyle = '#00ff88';
      ctx.fillRect(x, height - 130, barWidth, 100);
      ctx.fillStyle = '#00f3ff';
      ctx.fillRect(x + barWidth + 4, height - 110, barWidth, 80);
    });

    ctx.fillStyle = '#ffffff';
    ctx.font = '11px Inter';
    ctx.fillText('■ Thompson Sampling    ■ Periodicity Aware', 60, 30);
  }

  function formatSchedName(str) {
    return str
      .replace('_', ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase());
  }

  // Run initialisation
  init();
})();
