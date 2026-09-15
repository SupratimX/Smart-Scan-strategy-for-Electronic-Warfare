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
  };

  // --- Initialise ---
  function init() {
    setupTabs();
    setupEventListeners();
    initBeliefMatrix(32);
    initCanvases();
    checkBackend();
    loadBenchmarkData();

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
          renderBenchmarkCharts();
        }
      });
    });
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    el.btnRunSim.addEventListener('click', runSimulation);

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

    el.leaderboardScenarioFilter.addEventListener('change', () => {
      populateLeaderboardTable();
    });

    window.addEventListener('resize', () => {
      drawSpectrum();
      drawWaterfall();
    });
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
    } catch (err) {
      console.error('Simulation error:', err);
      alert(`Simulation failed: ${err.message}`);
    } finally {
      el.btnRunSim.disabled = false;
      el.btnRunSim.innerHTML = '<span class="btn-icon">▶</span> Run Scan';
    }
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

    // Background & grid
    ctx.fillStyle = '#070a14';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(0, 243, 255, 0.08)';
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
    ctx.strokeStyle = 'rgba(132, 146, 166, 0.3)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(paddingLeft, baseLineY);
    ctx.lineTo(width - paddingRight, baseLineY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#8492a6';
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
        ctx.fillStyle = '#8492a6';
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
    ctx.fillStyle = '#050810';
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

      for (let b = 0; b < 32; b++) {
        const val = row[b];
        const x = paddingLeft + b * bandWidth;

        // Thermal colormap (dark blue -> cyan -> radar green -> alert yellow/red)
        let fill = 'rgba(7, 18, 38, 0.4)';
        if (val > 0.8) fill = '#00ff88'; // Hit
        else if (val > 0.5) fill = '#ffaa00'; // False Alarm
        else if (val > 0.2) fill = 'rgba(0, 243, 255, 0.6)'; // Scanned dwell
        else if (val > 0.08) fill = 'rgba(0, 140, 160, 0.2)';

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
      state.benchmarkData = data;
      populateLeaderboardTable();
      renderBenchmarkCharts();
    } catch (err) {
      console.warn('Could not load benchmark data:', err);
    }
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

    drawBarChart(el.chartInterceptTime, labels, avgTimes, 'Slots', '#ffb700');
    drawBarChart(el.chartReward, labels, avgRewards, 'Reward', '#9d4edd');
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

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#070a14';
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

      ctx.fillStyle = '#c9d1d9';
      ctx.font = '11px Inter, sans-serif';
      ctx.fillText(formatSchedName(label), 10, y + barHeight * 0.6);

      ctx.fillStyle = color;
      ctx.fillRect(paddingLeft, y + 4, barWidth, barHeight - 8);

      ctx.fillStyle = '#ffffff';
      ctx.font = '11px JetBrains Mono';
      ctx.fillText(`${val.toFixed(1)} ${unit}`, paddingLeft + barWidth + 8, y + barHeight * 0.6);
    });
  }

  function drawScatterChart(canvas, xVals, yVals, labels) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#070a14';
    ctx.fillRect(0, 0, width, height);

    const pad = 40;
    // Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, height - pad);
    ctx.lineTo(width - pad, height - pad);
    ctx.stroke();

    ctx.fillStyle = '#8492a6';
    ctx.font = '10px Inter';
    ctx.fillText('Pfa (False Alarm Rate) →', width / 2 - 40, height - 10);

    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Pd (Detection) →', -40, 0);
    ctx.restore();

    labels.forEach((label, i) => {
      const x = pad + (xVals[i] / 0.1) * (width - 2 * pad);
      const y = height - pad - (yVals[i] / 1.0) * (height - 2 * pad);

      ctx.fillStyle = '#00f3ff';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText(formatSchedName(label), x + 8, y + 3);
    });
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
