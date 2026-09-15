/**
 * Drosophila Neural Action Decoder — Warm Editorial Engine
 * Scientific monograph aesthetic: Ivory, Charcoal, Stone, & Dusty Rose
 * Precision 6-legged kinematics, calibrated electrophysiology raster, and predictive telemetry.
 */

// Application State
const STATE = {
  streamData: null,
  currentIndex: 0,
  isPlaying: true,
  playbackSpeed: 1.0,
  isStimulating: false,
  stimulatedNeuron: null,
  optoTimer: null,
  flyPos: { x: 410, y: 185, yaw: 0, scale: 1.0 },
  gaitPhase: 0,
  mediaRecorder: null,
  recordedChunks: []
};

// Actions with editorial palette mappings
const ACTIONS = [
  { id: 0, name: "Quiescence / Idle", icon: "·", color: "#6C6761" },
  { id: 1, name: "Forward Walk", icon: "→", color: "#844D43" },
  { id: 2, name: "Backward Walk (Moonwalk)", icon: "←", color: "#9E6D60" },
  { id: 3, name: "Turn Left", icon: "↖", color: "#557864" },
  { id: 4, name: "Turn Right", icon: "↗", color: "#6B8576" },
  { id: 5, name: "Groom Head/Antennae", icon: "◇", color: "#B2887B" },
  { id: 6, name: "Escape Jump", icon: "↑", color: "#945338" },
  { id: 7, name: "Courtship Wing Flare", icon: "♫", color: "#844D43" }
];

// Electrophysiology Channels (Descending Command Bottleneck)
const NEURONS = [
  { id: "DNp09", name: "DNp09", color: "#557864", maxRate: 80 },
  { id: "MDN", name: "MDN", color: "#844D43", maxRate: 80 },
  { id: "DNg02_L", name: "DNg02-L", color: "#9E6D60", maxRate: 70 },
  { id: "DNg02_R", name: "DNg02-R", color: "#B2887B", maxRate: 70 },
  { id: "aDN", name: "aDN", color: "#C2A193", maxRate: 75 },
  { id: "GF", name: "GF", color: "#945338", maxRate: 110 },
  { id: "pIP10", name: "pIP10", color: "#783A35", maxRate: 70 },
  { id: "EPG_Compass", name: "E-PG", color: "#8C7769", maxRate: 50 }
];

// Canvas References
const flyCanvas = document.getElementById("fly-canvas");
const flyCtx = flyCanvas.getContext("2d");
const rasterCanvas = document.getElementById("raster-canvas");
const rasterCtx = rasterCanvas.getContext("2d");

// Rolling history buffer for electrophysiology chart
const RASTER_BUFFER_SIZE = 280;
const rasterBuffer = Array.from({ length: NEURONS.length }, () => new Array(RASTER_BUFFER_SIZE).fill(0));

// Initialize Application
async function initApp() {
  buildNeuronLegend();
  buildDistributionTable();
  setupEventListeners();

  try {
    const res = await fetch("data/test_neural_stream.json");
    if (!res.ok) throw new Error("Failed to load stream data");
    STATE.streamData = await res.json();
    
    if (STATE.streamData.metadata && STATE.streamData.metadata.benchmark) {
      const acc = (STATE.streamData.metadata.benchmark.rf_accuracy * 100).toFixed(1);
      const statEl = document.getElementById("stat-accuracy");
      if (statEl) statEl.innerText = `${acc}%`;
    }
  } catch (err) {
    console.warn("Using fallback synthetic stream:", err);
    STATE.streamData = generateSyntheticStream();
  }

  requestAnimationFrame(simulationLoop);
}

// Build Telemetry Channel Legend
function buildNeuronLegend() {
  const container = document.getElementById("neuron-legend");
  if (!container) return;
  container.innerHTML = "";
  NEURONS.forEach(n => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `
      <span class="legend-swatch" style="background: ${n.color};"></span>
      <span>${n.id}</span>
    `;
    container.appendChild(item);
  });
}

// Build Probability Distribution Table
function buildDistributionTable() {
  const container = document.getElementById("probabilities-list");
  if (!container) return;
  container.innerHTML = "";
  ACTIONS.forEach(act => {
    const row = document.createElement("div");
    row.className = "dist-row";
    row.id = `dist-row-${act.id}`;
    row.innerHTML = `
      <div class="dist-label" title="${act.name}">${act.icon} ${act.name}</div>
      <div class="dist-track">
        <div class="dist-fill" id="dist-fill-${act.id}" style="width: 0%;"></div>
      </div>
      <div class="dist-val" id="dist-val-${act.id}">0.0%</div>
    `;
    container.appendChild(row);
  });
}

// Event Listeners
function setupEventListeners() {
  const btnPlayPause = document.getElementById("btn-play-pause");
  if (btnPlayPause) {
    btnPlayPause.addEventListener("click", () => {
      STATE.isPlaying = !STATE.isPlaying;
      btnPlayPause.innerHTML = STATE.isPlaying ? "⏸ Pause" : "▶ Resume";
    });
  }

  const btnSpeed = document.getElementById("btn-speed");
  if (btnSpeed) {
    const speeds = [0.5, 1.0, 1.5, 2.0];
    btnSpeed.addEventListener("click", () => {
      const currIdx = speeds.indexOf(STATE.playbackSpeed);
      STATE.playbackSpeed = speeds[(currIdx + 1) % speeds.length];
      btnSpeed.innerText = `${STATE.playbackSpeed.toFixed(1)}×`;
    });
  }

  const optoButtons = document.querySelectorAll(".opto-button[data-neuron]");
  optoButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const neuronId = btn.getAttribute("data-neuron");
      triggerOptogeneticStimulation(neuronId);
      optoButtons.forEach(b => b.classList.remove("active-opto"));
      btn.classList.add("active-opto");
    });
  });

  const resumeBtn = document.getElementById("stim-resume");
  if (resumeBtn) {
    resumeBtn.addEventListener("click", () => {
      resumeAutonomousStream();
      optoButtons.forEach(b => b.classList.remove("active-opto"));
    });
  }

  const btnRecord = document.getElementById("btn-record-clip");
  if (btnRecord) btnRecord.addEventListener("click", startClipRecording);

  const btnStop = document.getElementById("btn-stop-recording");
  if (btnStop) btnStop.addEventListener("click", stopClipRecording);
}

// Optogenetic Stimulation Logic
function triggerOptogeneticStimulation(neuronId) {
  STATE.isStimulating = true;
  STATE.stimulatedNeuron = neuronId;
  
  const statusChip = document.getElementById("stream-status");
  const statusText = document.getElementById("status-text");
  if (statusChip && statusText) {
    statusChip.style.background = "#F4ECE7";
    statusChip.style.borderColor = "#9E6D60";
    statusText.innerText = `INTERVENTION ACTIVE · ${neuronId}`;
  }

  if (STATE.optoTimer) clearTimeout(STATE.optoTimer);
  STATE.optoTimer = setTimeout(() => {
    resumeAutonomousStream();
  }, 6000);
}

function resumeAutonomousStream() {
  STATE.isStimulating = false;
  STATE.stimulatedNeuron = null;
  
  const statusChip = document.getElementById("stream-status");
  const statusText = document.getElementById("status-text");
  if (statusChip && statusText) {
    statusChip.style.background = "var(--bg-surface)";
    statusChip.style.borderColor = "var(--border-stone)";
    statusText.innerText = "DECODING STREAM · 100 HZ";
  }
  document.querySelectorAll(".opto-button").forEach(b => b.classList.remove("active-opto"));
}

// Synthetic Fallback Generator
function generateSyntheticStream() {
  const frames = [];
  for (let i = 0; i < 900; i++) {
    const act = Math.floor((i / 50) % 8);
    const rates = NEURONS.map(() => 4 + Math.random() * 6);
    const probs = new Array(8).fill(0.015);
    probs[act] = 0.88;
    frames.push({
      t: i * 0.02,
      rates: rates,
      actual_action: act,
      predicted_next_action: act,
      next_actual_action: act,
      probabilities: probs
    });
  }
  return { stream: frames };
}

// -------------------------------------------------------------
// SIMULATION & KINEMATICS LOOP
// -------------------------------------------------------------
let lastTimestamp = performance.now();

function simulationLoop(timestamp) {
  const delta = (timestamp - lastTimestamp) / 1000;
  lastTimestamp = timestamp;

  if (STATE.isPlaying && STATE.streamData && STATE.streamData.stream.length > 0) {
    updateState(delta);
  }

  renderFlyPlate();
  renderOscilloscopePlate();

  requestAnimationFrame(simulationLoop);
}

function updateState(delta) {
  let frame = STATE.streamData.stream[STATE.currentIndex];

  if (STATE.isStimulating && STATE.stimulatedNeuron) {
    frame = JSON.parse(JSON.stringify(frame));
    const nIdx = NEURONS.findIndex(n => n.id === STATE.stimulatedNeuron);
    if (nIdx !== -1) {
      frame.rates[nIdx] = 72.0 + Math.random() * 12.0;
    }

    let forcedAction = 1;
    if (STATE.stimulatedNeuron === "MDN") forcedAction = 2;
    else if (STATE.stimulatedNeuron === "GF") forcedAction = 6;
    else if (STATE.stimulatedNeuron === "DNg02_L") forcedAction = 3;
    else if (STATE.stimulatedNeuron === "aDN") forcedAction = 5;
    else if (STATE.stimulatedNeuron === "pIP10") forcedAction = 7;

    frame.predicted_next_action = forcedAction;
    frame.actual_action = forcedAction;
    frame.probabilities = new Array(8).fill(0.01);
    frame.probabilities[forcedAction] = 0.97;
  }

  for (let i = 0; i < NEURONS.length; i++) {
    rasterBuffer[i].shift();
    rasterBuffer[i].push(frame.rates[i] || 0);
  }

  updateForecastUI(frame);
  updateKinematics(frame.actual_action, delta);

  STATE.currentIndex = (STATE.currentIndex + 1) % STATE.streamData.stream.length;
}

function updateKinematics(actionId, delta) {
  const speed = STATE.playbackSpeed;

  switch (actionId) {
    case 1: // Forward Walk
      STATE.gaitPhase += 14 * delta * speed;
      STATE.flyPos.y = 185 + Math.sin(STATE.gaitPhase * 0.5) * 3;
      STATE.flyPos.yaw = Math.sin(STATE.gaitPhase * 0.25) * 0.04;
      document.getElementById("gait-phase-tag").innerText = "Tripod Forward Locomotion (12 Hz)";
      break;

    case 2: // Backward Walk (Moonwalk)
      STATE.gaitPhase -= 12 * delta * speed;
      STATE.flyPos.y = 185 + Math.sin(STATE.gaitPhase * 0.5) * 4;
      STATE.flyPos.yaw = 0;
      document.getElementById("gait-phase-tag").innerText = "Inverted Tripod Gait (Moonwalk)";
      break;

    case 3: // Turn Left
      STATE.gaitPhase += 10 * delta * speed;
      STATE.flyPos.yaw = -0.32 + Math.sin(STATE.gaitPhase * 0.5) * 0.06;
      document.getElementById("gait-phase-tag").innerText = "Asymmetric Leg Pivot: Left Yaw";
      break;

    case 4: // Turn Right
      STATE.gaitPhase += 10 * delta * speed;
      STATE.flyPos.yaw = 0.32 + Math.sin(STATE.gaitPhase * 0.5) * 0.06;
      document.getElementById("gait-phase-tag").innerText = "Asymmetric Leg Pivot: Right Yaw";
      break;

    case 5: // Groom Head/Antennae
      STATE.gaitPhase += 18 * delta * speed;
      STATE.flyPos.yaw = 0;
      document.getElementById("gait-phase-tag").innerText = "Antennal & Head Sweeping Program";
      break;

    case 6: // Escape Jump
      STATE.gaitPhase += 25 * delta * speed;
      const jumpCycle = Math.sin(STATE.gaitPhase * 0.4);
      STATE.flyPos.scale = jumpCycle > 0 ? 1.0 + jumpCycle * 0.32 : 0.96;
      STATE.flyPos.y = 185 - Math.max(0, jumpCycle) * 35;
      document.getElementById("gait-phase-tag").innerText = "Ballistic Escape Takeoff Jump";
      break;

    case 7: // Courtship Wing Flare
      STATE.gaitPhase += 35 * delta * speed;
      STATE.flyPos.scale = 1.0;
      STATE.flyPos.yaw = 0.12;
      document.getElementById("gait-phase-tag").innerText = "Unilateral Wing Song Flare (35 Hz)";
      break;

    default: // Quiescent
      STATE.gaitPhase += 2 * delta * speed;
      STATE.flyPos.scale = 1.0 + Math.sin(STATE.gaitPhase) * 0.012;
      STATE.flyPos.yaw = 0;
      document.getElementById("gait-phase-tag").innerText = "Quiescent Posture";
      break;
  }
}

function updateForecastUI(frame) {
  const currentAction = ACTIONS[frame.actual_action] || ACTIONS[0];
  const predictedAction = ACTIONS[frame.predicted_next_action] || ACTIONS[0];

  const currentActionEl = document.getElementById("current-action-text");
  if (currentActionEl) {
    currentActionEl.innerText = `${currentAction.icon} ${currentAction.name}`;
    currentActionEl.style.color = currentAction.color;
  }

  const predNameEl = document.getElementById("pred-action-name");
  if (predNameEl) predNameEl.innerText = predictedAction.name;

  const predConfEl = document.getElementById("pred-confidence");
  if (predConfEl) {
    const prob = (frame.probabilities[predictedAction.id] || 0.95) * 100;
    predConfEl.innerText = `${prob.toFixed(1)}%`;
  }

  ACTIONS.forEach(act => {
    const prob = (frame.probabilities[act.id] || 0) * 100;
    const fillEl = document.getElementById(`dist-fill-${act.id}`);
    const valEl = document.getElementById(`dist-val-${act.id}`);
    const rowEl = document.getElementById(`dist-row-${act.id}`);

    if (fillEl && valEl && rowEl) {
      fillEl.style.width = `${prob.toFixed(1)}%`;
      valEl.innerText = `${prob.toFixed(1)}%`;

      if (act.id === predictedAction.id) {
        rowEl.classList.add("active");
        fillEl.style.background = "#844D43";
      } else {
        rowEl.classList.remove("active");
        fillEl.style.background = "#D6C7BD";
      }
    }
  });
}

// -------------------------------------------------------------
// SCIENTIFIC PLATE RENDERING: DROSOPHILA KINEMATICS
// -------------------------------------------------------------
function renderFlyPlate() {
  const w = flyCanvas.width;
  const h = flyCanvas.height;

  // Archival Warm Canvas Background
  flyCtx.fillStyle = "#FAF7F2";
  flyCtx.fillRect(0, 0, w, h);

  // Precision Millimeter Grid (Warm Gray rules)
  flyCtx.save();
  flyCtx.strokeStyle = "#EBE4DC";
  flyCtx.lineWidth = 0.75;
  const step = 32;
  for (let x = 0; x < w; x += step) {
    flyCtx.beginPath();
    flyCtx.moveTo(x, 0);
    flyCtx.lineTo(x, h);
    flyCtx.stroke();
  }
  for (let y = 0; y < h; y += step) {
    flyCtx.beginPath();
    flyCtx.moveTo(0, y);
    flyCtx.lineTo(w, y);
    flyCtx.stroke();
  }
  flyCtx.restore();

  // Subtle Floor Shadow
  flyCtx.save();
  flyCtx.beginPath();
  flyCtx.ellipse(STATE.flyPos.x, STATE.flyPos.y + 14, 60 * STATE.flyPos.scale, 24 * STATE.flyPos.scale, 0, 0, Math.PI * 2);
  flyCtx.fillStyle = "rgba(72, 68, 63, 0.08)";
  flyCtx.fill();
  flyCtx.restore();

  // Draw Fly
  flyCtx.save();
  flyCtx.translate(STATE.flyPos.x, STATE.flyPos.y);
  flyCtx.rotate(STATE.flyPos.yaw);
  flyCtx.scale(STATE.flyPos.scale, STATE.flyPos.scale);

  const phase = STATE.gaitPhase;
  const currentAction = STATE.streamData && STATE.streamData.stream[STATE.currentIndex] 
    ? STATE.streamData.stream[STATE.currentIndex].actual_action : 1;

  drawEditorialLegs(flyCtx, phase, currentAction);
  drawEditorialAbdomen(flyCtx);
  drawEditorialHalteres(flyCtx, phase);
  drawEditorialThorax(flyCtx);
  drawEditorialWings(flyCtx, phase, currentAction);
  drawEditorialHead(flyCtx, phase, currentAction);

  flyCtx.restore();
}

// Natural Anatomical Leg Kinematics
function drawEditorialLegs(ctx, phase, actionId) {
  ctx.save();
  ctx.lineWidth = 3.2;
  ctx.strokeStyle = "#6E5034";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const isGrooming = (actionId === 5);
  const swingAmp = (actionId === 1 || actionId === 2 || actionId === 3 || actionId === 4) ? 1.0 : 0.12;

  const legs = [
    { name: "L1", x: -14, y: -18, baseAngle: -2.3, right: false, setA: true },
    { name: "L2", x: -18, y: 0,   baseAngle: -1.6, right: false, setA: false },
    { name: "L3", x: -14, y: 18,  baseAngle: -0.9, right: false, setA: true },
    { name: "R1", x: 14,  y: -18, baseAngle: -0.84, right: true,  setA: false },
    { name: "R2", x: 18,  y: 0,   baseAngle: -1.54, right: true,  setA: true },
    { name: "R3", x: 14,  y: 18,  baseAngle: -2.24, right: true,  setA: false }
  ];

  legs.forEach(leg => {
    ctx.save();
    const sign = leg.right ? 1 : -1;
    const legPhase = leg.setA ? phase : phase + Math.PI;
    let swing = Math.sin(legPhase) * 0.28 * swingAmp;

    if (actionId === 3 && leg.right) swing *= 1.7;
    if (actionId === 4 && !leg.right) swing *= 1.7;

    // Grooming Front Leg Sweep
    if (isGrooming && (leg.name === "L1" || leg.name === "R1")) {
      const sweep = Math.sin(phase * 1.5);
      const j1X = leg.x + sign * 14;
      const j1Y = leg.y - 26 + sweep * 6;
      const tipX = leg.x + sign * 5 + sweep * 8;
      const tipY = leg.y - 46 + sweep * 10;

      ctx.strokeStyle = "#825E3E";
      ctx.beginPath();
      ctx.moveTo(leg.x, leg.y);
      ctx.lineTo(j1X, j1Y);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      ctx.restore();
      return;
    }

    const angle1 = (leg.baseAngle + swing * sign) * (leg.right ? -1 : 1);
    const femurLen = 27;
    const tibiaLen = 31;
    const tarsusLen = 13;

    const j1X = leg.x + Math.cos(angle1) * femurLen * sign;
    const j1Y = leg.y + Math.sin(angle1) * femurLen;
    const angle2 = angle1 + 0.5 * sign;
    const j2X = j1X + Math.cos(angle2) * tibiaLen * sign;
    const j2Y = j1Y + Math.sin(angle2) * tibiaLen;
    const tipX = j2X + Math.cos(angle2 - 0.2 * sign) * tarsusLen * sign;
    const tipY = j2Y + Math.sin(angle2 - 0.2 * sign) * tarsusLen;

    ctx.beginPath();
    ctx.moveTo(leg.x, leg.y);
    ctx.lineTo(j1X, j1Y);
    ctx.lineTo(j2X, j2Y);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Subtle Joint Rings
    ctx.fillStyle = "#4A3220";
    ctx.beginPath();
    ctx.arc(j1X, j1Y, 2.5, 0, Math.PI * 2);
    ctx.arc(j2X, j2Y, 2.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  });

  ctx.restore();
}

function drawEditorialAbdomen(ctx) {
  ctx.save();
  const grad = ctx.createLinearGradient(0, 5, 0, 75);
  grad.addColorStop(0, "#8C572D");
  grad.addColorStop(0.5, "#C48A49");
  grad.addColorStop(1, "#42240E");

  ctx.fillStyle = grad;
  ctx.strokeStyle = "#381E0C";
  ctx.lineWidth = 1.6;

  ctx.beginPath();
  ctx.ellipse(0, 42, 21, 37, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Tergite Stripes (Sepia segmented bands)
  ctx.strokeStyle = "rgba(48, 26, 12, 0.85)";
  ctx.lineWidth = 3.2;
  [22, 33, 45, 57, 68].forEach(y => {
    ctx.beginPath();
    const halfWidth = Math.sqrt(Math.max(0, 1 - Math.pow((y - 42) / 37, 2))) * 20;
    ctx.moveTo(-halfWidth, y);
    ctx.lineTo(halfWidth, y);
    ctx.stroke();
  });
  ctx.restore();
}

function drawEditorialHalteres(ctx, phase) {
  ctx.save();
  ctx.lineWidth = 1.8;
  ctx.strokeStyle = "#A38058";
  ctx.fillStyle = "#D6BE9C";

  [-1, 1].forEach(sign => {
    const haltereAngle = phase * 2 * sign;
    const hX = sign * 18;
    const hY = 12;
    const tipX = hX + Math.cos(haltereAngle) * 11 * sign;
    const tipY = hY + Math.sin(haltereAngle) * 5;

    ctx.beginPath();
    ctx.moveTo(hX, hY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(tipX, tipY, 2.8, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawEditorialThorax(ctx) {
  ctx.save();
  const tGrad = ctx.createRadialGradient(0, -6, 2, 0, -6, 25);
  tGrad.addColorStop(0, "#A36D3B");
  tGrad.addColorStop(0.8, "#6E421E");
  tGrad.addColorStop(1, "#40240E");

  ctx.fillStyle = tGrad;
  ctx.strokeStyle = "#2E1708";
  ctx.lineWidth = 1.8;

  ctx.beginPath();
  ctx.ellipse(0, -4, 20, 23, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawEditorialWings(ctx, phase, actionId) {
  ctx.save();
  const isCourtship = (actionId === 7);
  const isEscape = (actionId === 6);

  [-1, 1].forEach(sign => {
    ctx.save();
    ctx.translate(sign * 10, -8);

    let wingAngle = sign * 0.12;
    if (isCourtship && sign === 1) {
      wingAngle = 1.45 + Math.sin(phase * 4.0) * 0.10;
      
      // Fine acoustic oscillation rings
      ctx.save();
      ctx.strokeStyle = "rgba(132, 77, 67, 0.4)";
      ctx.lineWidth = 1.4;
      for (let r = 20; r <= 55; r += 14) {
        ctx.beginPath();
        ctx.arc(38, 18, r, -0.35, 0.35);
        ctx.stroke();
      }
      ctx.restore();
    } else if (isEscape) {
      wingAngle = sign * 0.72;
    }

    ctx.rotate(wingAngle);

    // Translucent Wing Membrane (Soft Editorial Parchment Sheen)
    ctx.fillStyle = "rgba(225, 235, 245, 0.45)";
    ctx.strokeStyle = "rgba(90, 110, 130, 0.65)";
    ctx.lineWidth = 1.1;

    ctx.beginPath();
    ctx.ellipse(sign * 14, 50, 16, 52, sign * 0.12, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Natural Wing Veining
    ctx.strokeStyle = "rgba(70, 90, 110, 0.45)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(sign * 4, 0);
    ctx.lineTo(sign * 23, 92);
    ctx.moveTo(sign * 4, 15);
    ctx.lineTo(sign * 14, 78);
    ctx.stroke();

    ctx.restore();
  });
  ctx.restore();
}

function drawEditorialHead(ctx, phase, actionId) {
  ctx.save();
  ctx.translate(0, -32);

  // Head Capsule
  ctx.fillStyle = "#8C572D";
  ctx.strokeStyle = "#381E0C";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.ellipse(0, 0, 18, 12, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Natural Red Eyes
  const eyeL = ctx.createRadialGradient(-13, 0, 2, -13, 0, 11);
  eyeL.addColorStop(0, "#C72C2C");
  eyeL.addColorStop(0.8, "#851212");
  eyeL.addColorStop(1, "#400808");

  const eyeR = ctx.createRadialGradient(13, 0, 2, 13, 0, 11);
  eyeR.addColorStop(0, "#C72C2C");
  eyeR.addColorStop(0.8, "#851212");
  eyeR.addColorStop(1, "#400808");

  ctx.fillStyle = eyeL;
  ctx.beginPath();
  ctx.ellipse(-13, 0, 8, 11, -0.2, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = eyeR;
  ctx.beginPath();
  ctx.ellipse(13, 0, 8, 11, 0.2, 0, Math.PI * 2);
  ctx.fill();

  // Antennae
  ctx.strokeStyle = "#381E0C";
  ctx.lineWidth = 1.4;
  [-1, 1].forEach(sign => {
    ctx.beginPath();
    ctx.moveTo(sign * 5, -8);
    ctx.lineTo(sign * 11, -17);
    ctx.stroke();
  });
  ctx.restore();
}

// -------------------------------------------------------------
// ELECTROPHYSIOLOGY OSCILLOSCOPE (CALIBRATED CHART RECORDER)
// -------------------------------------------------------------
function renderOscilloscopePlate() {
  const w = rasterCanvas.width;
  const h = rasterCanvas.height;

  // Chart Background
  rasterCtx.fillStyle = "#FAF7F2";
  rasterCtx.fillRect(0, 0, w, h);

  const numChannels = NEURONS.length;
  const channelHeight = h / numChannels;

  // Ruling lines & channel labels
  for (let ch = 0; ch < numChannels; ch++) {
    const y = ch * channelHeight;
    rasterCtx.strokeStyle = "#E8E0D5";
    rasterCtx.lineWidth = 0.8;
    rasterCtx.beginPath();
    rasterCtx.moveTo(0, y);
    rasterCtx.lineTo(w, y);
    rasterCtx.stroke();

    rasterCtx.fillStyle = "#8C867F";
    rasterCtx.font = "500 10px JetBrains Mono";
    rasterCtx.fillText(NEURONS[ch].id, 12, y + 15);
  }

  // Draw Waveforms
  const bufferLen = rasterBuffer[0].length;
  const dx = w / bufferLen;

  for (let ch = 0; ch < numChannels; ch++) {
    const neuron = NEURONS[ch];
    const data = rasterBuffer[ch];
    const baseY = (ch + 1) * channelHeight - 4;
    const maxH = channelHeight - 8;

    rasterCtx.save();
    rasterCtx.strokeStyle = neuron.color;
    rasterCtx.lineWidth = 1.6;

    rasterCtx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = i * dx;
      const rateNorm = Math.min(1.0, data[i] / neuron.maxRate);
      const y = baseY - rateNorm * maxH;

      if (i === 0) rasterCtx.moveTo(x, y);
      else rasterCtx.lineTo(x, y);
    }
    rasterCtx.stroke();

    // Subtle wash under the trace
    rasterCtx.lineTo(w, baseY);
    rasterCtx.lineTo(0, baseY);
    rasterCtx.closePath();
    rasterCtx.fillStyle = `${neuron.color}18`;
    rasterCtx.fill();

    rasterCtx.restore();
  }

  // Telemetry Sweep Line
  rasterCtx.strokeStyle = "rgba(108, 103, 97, 0.4)";
  rasterCtx.lineWidth = 1;
  rasterCtx.setLineDash([3, 3]);
  rasterCtx.beginPath();
  rasterCtx.moveTo(w - 12, 0);
  rasterCtx.lineTo(w - 12, h);
  rasterCtx.stroke();
  rasterCtx.setLineDash([]);
}

// -------------------------------------------------------------
// VIDEO EXPORT RECORDER
// -------------------------------------------------------------
function startClipRecording() {
  const modal = document.getElementById("recording-modal");
  if (modal) modal.classList.remove("hidden");

  const stream = flyCanvas.captureStream(30);
  STATE.recordedChunks = [];

  let options = { mimeType: 'video/webm; codecs=vp9' };
  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    options = { mimeType: 'video/webm' };
  }

  try {
    STATE.mediaRecorder = new MediaRecorder(stream, options);
  } catch (e) {
    STATE.mediaRecorder = new MediaRecorder(stream);
  }

  STATE.mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) STATE.recordedChunks.push(e.data);
  };

  STATE.mediaRecorder.onstop = exportRecordedClip;
  STATE.mediaRecorder.start(100);

  const progressBar = document.getElementById("record-progress");
  let progress = 0;
  const interval = setInterval(() => {
    progress += 1;
    if (progressBar) progressBar.style.width = `${progress}%`;
    if (progress >= 100) {
      clearInterval(interval);
      stopClipRecording();
    }
  }, 100);
}

function stopClipRecording() {
  if (STATE.mediaRecorder && STATE.mediaRecorder.state !== "inactive") {
    STATE.mediaRecorder.stop();
  }
}

function exportRecordedClip() {
  const blob = new Blob(STATE.recordedChunks, { type: 'video/webm' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.style.display = "none";
  a.href = url;
  a.download = "drosophila_neural_decoding_session.webm";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    const modal = document.getElementById("recording-modal");
    if (modal) modal.classList.add("hidden");
  }, 500);
}

window.addEventListener("DOMContentLoaded", initApp);
