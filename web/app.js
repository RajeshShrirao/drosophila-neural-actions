/**
 * Drosophila Neural Action Decoder - Interactive Engine
 * Handles Canvas fly avatar rendering with articulated tripod kinematics,
 * multi-channel neural raster oscilloscope, real-time prediction decoding,
 * optogenetic stimulation injection, and in-browser clip recording.
 */

// Configuration & State
const STATE = {
  streamData: null,
  currentIndex: 0,
  isPlaying: true,
  playbackSpeed: 1.0,
  isStimulating: false,
  stimulatedNeuron: null,
  optoTimer: null,
  flyPos: { x: 420, y: 190, yaw: 0, scale: 1.0 },
  gaitPhase: 0,
  actionHistory: [],
  mediaRecorder: null,
  recordedChunks: []
};

// Canonical Actions & Colors
const ACTIONS = [
  { id: 0, name: "Quiescence / Idle", icon: "💤", color: "#6e7681" },
  { id: 1, name: "Forward Walk", icon: "🏃", color: "#38ef7d" },
  { id: 2, name: "Backward Walk (Moonwalk)", icon: "🚶‍♂️", color: "#ff7043" },
  { id: 3, name: "Turn Left", icon: "↪️", color: "#00f2fe" },
  { id: 4, name: "Turn Right", icon: "↩️", color: "#4facfe" },
  { id: 5, name: "Groom Head/Antennae", icon: "✨", color: "#c084fc" },
  { id: 6, name: "Escape Jump", icon: "🚀", color: "#fbbf24" },
  { id: 7, name: "Courtship Wing Flare", icon: "🎵", color: "#fb7185" }
];

// Neurons & Oscilloscope Channels
const NEURONS = [
  { id: "DNp09", name: "DNp09 (Forward)", color: "#38ef7d", maxRate: 80 },
  { id: "MDN", name: "MDN (Moonwalker)", color: "#ff7043", maxRate: 80 },
  { id: "DNg02_L", name: "DNg02-L (Steer Left)", color: "#00f2fe", maxRate: 70 },
  { id: "DNg02_R", name: "DNg02-R (Steer Right)", color: "#4facfe", maxRate: 70 },
  { id: "aDN", name: "aDN (Grooming)", color: "#c084fc", maxRate: 75 },
  { id: "GF", name: "Giant Fiber (Escape)", color: "#fbbf24", maxRate: 110 },
  { id: "pIP10", name: "pIP10 (Courtship)", color: "#fb7185", maxRate: 70 },
  { id: "EPG_Compass", name: "E-PG (Compass)", color: "#facc15", maxRate: 50 }
];

// Canvas Elements
const flyCanvas = document.getElementById("fly-canvas");
const flyCtx = flyCanvas.getContext("2d");
const rasterCanvas = document.getElementById("raster-canvas");
const rasterCtx = rasterCanvas.getContext("2d");

// Raster History Buffer (Stores recent rates for smooth scrolling oscilloscope)
const RASTER_BUFFER_SIZE = 300;
const rasterBuffer = Array.from({ length: NEURONS.length }, () => new Array(RASTER_BUFFER_SIZE).fill(0));

// Initialize Application
async function initApp() {
  buildNeuronLegend();
  buildProbabilityBars();
  setupEventListeners();

  try {
    const res = await fetch("data/test_neural_stream.json");
    if (!res.ok) throw new Error("Failed to load stream data");
    STATE.streamData = await res.json();
    console.log("Loaded neural stream:", STATE.streamData.stream.length, "frames");
    
    // Update accuracy stats if available
    if (STATE.streamData.metadata && STATE.streamData.metadata.benchmark) {
      const acc = (STATE.streamData.metadata.benchmark.rf_accuracy * 100).toFixed(1);
      document.getElementById("stat-accuracy").innerText = `${acc}%`;
    }
  } catch (err) {
    console.warn("Could not load data/test_neural_stream.json, falling back to synthetic online generator:", err);
    STATE.streamData = generateSyntheticStream();
  }

  // Start animation loops
  requestAnimationFrame(gameLoop);
}

// Build Neuron Legend in Oscilloscope Card
function buildNeuronLegend() {
  const container = document.getElementById("neuron-legend");
  container.innerHTML = "";
  NEURONS.forEach(n => {
    const pill = document.createElement("div");
    pill.className = "legend-pill";
    pill.innerHTML = `
      <span class="legend-color-dot" style="background: ${n.color}; box-shadow: 0 0 6px ${n.color};"></span>
      <span style="color: #c9d1d9;">${n.id}</span>
    `;
    container.appendChild(pill);
  });
}

// Build Probability Bars in Decoder Card
function buildProbabilityBars() {
  const container = document.getElementById("probabilities-list");
  container.innerHTML = "";
  ACTIONS.forEach(act => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.id = `prob-row-${act.id}`;
    row.innerHTML = `
      <div class="prob-name" title="${act.name}">${act.icon} ${act.name}</div>
      <div class="prob-bar-wrapper">
        <div class="prob-bar-fill" id="prob-fill-${act.id}" style="width: 0%;"></div>
      </div>
      <div class="prob-val" id="prob-val-${act.id}">0.0%</div>
    `;
    container.appendChild(row);
  });
}

// Setup Event Handlers
function setupEventListeners() {
  // Play/Pause
  const btnPlayPause = document.getElementById("btn-play-pause");
  btnPlayPause.addEventListener("click", () => {
    STATE.isPlaying = !STATE.isPlaying;
    btnPlayPause.innerHTML = STATE.isPlaying ? "⏸️ Pause" : "▶️ Play";
  });

  // Speed Toggle
  const btnSpeed = document.getElementById("btn-speed");
  const speeds = [0.5, 1.0, 1.5, 2.0];
  btnSpeed.addEventListener("click", () => {
    let currIdx = speeds.indexOf(STATE.playbackSpeed);
    STATE.playbackSpeed = speeds[(currIdx + 1) % speeds.length];
    btnSpeed.innerText = `${STATE.playbackSpeed.toFixed(1)}x`;
  });

  // Optogenetic Stimulation Buttons
  const optoBtns = document.querySelectorAll(".opto-btn");
  optoBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const neuronId = btn.getAttribute("data-neuron");
      triggerOptogeneticStimulation(neuronId);
      optoBtns.forEach(b => b.classList.remove("active-opto"));
      btn.classList.add("active-opto");
    });
  });

  document.getElementById("stim-resume").addEventListener("click", () => {
    resumeAutonomousStream();
    optoBtns.forEach(b => b.classList.remove("active-opto"));
  });

  // Screen / Video Recording Button
  document.getElementById("btn-record-clip").addEventListener("click", startClipRecording);
  document.getElementById("btn-stop-recording").addEventListener("click", stopClipRecording);
}

// Optogenetic Stimulation Injection
function triggerOptogeneticStimulation(neuronId) {
  STATE.isStimulating = true;
  STATE.stimulatedNeuron = neuronId;
  document.getElementById("stream-status").style.background = "rgba(192, 132, 252, 0.2)";
  document.getElementById("stream-status").style.borderColor = "#c084fc";
  document.getElementById("status-text").innerText = `OPTOGENETIC INJECTION: ${neuronId}`;
  
  if (STATE.optoTimer) clearTimeout(STATE.optoTimer);
  // Revert back after 5 seconds of stimulation
  STATE.optoTimer = setTimeout(() => {
    resumeAutonomousStream();
  }, 6000);
}

function resumeAutonomousStream() {
  STATE.isStimulating = false;
  STATE.stimulatedNeuron = null;
  document.getElementById("stream-status").style.background = "rgba(56, 239, 125, 0.1)";
  document.getElementById("stream-status").style.borderColor = "rgba(56, 239, 125, 0.25)";
  document.getElementById("status-text").innerText = "DECODING STREAM (100 Hz)";
  document.querySelectorAll(".opto-btn").forEach(b => b.classList.remove("active-opto"));
}

// Fallback Synthetic Stream Generator
function generateSyntheticStream() {
  const frames = [];
  const total = 1000;
  for (let i = 0; i < total; i++) {
    const t = i * 0.02;
    const act = Math.floor((i / 50) % 8);
    const rates = NEURONS.map(() => 5 + Math.random() * 8);
    const probs = new Array(8).fill(0.02);
    probs[act] = 0.86;
    frames.push({
      t: t,
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
// MAIN RENDER & SIMULATION LOOP
// -------------------------------------------------------------
let lastTimestamp = performance.now();

function gameLoop(timestamp) {
  const delta = (timestamp - lastTimestamp) / 1000;
  lastTimestamp = timestamp;

  if (STATE.isPlaying && STATE.streamData && STATE.streamData.stream.length > 0) {
    updateState(delta);
  }

  renderFly();
  renderRaster();

  requestAnimationFrame(gameLoop);
}

// Update State from Stream or Optogenetic Injection
function updateState(delta) {
  let frame = STATE.streamData.stream[STATE.currentIndex];

  if (STATE.isStimulating && STATE.stimulatedNeuron) {
    // Override frame rates and predictions to simulate instant optogenetic control!
    frame = JSON.parse(JSON.stringify(frame)); // Clone
    const nIdx = NEURONS.findIndex(n => n.id === STATE.stimulatedNeuron);
    if (nIdx !== -1) {
      frame.rates[nIdx] = 75.0 + Math.random() * 15.0; // High frequency drive
    }

    // Map stimulated neuron to forced action
    let forcedAction = 1;
    if (STATE.stimulatedNeuron === "MDN") forcedAction = 2; // Moonwalk
    else if (STATE.stimulatedNeuron === "GF") forcedAction = 6; // Escape Jump
    else if (STATE.stimulatedNeuron === "DNg02_L") forcedAction = 3; // Turn Left
    else if (STATE.stimulatedNeuron === "aDN") forcedAction = 5; // Groom
    else if (STATE.stimulatedNeuron === "pIP10") forcedAction = 7; // Courtship Wing Flare

    frame.predicted_next_action = forcedAction;
    frame.actual_action = forcedAction;
    frame.probabilities = new Array(8).fill(0.01);
    frame.probabilities[forcedAction] = 0.96;
  }

  // Push rates to oscilloscope buffer
  for (let i = 0; i < NEURONS.length; i++) {
    rasterBuffer[i].shift();
    rasterBuffer[i].push(frame.rates[i] || 0);
  }

  // Update Prediction UI
  updatePredictionUI(frame);

  // Update Gait & Kinematics based on active action
  const currentAction = frame.actual_action;
  updateKinematics(currentAction, delta);

  // Advance index according to playback speed
  STATE.currentIndex = (STATE.currentIndex + 1) % STATE.streamData.stream.length;
}

// Kinematic movement updates
function updateKinematics(actionId, delta) {
  const speedScale = STATE.playbackSpeed;

  switch (actionId) {
    case 1: // Forward Walk
      STATE.gaitPhase += 14 * delta * speedScale;
      STATE.flyPos.y = 190 + Math.sin(STATE.gaitPhase * 0.5) * 3;
      STATE.flyPos.yaw = Math.sin(STATE.gaitPhase * 0.25) * 0.05;
      document.getElementById("gait-phase-tag").innerText = "Tripod Forward Gait (12 Hz)";
      break;

    case 2: // Backward Walk (Moonwalk)
      STATE.gaitPhase -= 12 * delta * speedScale; // Inverted phase!
      STATE.flyPos.y = 190 + Math.sin(STATE.gaitPhase * 0.5) * 4;
      STATE.flyPos.yaw = 0;
      document.getElementById("gait-phase-tag").innerText = "Inverted Tripod Moonwalk (10 Hz)";
      break;

    case 3: // Turn Left
      STATE.gaitPhase += 10 * delta * speedScale;
      STATE.flyPos.yaw = -0.35 + Math.sin(STATE.gaitPhase * 0.5) * 0.08;
      document.getElementById("gait-phase-tag").innerText = "Asymmetric Leg Pivot: Left Yaw";
      break;

    case 4: // Turn Right
      STATE.gaitPhase += 10 * delta * speedScale;
      STATE.flyPos.yaw = 0.35 + Math.sin(STATE.gaitPhase * 0.5) * 0.08;
      document.getElementById("gait-phase-tag").innerText = "Asymmetric Leg Pivot: Right Yaw";
      break;

    case 5: // Groom Head/Antennae
      STATE.gaitPhase += 18 * delta * speedScale; // Rapid antennal strokes
      STATE.flyPos.yaw = 0;
      document.getElementById("gait-phase-tag").innerText = "Antennal & Head Sweeping Program";
      break;

    case 6: // Escape Jump
      STATE.gaitPhase += 25 * delta * speedScale;
      // Crouch then explosive jump
      const jumpCycle = Math.sin(STATE.gaitPhase * 0.4);
      STATE.flyPos.scale = jumpCycle > 0 ? 1.0 + jumpCycle * 0.35 : 0.95;
      STATE.flyPos.y = 190 - Math.max(0, jumpCycle) * 35;
      document.getElementById("gait-phase-tag").innerText = "Giant Fiber Ballistic Takeoff Jump";
      break;

    case 7: // Courtship Wing Flare
      STATE.gaitPhase += 35 * delta * speedScale; // 35 Hz wing song vibration
      STATE.flyPos.scale = 1.0;
      STATE.flyPos.yaw = 0.15;
      document.getElementById("gait-phase-tag").innerText = "Unilateral Wing Song Extension (35 Hz)";
      break;

    default: // Quiescent
      STATE.gaitPhase += 2 * delta * speedScale;
      STATE.flyPos.scale = 1.0 + Math.sin(STATE.gaitPhase) * 0.015; // Breathing
      STATE.flyPos.yaw = 0;
      document.getElementById("gait-phase-tag").innerText = "Quiescent Posture";
      break;
  }
}

// Update UI badges, prediction gauge, and probability distribution
function updatePredictionUI(frame) {
  const currentAction = ACTIONS[frame.actual_action] || ACTIONS[0];
  const predictedAction = ACTIONS[frame.predicted_next_action] || ACTIONS[0];

  // Current Action text
  const currentActionEl = document.getElementById("current-action-text");
  currentActionEl.innerText = `${currentAction.icon} ${currentAction.name}`;
  currentActionEl.style.color = currentAction.color;

  // Prediction Hero
  document.getElementById("pred-icon").innerText = predictedAction.icon;
  document.getElementById("pred-action-name").innerText = predictedAction.name;
  
  const predProb = (frame.probabilities[predictedAction.id] || 0.95) * 100;
  document.getElementById("pred-confidence").innerText = `${predProb.toFixed(1)}%`;

  // Probability Bars
  ACTIONS.forEach(act => {
    const prob = (frame.probabilities[act.id] || 0) * 100;
    const fillEl = document.getElementById(`prob-fill-${act.id}`);
    const valEl = document.getElementById(`prob-val-${act.id}`);
    const rowEl = document.getElementById(`prob-row-${act.id}`);

    if (fillEl && valEl && rowEl) {
      fillEl.style.width = `${prob.toFixed(1)}%`;
      valEl.innerText = `${prob.toFixed(1)}%`;

      if (act.id === predictedAction.id) {
        rowEl.classList.add("active");
        fillEl.style.background = `linear-gradient(90deg, #4facfe, ${act.color})`;
      } else {
        rowEl.classList.remove("active");
        fillEl.style.background = `rgba(255, 255, 255, 0.15)`;
      }
    }
  });
}

// -------------------------------------------------------------
// RENDER DROSOPHILA AVATAR & KINEMATICS ON CANVAS
// -------------------------------------------------------------
function renderFly() {
  const width = flyCanvas.width;
  const height = flyCanvas.height;

  // Clear canvas
  flyCtx.fillStyle = "#060910";
  flyCtx.fillRect(0, 0, width, height);

  // Draw background walking grid / substrate dots
  flyCtx.save();
  flyCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  flyCtx.lineWidth = 1;
  const gridSize = 40;
  for (let x = 0; x < width; x += gridSize) {
    flyCtx.beginPath();
    flyCtx.moveTo(x, 0);
    flyCtx.lineTo(x, height);
    flyCtx.stroke();
  }
  for (let y = 0; y < height; y += gridSize) {
    flyCtx.beginPath();
    flyCtx.moveTo(0, y);
    flyCtx.lineTo(width, y);
    flyCtx.stroke();
  }
  flyCtx.restore();

  // Draw floor shadow
  const shadowY = STATE.flyPos.y + 12;
  flyCtx.save();
  flyCtx.beginPath();
  flyCtx.ellipse(STATE.flyPos.x, shadowY, 65 * STATE.flyPos.scale, 28 * STATE.flyPos.scale, 0, 0, Math.PI * 2);
  flyCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
  flyCtx.fill();
  flyCtx.restore();

  // Save context for fly body transformation
  flyCtx.save();
  flyCtx.translate(STATE.flyPos.x, STATE.flyPos.y);
  flyCtx.rotate(STATE.flyPos.yaw);
  flyCtx.scale(STATE.flyPos.scale, STATE.flyPos.scale);

  const phase = STATE.gaitPhase;
  const currentAction = STATE.streamData && STATE.streamData.stream[STATE.currentIndex] ? STATE.streamData.stream[STATE.currentIndex].actual_action : 1;

  // 1. Draw 6 Articulated Legs
  drawArticulatedLegs(flyCtx, phase, currentAction);

  // 2. Draw Abdomen (Tergites & Segmentation)
  drawAbdomen(flyCtx, currentAction);

  // 3. Draw Halteres (Balance Organs)
  drawHalteres(flyCtx, phase);

  // 4. Draw Thorax (Scutum)
  drawThorax(flyCtx);

  // 5. Draw Wings
  drawWings(flyCtx, phase, currentAction);

  // 6. Draw Head, Eyes, & Antennae
  drawHead(flyCtx, phase, currentAction);

  flyCtx.restore();
}

// Draw 6 Articulated Legs with Tripod Kinematics
function drawArticulatedLegs(ctx, phase, actionId) {
  ctx.save();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#8b7355";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  // Tripod sets:
  // Tripod A: Left Fore (L1), Right Mid (R2), Left Hind (L3)
  // Tripod B: Right Fore (R1), Left Mid (L2), Right Hind (R3)
  const isTripodA = Math.sin(phase) > 0;
  const swingAmp = (actionId === 1 || actionId === 2 || actionId === 3 || actionId === 4) ? 1.0 : 0.1;
  const isGrooming = actionId === 5;

  // Leg configurations [attachmentX, attachmentY, baseAngle, isRightSide, isTripodA]
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
    let legPhase = leg.setA ? phase : phase + Math.PI;
    let swing = Math.sin(legPhase) * 0.28 * swingAmp;

    // Turn asymmetry
    if (actionId === 3 && leg.right) swing *= 1.8; // Left turn: outer right legs push harder
    if (actionId === 4 && !leg.right) swing *= 1.8; // Right turn: outer left legs push harder

    let femurLen = 28;
    let tibiaLen = 32;
    let tarsusLen = 14;

    // Special case for front leg grooming
    if (isGrooming && (leg.name === "L1" || leg.name === "R1")) {
      const groomSweep = Math.sin(phase * 1.5);
      const sign = leg.right ? 1 : -1;
      
      // Front legs elevate and sweep inward toward antennae
      const joint1X = leg.x + sign * 14;
      const joint1Y = leg.y - 28 + groomSweep * 8;
      const tipX = leg.x + sign * 6 + groomSweep * 10;
      const tipY = leg.y - 48 + groomSweep * 12;

      ctx.strokeStyle = "#a3825a";
      ctx.beginPath();
      ctx.moveTo(leg.x, leg.y);
      ctx.lineTo(joint1X, joint1Y);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // Standard Walking / Moonwalking Leg Kinematics
    const sign = leg.right ? 1 : -1;
    const angle1 = (leg.baseAngle + swing * sign) * (leg.right ? -1 : 1);
    
    // Femur
    const j1X = leg.x + Math.cos(angle1) * femurLen * sign;
    const j1Y = leg.y + Math.sin(angle1) * femurLen;

    // Tibia
    const angle2 = angle1 + 0.5 * sign;
    const j2X = j1X + Math.cos(angle2) * tibiaLen * sign;
    const j2Y = j1Y + Math.sin(angle2) * tibiaLen;

    // Tarsus (Claw)
    const tipX = j2X + Math.cos(angle2 - 0.2 * sign) * tarsusLen * sign;
    const tipY = j2Y + Math.sin(angle2 - 0.2 * sign) * tarsusLen;

    ctx.beginPath();
    ctx.moveTo(leg.x, leg.y);
    ctx.lineTo(j1X, j1Y);
    ctx.lineTo(j2X, j2Y);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Draw leg joint rings
    ctx.fillStyle = "#3d2b1f";
    ctx.beginPath();
    ctx.arc(j1X, j1Y, 3, 0, Math.PI * 2);
    ctx.arc(j2X, j2Y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  });

  ctx.restore();
}

// Draw Fly Abdomen
function drawAbdomen(ctx, actionId) {
  ctx.save();
  const grad = ctx.createLinearGradient(0, 5, 0, 75);
  grad.addColorStop(0, "#8a572a");
  grad.addColorStop(0.5, "#d4974d");
  grad.addColorStop(1, "#361e0b");

  ctx.fillStyle = grad;
  ctx.strokeStyle = "#2b1708";
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  ctx.ellipse(0, 42, 22, 38, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Abdominal Segment Stripes (Tergites)
  ctx.strokeStyle = "rgba(43, 23, 8, 0.85)";
  ctx.lineWidth = 3.5;
  const stripes = [22, 33, 45, 57, 68];
  stripes.forEach(y => {
    ctx.beginPath();
    const halfWidth = Math.sqrt(Math.max(0, 1 - Math.pow((y - 42) / 38, 2))) * 21;
    ctx.moveTo(-halfWidth, y);
    ctx.lineTo(halfWidth, y);
    ctx.stroke();
  });

  ctx.restore();
}

// Draw Halteres
function drawHalteres(ctx, phase) {
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#c29b68";
  ctx.fillStyle = "#e8c99b";

  [-1, 1].forEach(sign => {
    const haltereAngle = phase * 2 * sign;
    const hX = sign * 18;
    const hY = 12;
    const tipX = hX + Math.cos(haltereAngle) * 12 * sign;
    const tipY = hY + Math.sin(haltereAngle) * 6;

    ctx.beginPath();
    ctx.moveTo(hX, hY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(tipX, tipY, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

// Draw Thorax
function drawThorax(ctx) {
  ctx.save();
  const tGrad = ctx.createRadialGradient(0, -6, 2, 0, -6, 26);
  tGrad.addColorStop(0, "#a66e38");
  tGrad.addColorStop(0.8, "#633c19");
  tGrad.addColorStop(1, "#38200c");

  ctx.fillStyle = tGrad;
  ctx.strokeStyle = "#261508";
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.ellipse(0, -4, 21, 24, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Thoracic bristles (macrochaetae)
  ctx.fillStyle = "#1a0f05";
  [[-10, -12], [10, -12], [-14, -2], [14, -2], [-8, 8], [8, 8]].forEach(([bx, by]) => {
    ctx.beginPath();
    ctx.arc(bx, by, 1.5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

// Draw Wings (Translucent with Veins)
function drawWings(ctx, phase, actionId) {
  ctx.save();

  const isCourtship = actionId === 7;
  const isEscape = actionId === 6;

  [-1, 1].forEach(sign => {
    ctx.save();
    ctx.translate(sign * 10, -8);

    let wingAngle = sign * 0.12; // Resting wing fold angle over back

    if (isCourtship && sign === 1) {
      // Unilateral Courtship Wing Extension: Right wing opens 85 degrees and vibrates at 35 Hz!
      wingAngle = 1.45 + Math.sin(phase * 4.0) * 0.12;

      // Draw courtship acoustic soundwave ripples!
      ctx.save();
      ctx.strokeStyle = "rgba(251, 113, 133, 0.4)";
      ctx.lineWidth = 2;
      for (let r = 20; r <= 60; r += 15) {
        ctx.beginPath();
        ctx.arc(40, 20, r, -0.4, 0.4);
        ctx.stroke();
      }
      ctx.restore();
    } else if (isEscape) {
      // Escape jump: wings spread out to 45 degrees
      wingAngle = sign * 0.75 + Math.sin(phase * 2.0) * 0.15;
    }

    ctx.rotate(wingAngle);

    // Wing Membrane (Glassmorphism Iridescent fill)
    ctx.fillStyle = "rgba(230, 245, 255, 0.35)";
    ctx.strokeStyle = "rgba(100, 160, 220, 0.7)";
    ctx.lineWidth = 1.2;

    ctx.beginPath();
    ctx.ellipse(sign * 14, 52, 17, 54, sign * 0.12, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Wing Veins (Costa, Radius, Medial, Cubitus)
    ctx.strokeStyle = "rgba(70, 120, 170, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(sign * 4, 0);
    ctx.lineTo(sign * 24, 95); // Leading edge vein
    ctx.moveTo(sign * 4, 15);
    ctx.lineTo(sign * 15, 80);
    ctx.moveTo(sign * 10, 45);
    ctx.lineTo(sign * 22, 60); // Cross vein
    ctx.stroke();

    ctx.restore();
  });

  ctx.restore();
}

// Draw Head, Eyes, & Antennae
function drawHead(ctx, phase, actionId) {
  ctx.save();
  ctx.translate(0, -32);

  // Head Capsule
  ctx.fillStyle = "#8a572a";
  ctx.strokeStyle = "#38200c";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(0, 0, 19, 13, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Distinctive Red Compound Eyes
  const eyeGradL = ctx.createRadialGradient(-14, 0, 2, -14, 0, 12);
  eyeGradL.addColorStop(0, "#ff4b4b");
  eyeGradL.addColorStop(0.7, "#c41c1c");
  eyeGradL.addColorStop(1, "#590a0a");

  const eyeGradR = ctx.createRadialGradient(14, 0, 2, 14, 0, 12);
  eyeGradR.addColorStop(0, "#ff4b4b");
  eyeGradR.addColorStop(0.7, "#c41c1c");
  eyeGradR.addColorStop(1, "#590a0a");

  // Left Eye
  ctx.fillStyle = eyeGradL;
  ctx.beginPath();
  ctx.ellipse(-14, 0, 8.5, 12, -0.2, 0, Math.PI * 2);
  ctx.fill();

  // Right Eye
  ctx.fillStyle = eyeGradR;
  ctx.beginPath();
  ctx.ellipse(14, 0, 8.5, 12, 0.2, 0, Math.PI * 2);
  ctx.fill();

  // Antennae & Feathered Aristae
  ctx.strokeStyle = "#38200c";
  ctx.lineWidth = 1.5;

  [-1, 1].forEach(sign => {
    const aX = sign * 5;
    const aY = -9;
    const tipX = aX + sign * 7;
    const tipY = aY - 10;

    ctx.beginPath();
    ctx.moveTo(aX, aY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Arista branching
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX + sign * 5, tipY - 4);
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX + sign * 4, tipY + 2);
    ctx.stroke();
  });

  // Ocelli (3 simple eyes in triangle)
  ctx.fillStyle = "#ffb300";
  ctx.beginPath();
  ctx.arc(0, -4, 1.5, 0, Math.PI * 2);
  ctx.arc(-2.5, -1, 1.5, 0, Math.PI * 2);
  ctx.arc(2.5, -1, 1.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

// -------------------------------------------------------------
// RENDER MULTI-CHANNEL NEURAL RASTER / OSCILLOSCOPE
// -------------------------------------------------------------
function renderRaster() {
  const width = rasterCanvas.width;
  const height = rasterCanvas.height;

  // Background
  rasterCtx.fillStyle = "#060910";
  rasterCtx.fillRect(0, 0, width, height);

  // Draw Grid Lines
  rasterCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  rasterCtx.lineWidth = 1;
  const numChannels = NEURONS.length;
  const channelHeight = height / numChannels;

  for (let ch = 0; ch < numChannels; ch++) {
    const y = ch * channelHeight;
    rasterCtx.beginPath();
    rasterCtx.moveTo(0, y);
    rasterCtx.lineTo(width, y);
    rasterCtx.stroke();

    // Channel label
    rasterCtx.fillStyle = "rgba(139, 148, 158, 0.5)";
    rasterCtx.font = "10px JetBrains Mono";
    rasterCtx.fillText(NEURONS[ch].id, 10, y + 14);
  }

  // Draw Waveforms for each neuron channel
  const bufferLen = rasterBuffer[0].length;
  const dx = width / bufferLen;

  for (let ch = 0; ch < numChannels; ch++) {
    const neuron = NEURONS[ch];
    const data = rasterBuffer[ch];
    const baseY = (ch + 1) * channelHeight - 4;
    const maxH = channelHeight - 8;

    rasterCtx.save();
    rasterCtx.strokeStyle = neuron.color;
    rasterCtx.lineWidth = 1.8;
    rasterCtx.shadowColor = neuron.color;
    rasterCtx.shadowBlur = 6;

    rasterCtx.beginPath();
    for (let i = 0; i < bufferLen; i++) {
      const x = i * dx;
      const rateNorm = Math.min(1.0, data[i] / neuron.maxRate);
      const y = baseY - rateNorm * maxH;

      if (i === 0) rasterCtx.moveTo(x, y);
      else rasterCtx.lineTo(x, y);
    }
    rasterCtx.stroke();

    // Subtle area fill under the curve
    rasterCtx.lineTo(width, baseY);
    rasterCtx.lineTo(0, baseY);
    rasterCtx.closePath();
    rasterCtx.fillStyle = `${neuron.color}15`; // 15 hex = ~8% opacity
    rasterCtx.fill();

    rasterCtx.restore();
  }

  // Draw vertical sweep line
  rasterCtx.strokeStyle = "rgba(255, 255, 255, 0.3)";
  rasterCtx.lineWidth = 1;
  rasterCtx.setLineDash([4, 4]);
  rasterCtx.beginPath();
  rasterCtx.moveTo(width - 15, 0);
  rasterCtx.lineTo(width - 15, height);
  rasterCtx.stroke();
  rasterCtx.setLineDash([]);
}

// -------------------------------------------------------------
// IN-BROWSER VIDEO CLIP RECORDING (HTML5 MediaRecorder)
// -------------------------------------------------------------
function startClipRecording() {
  const modal = document.getElementById("recording-modal");
  modal.classList.remove("hidden");
  
  // Combine Fly Canvas stream
  const stream = flyCanvas.captureStream(30); // 30 FPS
  STATE.recordedChunks = [];

  let options = { mimeType: 'video/webm; codecs=vp9' };
  if (!MediaRecorder.isTypeSupported(options.mimeType)) {
    options = { mimeType: 'video/webm' };
  }

  try {
    STATE.mediaRecorder = new MediaRecorder(stream, options);
  } catch (e) {
    console.warn("MediaRecorder creation error:", e);
    STATE.mediaRecorder = new MediaRecorder(stream);
  }

  STATE.mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) STATE.recordedChunks.push(e.data);
  };

  STATE.mediaRecorder.onstop = exportRecordedClip;
  STATE.mediaRecorder.start(100); // 100ms slice

  // Animate progress bar over 10 seconds recording window
  const progressBar = document.getElementById("record-progress");
  let progress = 0;
  const interval = setInterval(() => {
    progress += 1;
    progressBar.style.width = `${progress}%`;
    if (progress >= 100) {
      clearInterval(interval);
      stopClipRecording();
    }
  }, 100); // 100 steps * 100ms = 10s recording
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
  a.download = "drosophila_neural_action_prediction_demo.webm";
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    document.getElementById("recording-modal").classList.add("hidden");
  }, 500);
}

// Start app on DOMContentLoaded
window.addEventListener("DOMContentLoaded", initApp);
