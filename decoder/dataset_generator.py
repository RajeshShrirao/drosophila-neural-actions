"""
Neural Time-Series & Behavioral State Generator
Generates biologically calibrated population dynamics of Drosophila descending neurons (DNs)
coupled to motor state transitions with realistic pre-motor synaptic lead times (50ms - 150ms).
"""

import numpy as np
from typing import Dict, Tuple
from decoder.connectome_model import (
    ACTION_CLASSES,
    ACTION_TO_ID,
    NEURON_REGISTRY,
    NEURON_INDEX,
    NUM_NEURONS,
    ConnectomeCircuit
)


class FlyNeuralDataGenerator:
    """
    Simulates continuous neural population dynamics (100 Hz / 10ms binning)
    with biological lead times where descending command activity precedes motor execution.
    """

    def __init__(self, dt_sec: float = 0.01, random_seed: int = 42):
        self.dt = dt_sec  # 10ms time step (100 Hz)
        self.rng = np.random.default_rng(random_seed)
        self.circuit = ConnectomeCircuit()
        self.W = self.circuit.adj_matrix
        self.tau = 0.035  # 35ms synaptic membrane time constant

    def generate_session(
        self,
        duration_sec: float = 120.0,
        avg_action_duration_sec: float = 2.2,
        lead_time_ms: float = 100.0
    ) -> Dict[str, np.ndarray]:
        """
        Generates a continuous behavioral session.
        lead_time_ms: How many milliseconds BEFORE motor onset the descending neurons begin firing.
        """
        num_steps = int(duration_sec / self.dt)
        lead_steps = max(1, int((lead_time_ms / 1000.0) / self.dt))
        
        # 1. Generate sequence of behavioral states (Motor Ground Truth)
        motor_states = np.zeros(num_steps, dtype=np.int32)
        step = 0
        
        # Transition probabilities between actions
        state_probs = [
            0.20,  # Idle
            0.30,  # Forward Walk
            0.10,  # Moonwalk Backward
            0.12,  # Turn Left
            0.12,  # Turn Right
            0.08,  # Groom
            0.04,  # Escape Jump
            0.04   # Courtship Wing Flare
        ]
        
        while step < num_steps:
            action_id = self.rng.choice(len(ACTION_CLASSES), p=state_probs)
            # Duration in steps (exponential distribution around target)
            dur = max(int(0.6 / self.dt), int(self.rng.exponential(avg_action_duration_sec) / self.dt))
            end_step = min(num_steps, step + dur)
            motor_states[step:end_step] = action_id
            step = end_step

        # 2. Derive neural drive commands with pre-motor lead time
        # Intention/command drive starts `lead_steps` before motor execution begins
        command_states = np.zeros(num_steps, dtype=np.int32)
        command_states[:-lead_steps] = motor_states[lead_steps:]
        command_states[-lead_steps:] = motor_states[-1]

        # 3. Simulate neural firing rates r(t) using recurrent rate dynamics
        rates = np.zeros((num_steps, NUM_NEURONS), dtype=np.float32)
        baseline_rates = np.array([4.0, 2.5, 3.0, 3.0, 2.0, 0.8, 1.5, 8.0, 5.0, 6.0], dtype=np.float32)
        rates[0] = baseline_rates + self.rng.normal(0, 0.5, size=NUM_NEURONS)

        for t in range(1, num_steps):
            curr_cmd = command_states[t]
            ext_drive = np.zeros(NUM_NEURONS, dtype=np.float32)
            
            # Action-specific neural drive
            if curr_cmd == ACTION_TO_ID["Forward Walk"]:
                ext_drive[NEURON_INDEX["DNp09"]] += 42.0
                ext_drive[NEURON_INDEX["EPG_Compass"]] += 12.0
            elif curr_cmd == ACTION_TO_ID["Backward Walk (Moonwalk)"]:
                ext_drive[NEURON_INDEX["MDN"]] += 55.0  # Moonwalker burst
            elif curr_cmd == ACTION_TO_ID["Turn Left"]:
                ext_drive[NEURON_INDEX["DNp09"]] += 20.0
                ext_drive[NEURON_INDEX["DNg02_L"]] += 48.0
                ext_drive[NEURON_INDEX["PEN_Steering"]] += 25.0
            elif curr_cmd == ACTION_TO_ID["Turn Right"]:
                ext_drive[NEURON_INDEX["DNp09"]] += 20.0
                ext_drive[NEURON_INDEX["DNg02_R"]] += 48.0
                ext_drive[NEURON_INDEX["PEN_Steering"]] -= 25.0
            elif curr_cmd == ACTION_TO_ID["Groom Head/Antennae"]:
                ext_drive[NEURON_INDEX["aDN"]] += 50.0  # Antennal grooming sweep
            elif curr_cmd == ACTION_TO_ID["Escape Jump"]:
                ext_drive[NEURON_INDEX["GF"]] += 90.0   # Giant fiber massive spike
            elif curr_cmd == ACTION_TO_ID["Courtship Wing Flare"]:
                ext_drive[NEURON_INDEX["pIP10"]] += 52.0
                ext_drive[NEURON_INDEX["MBON_Valence"]] += 18.0

            # Recurrent network update: dr/dt = (-r + f(W*r + ext)) / tau
            synaptic_input = np.dot(self.W, rates[t - 1]) + ext_drive + baseline_rates
            target_rate = np.maximum(0.0, synaptic_input)  # ReLU activation
            
            # Euler integration + Poisson-like noise
            dr = (self.dt / self.tau) * (target_rate - rates[t - 1])
            noise = self.rng.normal(0.0, np.sqrt(np.maximum(0.1, rates[t - 1])) * 0.45)
            rates[t] = np.clip(rates[t - 1] + dr + noise, 0.0, 160.0)

        # 4. Synthesize simulated calcium fluorescence dF/F (low-pass filter of firing rate)
        tau_calcium = 0.25  # 250ms GCaMP indicator decay
        calcium_dff = np.zeros_like(rates)
        for t in range(1, num_steps):
            dca = (self.dt / tau_calcium) * (rates[t] / 30.0 - calcium_dff[t - 1])
            calcium_dff[t] = np.maximum(0.0, calcium_dff[t - 1] + dca)

        time_axis = np.arange(num_steps) * self.dt

        return {
            "time_sec": time_axis,
            "firing_rates": rates,          # Shape: (num_steps, NUM_NEURONS)
            "calcium_dff": calcium_dff,     # Shape: (num_steps, NUM_NEURONS)
            "motor_states": motor_states,   # Shape: (num_steps,) - Actual physical movement
            "command_states": command_states,
            "lead_time_ms": lead_time_ms,
            "dt_sec": self.dt,
            "action_names": ACTION_CLASSES,
            "neuron_names": [n["id"] for n in NEURON_REGISTRY]
        }


def create_lookahead_dataset(
    session_data: Dict[str, np.ndarray],
    lookahead_ms: float = 100.0,
    window_ms: float = 150.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepares (X, y) where:
    - X[t] is window of neural activity ending at current time t
    - y[t] is the fly's NEXT motor action at time t + lookahead_ms
    """
    rates = session_data["firing_rates"]
    motor_states = session_data["motor_states"]
    dt = session_data["dt_sec"]

    lookahead_steps = max(1, int((lookahead_ms / 1000.0) / dt))
    window_steps = max(1, int((window_ms / 1000.0) / dt))

    X_list = []
    y_list = []

    # Slide window across session
    for t in range(window_steps, len(rates) - lookahead_steps):
        # Extract features: mean rate, max rate, and rate trend (slope) in window
        window = rates[t - window_steps : t]
        mean_feat = np.mean(window, axis=0)
        max_feat = np.max(window, axis=0)
        current_feat = rates[t - 1]
        
        # Combine into feature vector
        features = np.concatenate([mean_feat, max_feat, current_feat])
        
        # Target is the NEXT action at t + lookahead_steps
        target_action = motor_states[t + lookahead_steps]
        
        X_list.append(features)
        y_list.append(target_action)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)
