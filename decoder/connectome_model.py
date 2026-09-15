"""
Connectome Model & Neural Circuit Architecture
Grounded in Janelia FlyEM Drosophila Male CNS Connectome (malecns / MANC / hemibrain).
Encodes the descending command bottleneck (DNs) that mediates between central brain 
decision circuits and Ventral Nerve Cord (VNC) motor execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

# Canonical actions in Drosophila behavior repertoire
ACTION_CLASSES = [
    "Quiescence / Idle",
    "Forward Walk",
    "Backward Walk (Moonwalk)",
    "Turn Left",
    "Turn Right",
    "Groom Head/Antennae",
    "Escape Jump",
    "Courtship Wing Flare"
]

ACTION_TO_ID = {action: i for i, action in enumerate(ACTION_CLASSES)}
ID_TO_ACTION = {i: action for i, action in enumerate(ACTION_CLASSES)}

# Canonical Descending Neurons (DNs) and Central Complex heading circuits
NEURON_REGISTRY = [
    {"id": "DNp09", "name": "DNp09 (Forward Command)", "category": "Descending", "neurotransmitter": "Acetylcholine", "sign": 1.0, "primary_action": "Forward Walk", "target_vnc": "T1-T3 Leg Neuromeres"},
    {"id": "MDN", "name": "Moonwalker DN (MDN)", "category": "Descending", "neurotransmitter": "Acetylcholine", "sign": 1.0, "primary_action": "Backward Walk (Moonwalk)", "target_vnc": "T1-T3 Inverted Tripod Motor Units"},
    {"id": "DNg02_L", "name": "DNg02-Left (Steering Yaw)", "category": "Descending", "neurotransmitter": "Acetylcholine", "sign": 1.0, "primary_action": "Turn Left", "target_vnc": "Ipsilateral Leg Retractors"},
    {"id": "DNg02_R", "name": "DNg02-Right (Steering Yaw)", "category": "Descending", "neurotransmitter": "Acetylcholine", "sign": 1.0, "primary_action": "Turn Right", "target_vnc": "Contralateral Leg Retractors"},
    {"id": "aDN", "name": "Antennal DN (aDN)", "category": "Descending", "neurotransmitter": "GABA/ACh", "sign": 1.0, "primary_action": "Groom Head/Antennae", "target_vnc": "Prothoracic T1 Grooming Pattern Generator"},
    {"id": "GF", "name": "Giant Fiber (GF)", "category": "Descending", "neurotransmitter": "Electrical/ACh", "sign": 1.0, "primary_action": "Escape Jump", "target_vnc": "T2 TTMn Tergotrochanteral & DLMn Flight"},
    {"id": "pIP10", "name": "pIP10 (Courtship Song Command)", "category": "Descending", "neurotransmitter": "Acetylcholine", "sign": 1.0, "primary_action": "Courtship Wing Flare", "target_vnc": "Mesothoracic T2 Wing Steering Neuromere"},
    {"id": "EPG_Compass", "name": "E-PG Heading Compass", "category": "Central Complex", "neurotransmitter": "Acetylcholine", "sign": 1.0, "primary_action": "Steering Modulation", "target_vnc": "Fan-shaped Body & Bridge"},
    {"id": "PEN_Steering", "name": "P-EN Phase Shift Steering", "category": "Central Complex", "neurotransmitter": "GABA", "sign": -1.0, "primary_action": "Angular Velocity", "target_vnc": "DNg Steering DNs"},
    {"id": "MBON_Valence", "name": "MBON (Mushroom Body Valence)", "category": "Mushroom Body", "neurotransmitter": "Glutamate/GABA", "sign": -1.0, "primary_action": "Context/Arousal", "target_vnc": "Pre-motor DN inputs"}
]

NEURON_NAMES = [n["id"] for n in NEURON_REGISTRY]
NUM_NEURONS = len(NEURON_NAMES)
NEURON_INDEX = {n_id: i for i, n_id in enumerate(NEURON_NAMES)}


@dataclass
class ConnectomeCircuit:
    """Connectome synaptic connectivity graph between command circuits and descending neurons."""
    neurons: List[Dict] = field(default_factory=lambda: NEURON_REGISTRY)
    adj_matrix: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.adj_matrix is None:
            self.adj_matrix = self._build_synaptic_weights()

    def _build_synaptic_weights(self) -> np.ndarray:
        """
        Builds calibrated synaptic weight matrix reflecting known Janelia Male CNS connectomics.
        Positive: excitatory (cholinergic), Negative: mutual inhibition (e.g. MDN inhibits forward DNp09).
        """
        W = np.zeros((NUM_NEURONS, NUM_NEURONS), dtype=np.float32)
        
        # Mutual exclusion between forward walk (DNp09) and backward walk (MDN)
        idx_dnp09 = NEURON_INDEX["DNp09"]
        idx_mdn = NEURON_INDEX["MDN"]
        W[idx_mdn, idx_dnp09] = -2.8  # MDN strongly suppresses forward walking
        W[idx_dnp09, idx_mdn] = -1.5  # Forward activity provides cross-inhibition
        
        # Steering modulation from central complex (EPG/PEN to DNg02)
        idx_epg = NEURON_INDEX["EPG_Compass"]
        idx_pen = NEURON_INDEX["PEN_Steering"]
        idx_dng_l = NEURON_INDEX["DNg02_L"]
        idx_dng_r = NEURON_INDEX["DNg02_R"]
        
        W[idx_epg, idx_pen] = 1.8
        W[idx_pen, idx_dng_l] = 2.4
        W[idx_pen, idx_dng_r] = -2.4  # Differential steering drive
        W[idx_dng_l, idx_dng_r] = -1.2  # Reciprocal lateral inhibition
        W[idx_dng_r, idx_dng_l] = -1.2
        
        # Giant fiber escape override (GF inhibits routine walking & grooming)
        idx_gf = NEURON_INDEX["GF"]
        idx_adn = NEURON_INDEX["aDN"]
        W[idx_gf, idx_dnp09] = -3.5
        W[idx_gf, idx_mdn] = -3.5
        W[idx_gf, idx_adn] = -3.0
        
        # Self-excitation / persistence (recurrent dynamics within command loops)
        for i in range(NUM_NEURONS):
            W[i, i] = 0.65
            
        return W

    def get_circuit_summary(self) -> Dict:
        """Export circuit metadata for documentation and web visualization."""
        return {
            "num_neurons": NUM_NEURONS,
            "neuron_registry": self.neurons,
            "actions": ACTION_CLASSES,
            "action_map": {a: i for i, a in enumerate(ACTION_CLASSES)},
            "synaptic_weights": self.adj_matrix.tolist()
        }
