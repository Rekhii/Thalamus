import numpy as np

# Simulation constants
DT = 0.1  # ms, integration timestep (0.1 ms needed for T-channel dynamics)

# LIF base parameters (Shared By All Neurons)
# These are the same fundamental LIF parameters from BioMind-BG
CM = 1.0            # Membrane capacitance
V_REST = -65.0      # mV, Resting Membrane Potential
V_THRESH = -50.0    # mV, Spike Threshold
V_RESET = -70.0     # mV, Post spike reset voltage
TAU_REF = 2.0       # ms, Absolute Refractory Period

# T-Type calcium channels (parameters relay neurons only)
G_T = 0.05         # mS/cm^2, max T-channel conductance
V_T_HALF = -60.0   # mV, half-activation voltage for T-channel
V_H_HALF = -75.0   # mV, half-inactivation voltage for h gate
K_T = 6.2          # mV, activation slope factor
K_H = 4.0          # mV, inactivation slope factor
TAU_H_BASE = 100.0 # ms, base time constant for h de-inactivation
E_CA = 120.0       # mV, calcium reversal potential

# Leak conductance (per neuron type)
G_LEAK_RELAY = 0.05   # mS/cm^2, relay neurons (lower = higher input resistance)
G_LEAK_TRN = 0.1      # mS/cm^2, TRN neurons (higher = faster response)
E_LEAK = -65.0        # mV, leak reversal (equals V_REST)

# Synaptic reversal potential
E_AMPA = 0.0       # mV, excitatory (glutamate AMPA)
E_NMDA = 0.0       # mV, excitatory (glutamate NMDA)
E_GABA = -80.0     # mV, inhibitory (GABA-A)

# Synaptic time constants
TAU_AMPA = 2.0     # ms, fast excitatory decay
TAU_NMDA = 100.0   # ms, slow excitatory decay
TAU_GABA = 5.0     # ms, inhibitory decay (slightly longer than cortical GABA)

# NMDA voltage dependence (Mg2+ block)
NMDA_MG = 1.0      # mM, external magnesium concentration
NMDA_ALPHA = 0.062  # per mV, Mg block slope
NMDA_BETA = 3.57   # scaling factor

# Population sizes
N_VL = 200         # ventral lateral relay neurons
N_VA = 150         # ventral anterior relay neurons
N_MD = 250         # mediodorsal relay neurons
N_IL = 120         # intralaminar relay neurons
N_TRN_M = 70       # TRN motor sector
N_TRN_C = 75       # TRN cognitive sector

# Number of action channels (must match BioMind-BG)
N_CHANNELS = 3

# Neurons per channel (for channelized populations)
N_VL_PER_CH = N_VL // N_CHANNELS    # ~66 per channel
N_VA_PER_CH = N_VA // N_CHANNELS    # 50 per channel
N_TRN_M_PER_CH = N_TRN_M // N_CHANNELS  # ~23 per channel
N_BG_PRE = 20               # number of presynaptic BG neurons per channel

# Baseline firing rates (Hz, for validation)
FR_VL_SUPPRESSED = 3.0    # Hz, VL under GPi inhibition
FR_VA_SUPPRESSED = 3.0    # Hz, VA under GPi inhibition
FR_MD_SUPPRESSED = 7.0    # Hz, MD under SNr inhibition (less suppressed)
FR_IL_BASELINE = 12.0     # Hz, intralaminar tonic baseline
FR_TRN_BASELINE = 20.0    # Hz, TRN spontaneous firing

# GPi/SNr tonic inhibition parameters
GPi_TONIC_RATE = 70.0     # Hz, resting GPi firing rate (from BioMind-BG)
SNr_TONIC_RATE = 65.0     # Hz, resting SNr firing rate
G_GPi_VL = 0.008           # mS/cm^2, GPi -> VL synaptic conductance
G_GPi_VA = 0.006           # mS/cm^2, GPi -> VA synaptic conductance
G_SNr_MD = 0.008           # mS/cm^2, SNr -> MD synaptic conductance
G_SNr_IL = 0.007           # mS/cm^2, SNr -> IL synaptic conductance

# TRN connection weights
G_RELAY_TRN = 0.008       # mS/cm^2, relay -> TRN (feedforward collateral)
G_RELAY_TRN_C = 0.002     # cognitive sector, lower to compensate for more relay neurons
G_TRN_RELAY = 0.003       # mS/cm^2, TRN -> relay (feedback inhibition)
G_TRN_LATERAL = 0.0015    # mS/cm^2, TRN -> relay (lateral inhibition, cross-channel)

# Corticothalamic feedback (simulated as external input)
G_CTX_RELAY = 0.005        # mS/cm^2, cortex L6 -> relay (modulator, weak)
G_CTX_TRN = 0.010          # mS/cm^2, cortex L6 -> TRN (drives attentional gating)
G_CTX_HO_DRIVE = 0.030     # mS/cm^2, cortex L5 -> MD/IL (driver, strong)
CTX_BACKGROUND_RATE = 20.0 # Hz, baseline cortical feedback rate

# Neuromodulatory control (brainstem ACh/NE)
NEUROMOD_WAKE = 1.0       # full wakefulness, relay neurons depolarized
NEUROMOD_DROWSY = 0.5     # partial, mixed burst/tonic
NEUROMOD_SLEEP = 0.0      # deep sleep, relay neurons hyperpolarized
I_NEUROMOD_MAX = 1.0     # nA, max depolarizing current from neuromodulators

# Noise
NOISE_RELAY = 2.0        # nA, background noise current (relay)
NOISE_TRN = 2.5          # nA, background noise current (TRN, slightly higher)

# TRN baseline excitatory drive
# TRN receives tonic cortical and relay excitation in vivo
# This current represents that background drive
I_TRN_DRIVE = 1.0      # nA

# Connection probability
P_GPi_VL = 0.3            # GPi -> VL (within-channel, focused)
P_GPi_VA = 0.3            # GPi -> VA (within-channel)
P_SNr_MD = 0.2            # SNr -> MD (broader)
P_SNr_IL = 0.15           # SNr -> IL (diffuse)
P_RELAY_TRN = 0.4         # relay -> TRN (collateral, same sector)
P_TRN_RELAY_FEED = 0.3    # TRN -> relay (feedback, same channel)
P_TRN_RELAY_LAT = 0.15    # TRN -> relay (lateral, cross-channel)
P_CTX_RELAY = 0.2         # cortex -> relay (modulatory)
P_CTX_TRN = 0.25          # cortex -> TRN (attentional)
P_CTX_HO = 0.3            # cortex L5 -> MD/IL (driver)

# Burst detection parameters
BURST_ISI_THRESHOLD = 8.0  # ms, max inter-spike interval within a burst
BURST_MIN_SPIKES = 2       # minimum spikes to count as a burst
