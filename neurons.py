import numpy as np
from params import *

class LIFNeuron:
    """
    Standard Leaky Integrate-and-Fire neuron.
    Used for TRN neurons.
    Same model as BioMind-BG but organized as a class.
    """
    def __init__(self,n, g_leak, noise_std, I_drive=0.0):
        self.n = n                                # number of neurons
        self.v = V_REST * np.ones(n)              # membrane potential (mV)
        self.g_leak = g_leak                      # leak conductance (mS/cm^2)
        self.noise_std = noise_std                # noise current std (nA)
        self.I_drive = I_drive                    # baseline tonic drive current (nA)
        self.refractory = np.zeros(n)             # refractory timer (ms)
        self.spikes = np.zeros(n, dtype=bool)     # spike flags this timestep

    def update(self, I_syn):
        """
        Advance all neurons by one timestep.

        I_syn: total synaptic current array (nA), shape (n,)
        Returns: spike boolean array
        """
        # Reset spike flags
        self.spikes[:] = False

        # Find neurons not in refractory period
        active = self.refractory <= 0

        # Leak current: pulls voltage toward E_LEAK
        I_leak = -self.g_leak * (self.v[active] - E_LEAK)

        # Noise current: Gaussian, independent per neuron per timestep
        I_noise = self.noise_std * np.random.randn(np.sum(active))

        # Total current
        I_total = I_leak + I_syn[active] + I_noise + self.I_drive

        # Voltage update: dV/dt = I_total / CM
        self.v[active] += (I_total / CM) * DT

        # Spike detection
        spiked = np.where(active)[0][self.v[active] >= V_THRESH]
        self.spikes[spiked] = True
        self.v[spiked] = V_RESET
        self.refractory[spiked] = TAU_REF

        # Decrement refractory timers
        self.refractory -= DT
        self.refractory = np.maximum(self.refractory, 0)

        return self.spikes

class RelayNeuron(LIFNeuron):
    """
    Thalamic relay neuron: LIF + T-type calcium current.
    Supports burst mode (h high, hyperpolarized) and
    tonic mode (h low, depolarized).
    """

    def __init__(self, n, g_leak, noise_std):
        super().__init__(n, g_leak, noise_std)
        self.h = np.zeros(n)          # T-channel de-inactivation gate (0 to 1)
        self.I_T = np.zeros(n)        # T-type calcium current (for logging)

    def update(self, I_syn, neuromod=NEUROMOD_WAKE):
        """
        Advance relay neurons by one timestep.
        Adds T-type calcium current on top of LIF dynamics.

        I_syn: total synaptic current array (nA), shape (n,)
        neuromod: float 0-1, brainstem arousal level
        Returns: spike boolean array
        """
        # Reset spike flags
        self.spikes[:] = False

        # Find neurons not in refractory period
        active = self.refractory <= 0

        # T-channel de-inactivation gate (h)
        h_inf = 1.0 / (1.0 + np.exp((self.v[active] - V_H_HALF) / K_H))

        # tau_h: voltage-dependent time constant
        tau_h = TAU_H_BASE / (np.exp((self.v[active] - V_H_HALF) / (2 * K_H))
                              + np.exp(-(self.v[active] - V_H_HALF) / (2 * K_H)))

        # Update h toward h_inf with time constant tau_h
        self.h[active] += ((h_inf - self.h[active]) / tau_h) * DT

        # T-channel activation gate (m_inf, instantaneous)
        m_inf = 1.0 / (1.0 + np.exp(-(self.v[active] - V_T_HALF) / K_T))

        # T-type calcium current (inward = depolarizing = positive in our convention)
        self.I_T[active] = G_T * m_inf * self.h[active] * (E_CA - self.v[active])

        # Asymmetric neuromodulatory current:
        # Wake (neuromod=1.0): +I_NEUROMOD_MAX (full depolarization)
        # Neutral (neuromod=0.5): 0 (no effect)
        # Sleep (neuromod=0.0): -I_NEUROMOD_MAX * 0.55 (moderate hyperpolarization)
        if neuromod >= 0.5:
            I_neuromod = (neuromod - 0.5) * 2.0 * I_NEUROMOD_MAX
        else:
            I_neuromod = -(0.5 - neuromod) * 2.0 * I_NEUROMOD_MAX * 0.55

        # Leak current
        I_leak = -self.g_leak * (self.v[active] - E_LEAK)

        # Noise
        I_noise = self.noise_std * np.random.randn(np.sum(active))

        # Total current
        I_total = I_leak + I_syn[active] + I_noise + self.I_T[active] + I_neuromod + self.I_drive

        # Voltage update
        self.v[active] += (I_total / CM) * DT

        # Spike detection
        spiked = np.where(active)[0][self.v[active] >= V_THRESH]
        self.spikes[spiked] = True
        self.v[spiked] = V_RESET
        self.refractory[spiked] = TAU_REF

        # Decrement refractory timers
        self.refractory -= DT
        self.refractory = np.maximum(self.refractory, 0)

        return self.spikes

    def get_burst_mask(self, spike_times_ms):
        burst_mask = np.zeros(self.n, dtype=bool)
        for idx, times in spike_times_ms.items():
            if len(times) >= BURST_MIN_SPIKES:
                recent = times[-BURST_MIN_SPIKES:]
                isi = recent[-1] - recent[0]
                if isi < BURST_ISI_THRESHOLD * (BURST_MIN_SPIKES - 1):
                    burst_mask[idx] = True
        return burst_mask

    def get_h_mean(self):
        return np.mean(self.h)

    def get_mode_fractions(self):
        burst_ready = np.mean(self.h > 0.5)
        tonic = 1.0 - burst_ready
        return {'burst_ready': burst_ready, 'tonic': tonic}