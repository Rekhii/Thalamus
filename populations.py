import numpy as np
from params import *
from neurons import LIFNeuron, RelayNeuron


class ThalamusNetwork:
    """
    Container for all 6 thalamic populations.
    Handles creation, spike recording, and firing rate computation.
    """

    def __init__(self):
        # Relay populations (LIF + T-channel)
        self.vl = RelayNeuron(N_VL, G_LEAK_RELAY, NOISE_RELAY)
        self.va = RelayNeuron(N_VA, G_LEAK_RELAY, NOISE_RELAY)
        self.md = RelayNeuron(N_MD, G_LEAK_RELAY, NOISE_RELAY)
        self.il = RelayNeuron(N_IL, G_LEAK_RELAY, NOISE_RELAY)

        # TRN populations (standard LIF)
        self.trn_m = LIFNeuron(N_TRN_M, G_LEAK_TRN, NOISE_TRN, I_TRN_DRIVE)
        self.trn_c = LIFNeuron(N_TRN_C, G_LEAK_TRN, NOISE_TRN, I_TRN_DRIVE)

        # Spike history: dict of {neuron_idx: [spike_times]} per population
        self.spike_history = {
            'vl': {}, 'va': {}, 'md': {}, 'il': {},
            'trn_m': {}, 'trn_c': {}
        }

        # Firing rate accumulators
        self.spike_counts = {
            'vl': np.zeros(N_VL),
            'va': np.zeros(N_VA),
            'md': np.zeros(N_MD),
            'il': np.zeros(N_IL),
            'trn_m': np.zeros(N_TRN_M),
            'trn_c': np.zeros(N_TRN_C)
        }
        self.time_elapsed = 0.0  # ms, for firing rate calculation

    def record_spikes(self, pop_name, spikes, t):
        """
        Record spike times for a population.

        pop_name: string key ('vl', 'va', 'md', 'il', 'trn_m', 'trn_c')
        spikes: boolean array from neuron update
        t: current simulation time in ms
        """
        spiked_indices = np.where(spikes)[0]
        for idx in spiked_indices:
            if idx not in self.spike_history[pop_name]:
                self.spike_history[pop_name][idx] = []
            self.spike_history[pop_name][idx].append(t)
        self.spike_counts[pop_name][spiked_indices] += 1
        self.time_elapsed = t

    def get_firing_rates(self):
        """
        Compute mean firing rate (Hz) for each population.
        Returns dict of population_name -> mean rate in Hz.
        """
        if self.time_elapsed <= 0:
            return {name: 0.0 for name in self.spike_counts}

        rates = {}
        t_seconds = self.time_elapsed / 1000.0  # ms to seconds
        for name, counts in self.spike_counts.items():
            rates[name] = np.mean(counts) / t_seconds
        return rates

    def get_channel_firing_rates(self, pop_name, n_per_channel):
        """
        Compute firing rate per action channel for channelized populations.
        Used for VL, VA, TRN-motor.

        pop_name: population name
        n_per_channel: neurons per channel
        Returns: array of firing rates, one per channel
        """
        if self.time_elapsed <= 0:
            return np.zeros(N_CHANNELS)

        t_seconds = self.time_elapsed / 1000.0
        counts = self.spike_counts[pop_name]
        channel_rates = np.zeros(N_CHANNELS)
        for ch in range(N_CHANNELS):
            start = ch * n_per_channel
            end = start + n_per_channel
            channel_rates[ch] = np.mean(counts[start:end]) / t_seconds
        return channel_rates

    def reset(self):
        """
        Reset all populations to initial state.
        Called between experiments.
        """
        # Reset relay populations
        for pop in [self.vl, self.va, self.md, self.il]:
            pop.v[:] = V_REST
            pop.h[:] = 0.0
            pop.I_T[:] = 0.0
            pop.refractory[:] = 0.0
            pop.spikes[:] = False

        # Reset TRN populations
        for pop in [self.trn_m, self.trn_c]:
            pop.v[:] = V_REST
            pop.refractory[:] = 0.0
            pop.spikes[:] = False

        # Clear spike history
        for name in self.spike_history:
            self.spike_history[name] = {}

        # Clear spike counts
        for name in self.spike_counts:
            self.spike_counts[name][:] = 0

        self.time_elapsed = 0.0

    def get_population(self, name):
        """
        Get population object by name string.
        Useful for generic operations across populations.
        """
        pop_map = {
            'vl': self.vl, 'va': self.va,
            'md': self.md, 'il': self.il,
            'trn_m': self.trn_m, 'trn_c': self.trn_c
        }
        return pop_map[name]

    def get_all_relay(self):
        """Return list of all relay populations."""
        return [self.vl, self.va, self.md, self.il]

    def get_all_trn(self):
        """Return list of all TRN populations."""
        return [self.trn_m, self.trn_c]

    def summary(self):
        """Print population sizes and current state."""
        rates = self.get_firing_rates()
        print("=" * 50)
        print("BioMind-Thalamus Network State")
        print("=" * 50)
        for name, pop in [('VL', self.vl), ('VA', self.va),
                          ('MD', self.md), ('IL', self.il),
                          ('TRN-M', self.trn_m), ('TRN-C', self.trn_c)]:
            key = name.lower().replace('-', '_')
            rate = rates[key]
            line = f"  {name:6s}: n={pop.n:4d}  rate={rate:6.1f} Hz"
            if hasattr(pop, 'h'):
                mode = pop.get_mode_fractions()
                line += f"  burst_ready={mode['burst_ready']:.0%}  tonic={mode['tonic']:.0%}"
            print(line)
        print("=" * 50)


