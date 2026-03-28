import numpy as np
from params import *

class Synapse:
    """
    Single synaptic pathway between two populations.
    Tracks conductance state with exponential decay.
    """

    def __init__(self, n_pre, n_post, g_max, tau, e_rev, prob, name=""):
        self.name = name
        self.n_pre = n_pre
        self.n_post = n_post
        self.g_max = g_max  # max conductance per synapse
        self.tau = tau  # decay time constant
        self.e_rev = e_rev  # reversal potential
        self.g = np.zeros(n_post)  # current conductance state per postsynaptic neuron

        # Build connection matrix: (n_post, n_pre) boolean
        self.W = np.random.rand(n_post, n_pre) < prob

    def update(self, pre_spikes, post_v):
        """
        Update synaptic conductance and compute current.

        pre_spikes: boolean array (n_pre,)
        post_v: voltage array of postsynaptic neurons (n_post,)
        Returns: synaptic current array (n_post,)
        """
        # Decay existing conductance
        self.g *= np.exp(-DT / self.tau)

        # Add conductance from presynaptic spikes
        # Each spike increments g by g_max for connected postsynaptic neurons
        if np.any(pre_spikes):
            self.g += self.g_max * self.W[:, pre_spikes].sum(axis=1)

        # Compute synaptic current: I = g * (E_rev - V)
        I_syn = self.g * (self.e_rev - post_v)

        return I_syn

    def reset(self):
        """Clear conductance state."""
        self.g[:] = 0.0

# NMDA synapse with Mg2+ block
class NMDASynapse(Synapse):
    """
    NMDA synapse with voltage-dependent magnesium block.
    At resting potential, Mg2+ ions block the channel.
    Depolarization relieves the block.
    This creates a coincidence detector: presynaptic spike
    AND postsynaptic depolarization both needed for current flow.
    """

    def update(self, pre_spikes, post_v):
        """
        Same as Synapse but multiplies current by Mg2+ block factor.
        """
        # Decay existing conductance
        self.g *= np.exp(-DT / self.tau)

        # Add conductance from presynaptic spikes
        if np.any(pre_spikes):
            self.g += self.g_max * self.W[:, pre_spikes].sum(axis=1)

        # Mg2+ block: voltage-dependent gating factor (0 to 1)
        # Near rest (-65 mV): mg_gate ~ 0.07 (93% blocked)
        # At -40 mV: mg_gate ~ 0.29
        # At -20 mV: mg_gate ~ 0.67
        # At 0 mV: mg_gate ~ 0.88 (nearly unblocked)
        mg_gate = 1.0 / (1.0 + (NMDA_MG / NMDA_BETA) * np.exp(-NMDA_ALPHA * post_v))

        # Synaptic current with Mg block
        I_syn = self.g * mg_gate * (self.e_rev - post_v)

        return I_syn

# Connection builder — all 18 pathways
class ThalamusConnections:
    """
    Builds and manages all 18 synaptic pathways.
    Organizes by source type: BG inputs, cortical inputs,
    relay-TRN loops, TRN-relay inhibition.
    """

    def __init__(self):
        """
        GROUP 1: Basal ganglia -> Thalamus (GABAergic)
        Tonic inhibition from GPi/SNr
        These are the pathways BioMind-BG controls

        Pathway 1: GPi -> VL (motor action gating)
        Channelized: each BG channel inhibits its VL channel
        """

        self.gpi_vl = []
        for ch in range(N_CHANNELS):
            syn = Synapse(
                n_pre=N_BG_PRE,     # GPi channel modeled as population of rate inputs
                n_post=N_VL_PER_CH,
                g_max=G_GPi_VL,
                tau=TAU_GABA,
                e_rev=E_GABA,
                prob=P_GPi_VL,
                name=f"GPi->VL_ch{ch}"
            )
            self.gpi_vl.append(syn)

        # Pathway 2: GPi -> VA (cognitive action gating)
        self.gpi_va = []
        for ch in range(N_CHANNELS):
            syn = Synapse(
                n_pre=N_BG_PRE,
                n_post=N_VA_PER_CH,
                g_max=G_GPi_VA,
                tau=TAU_GABA,
                e_rev=E_GABA,
                prob=P_GPi_VA,
                name=f"GPi->VA_ch{ch}"
            )
            self.gpi_va.append(syn)

        # Pathway 3: SNr -> MD (executive gating)
        self.snr_md = Synapse(
            n_pre=N_BG_PRE,
            n_post=N_MD,
            g_max=G_SNr_MD,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_SNr_MD,
            name="SNr->MD"
        )

        # Pathway 4: SNr -> IL (arousal gating)
        self.snr_il = Synapse(
            n_pre=N_BG_PRE,
            n_post=N_IL,
            g_max=G_SNr_IL,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_SNr_IL,
            name="SNr->IL"
        )

        """
         GROUP 2: Cortex -> Thalamus (Glutamatergic)
         Modulatory feedback (L6) and higher-order drivers (L5)
         Simulated as external Poisson inputs until Component 3
        """


        # Pathway 5: Cortex L6 -> VL (modulatory feedback, AMPA)
        self.ctx_vl_ampa = Synapse(
            n_pre=1,
            n_post=N_VL,
            g_max=G_CTX_RELAY,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_CTX_RELAY,
            name="CtxL6->VL_AMPA"
        )

        # Pathway 5b: Cortex L6 -> VL (modulatory feedback, NMDA)
        self.ctx_vl_nmda = NMDASynapse(
            n_pre=1,
            n_post=N_VL,
            g_max=G_CTX_RELAY * 0.5,
            tau=TAU_NMDA,
            e_rev=E_NMDA,
            prob=P_CTX_RELAY,
            name="CtxL6->VL_NMDA"
        )

        # Pathway 6: Cortex L6 -> VA (modulatory feedback)
        self.ctx_va_ampa = Synapse(
            n_pre=1,
            n_post=N_VA,
            g_max=G_CTX_RELAY,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_CTX_RELAY,
            name="CtxL6->VA_AMPA"
        )

        self.ctx_va_nmda = NMDASynapse(
            n_pre=1,
            n_post=N_VA,
            g_max=G_CTX_RELAY * 0.5,
            tau=TAU_NMDA,
            e_rev=E_NMDA,
            prob=P_CTX_RELAY,
            name="CtxL6->VA_NMDA"
        )

        # Pathway 7: Cortex L5 -> MD (higher-order driver, AMPA)
        self.ctx_md_ampa = Synapse(
            n_pre=1,
            n_post=N_MD,
            g_max=G_CTX_HO_DRIVE,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_CTX_HO,
            name="CtxL5->MD_AMPA"
        )

        self.ctx_md_nmda = NMDASynapse(
            n_pre=1,
            n_post=N_MD,
            g_max=G_CTX_HO_DRIVE * 0.5,
            tau=TAU_NMDA,
            e_rev=E_NMDA,
            prob=P_CTX_HO,
            name="CtxL5->MD_NMDA"
        )

        # Pathway 8: Cortex L5 -> IL (arousal driver, AMPA)
        self.ctx_il_ampa = Synapse(
            n_pre=1,
            n_post=N_IL,
            g_max=G_CTX_HO_DRIVE,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_CTX_HO,
            name="CtxL5->IL_AMPA"
        )

        self.ctx_il_nmda = NMDASynapse(
            n_pre=1,
            n_post=N_IL,
            g_max=G_CTX_HO_DRIVE * 0.5,
            tau=TAU_NMDA,
            e_rev=E_NMDA,
            prob=P_CTX_HO,
            name="CtxL5->IL_NMDA"
        )

        # Pathway 13: Cortex L6 -> TRN-motor (attentional gating)
        self.ctx_trn_m = Synapse(
            n_pre=1,
            n_post=N_TRN_M,
            g_max=G_CTX_TRN,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_CTX_TRN,
            name="CtxL6->TRN_M"
        )

        # Pathway 14: Cortex L6 -> TRN-cognitive (attentional gating)
        self.ctx_trn_c = Synapse(
            n_pre=1,
            n_post=N_TRN_C,
            g_max=G_CTX_TRN,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_CTX_TRN,
            name="CtxL6->TRN_C"
        )
        """
         GROUP 3: Relay -> TRN (feedforward collaterals)
         Every relay neuron excites TRN on its way to cortex
        """

        # Pathway 9: VL -> TRN-motor
        self.vl_trn = Synapse(
            n_pre=N_VL,
            n_post=N_TRN_M,
            g_max=G_RELAY_TRN,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_RELAY_TRN,
            name="VL->TRN_M"
        )

        # Pathway 10: VA -> TRN-motor
        self.va_trn = Synapse(
            n_pre=N_VA,
            n_post=N_TRN_M,
            g_max=G_RELAY_TRN,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_RELAY_TRN,
            name="VA->TRN_M"
        )

        # Pathway 11: MD -> TRN-cognitive (FIXED: uses G_RELAY_TRN_C)
        self.md_trn = Synapse(
            n_pre=N_MD,
            n_post=N_TRN_C,
            g_max=G_RELAY_TRN_C,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_RELAY_TRN,
            name="MD->TRN_C"
        )

        # Pathway 12: IL -> TRN-cognitive (FIXED: uses G_RELAY_TRN_C)
        self.il_trn = Synapse(
            n_pre=N_IL,
            n_post=N_TRN_C,
            g_max=G_RELAY_TRN_C,
            tau=TAU_AMPA,
            e_rev=E_AMPA,
            prob=P_RELAY_TRN,
            name="IL->TRN_C"
        )

        """
         GROUP 4: TRN -> Relay (feedback and lateral inhibition)
         TRN suppresses relay neurons it receives input from
         (feedback) and neighboring relay neurons (lateral)
        """

        # Pathway 15: TRN-motor -> VL (feedback inhibition)
        self.trn_vl_feed = Synapse(
            n_pre=N_TRN_M,
            n_post=N_VL,
            g_max=G_TRN_RELAY,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_FEED,
            name="TRN_M->VL_feedback"
        )

        # Pathway 15b: TRN-motor -> VL (lateral inhibition, cross-channel)
        self.trn_vl_lat = Synapse(
            n_pre=N_TRN_M,
            n_post=N_VL,
            g_max=G_TRN_LATERAL,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_LAT,
            name="TRN_M->VL_lateral"
        )

        # Pathway 16: TRN-motor -> VA (feedback inhibition)
        self.trn_va_feed = Synapse(
            n_pre=N_TRN_M,
            n_post=N_VA,
            g_max=G_TRN_RELAY,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_FEED,
            name="TRN_M->VA_feedback"
        )

        # Pathway 16b: TRN-motor -> VA (lateral inhibition)
        self.trn_va_lat = Synapse(
            n_pre=N_TRN_M,
            n_post=N_VA,
            g_max=G_TRN_LATERAL,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_LAT,
            name="TRN_M->VA_lateral"
        )

        # Pathway 17: TRN-cognitive -> MD (feedback inhibition)
        self.trn_md_feed = Synapse(
            n_pre=N_TRN_C,
            n_post=N_MD,
            g_max=G_TRN_RELAY,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_FEED,
            name="TRN_C->MD_feedback"
        )

        # Pathway 17b: TRN-cognitive -> MD (lateral inhibition)
        self.trn_md_lat = Synapse(
            n_pre=N_TRN_C,
            n_post=N_MD,
            g_max=G_TRN_LATERAL,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_LAT,
            name="TRN_C->MD_lateral"
        )

        # Pathway 18: TRN-cognitive -> IL (feedback inhibition)
        self.trn_il_feed = Synapse(
            n_pre=N_TRN_C,
            n_post=N_IL,
            g_max=G_TRN_RELAY,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_FEED,
            name="TRN_C->IL_feedback"
        )

        # Pathway 18b: TRN-cognitive -> IL (lateral inhibition)
        self.trn_il_lat = Synapse(
            n_pre=N_TRN_C,
            n_post=N_IL,
            g_max=G_TRN_LATERAL,
            tau=TAU_GABA,
            e_rev=E_GABA,
            prob=P_TRN_RELAY_LAT,
            name="TRN_C->IL_lateral"
        )

    def reset(self):
        """Reset all synaptic conductances to zero."""
        # BG inputs
        for syn_list in [self.gpi_vl, self.gpi_va]:
            for syn in syn_list:
                syn.reset()
        self.snr_md.reset()
        self.snr_il.reset()

        # Cortical inputs
        for syn in [self.ctx_vl_ampa, self.ctx_vl_nmda,
                    self.ctx_va_ampa, self.ctx_va_nmda,
                    self.ctx_md_ampa, self.ctx_md_nmda,
                    self.ctx_il_ampa, self.ctx_il_nmda,
                    self.ctx_trn_m, self.ctx_trn_c]:
            syn.reset()

        # Relay -> TRN
        for syn in [self.vl_trn, self.va_trn, self.md_trn, self.il_trn]:
            syn.reset()

        # TRN -> Relay
        for syn in [self.trn_vl_feed, self.trn_vl_lat,
                    self.trn_va_feed, self.trn_va_lat,
                    self.trn_md_feed, self.trn_md_lat,
                    self.trn_il_feed, self.trn_il_lat]:
            syn.reset()

    def compute_bg_input(self, net, gpi_rates, snr_rates):
        """
        Compute synaptic current from BG tonic inhibition.

        net: ThalamusNetwork
        gpi_rates: array of GPi firing rates per channel (Hz)
        snr_rates: array [snr_md_rate, snr_il_rate] (Hz)
        Returns: dict of population_name -> current array
        """
        currents = {
            'vl': np.zeros(N_VL),
            'va': np.zeros(N_VA),
            'md': np.zeros(N_MD),
            'il': np.zeros(N_IL)
        }

        # GPi -> VL (per channel)
        for ch in range(N_CHANNELS):
            # Generate Poisson spikes from GPi rate
            gpi_spike = np.random.rand(N_BG_PRE) < (gpi_rates[ch] * DT / 1000.0)
            start = ch * N_VL_PER_CH
            end = start + N_VL_PER_CH
            I = self.gpi_vl[ch].update(gpi_spike, net.vl.v[start:end])
            currents['vl'][start:end] += I

        # GPi -> VA (per channel)
        for ch in range(N_CHANNELS):
            gpi_spike = np.random.rand(N_BG_PRE) < (gpi_rates[ch] * DT / 1000.0)
            start = ch * N_VA_PER_CH
            end = start + N_VA_PER_CH
            I = self.gpi_va[ch].update(gpi_spike, net.va.v[start:end])
            currents['va'][start:end] += I

        # SNr -> MD
        snr_md_spike = np.random.rand(N_BG_PRE) < (snr_rates[0] * DT / 1000.0)
        currents['md'] += self.snr_md.update(snr_md_spike, net.md.v)

        # SNr -> IL
        snr_il_spike = np.random.rand(N_BG_PRE) < (snr_rates[1] * DT / 1000.0)
        currents['il'] += self.snr_il.update(snr_il_spike, net.il.v)

        return currents

    def compute_ctx_input(self, net, ctx_rate):
        """
        Compute synaptic current from cortical feedback.
        Simulated as Poisson input at ctx_rate Hz.

        net: ThalamusNetwork
        ctx_rate: cortical feedback firing rate (Hz)
        Returns: dict of population_name -> current array
        """
        currents = {
            'vl': np.zeros(N_VL),
            'va': np.zeros(N_VA),
            'md': np.zeros(N_MD),
            'il': np.zeros(N_IL),
            'trn_m': np.zeros(N_TRN_M),
            'trn_c': np.zeros(N_TRN_C)
        }

        # Generate single Poisson spike for this timestep
        ctx_spike = np.random.rand(1) < (ctx_rate * DT / 1000.0)

        # L6 -> relay populations (AMPA + NMDA)
        currents['vl'] += self.ctx_vl_ampa.update(ctx_spike, net.vl.v)
        currents['vl'] += self.ctx_vl_nmda.update(ctx_spike, net.vl.v)

        currents['va'] += self.ctx_va_ampa.update(ctx_spike, net.va.v)
        currents['va'] += self.ctx_va_nmda.update(ctx_spike, net.va.v)

        # L5 -> higher-order nuclei (AMPA + NMDA, stronger driver)
        currents['md'] += self.ctx_md_ampa.update(ctx_spike, net.md.v)
        currents['md'] += self.ctx_md_nmda.update(ctx_spike, net.md.v)

        currents['il'] += self.ctx_il_ampa.update(ctx_spike, net.il.v)
        currents['il'] += self.ctx_il_nmda.update(ctx_spike, net.il.v)

        # L6 -> TRN (AMPA only, drives attentional gating)
        currents['trn_m'] += self.ctx_trn_m.update(ctx_spike, net.trn_m.v)
        currents['trn_c'] += self.ctx_trn_c.update(ctx_spike, net.trn_c.v)

        return currents

    def compute_internal(self, net):
        """
        Compute all internal thalamic currents.
        Relay -> TRN (feedforward) and TRN -> Relay (feedback + lateral).
        These use actual spike arrays from the populations.

        net: ThalamusNetwork
        Returns: dict of population_name -> current array
        """
        currents = {
            'vl': np.zeros(N_VL),
            'va': np.zeros(N_VA),
            'md': np.zeros(N_MD),
            'il': np.zeros(N_IL),
            'trn_m': np.zeros(N_TRN_M),
            'trn_c': np.zeros(N_TRN_C)
        }

        # Relay -> TRN (feedforward collaterals)
        currents['trn_m'] += self.vl_trn.update(net.vl.spikes, net.trn_m.v)
        currents['trn_m'] += self.va_trn.update(net.va.spikes, net.trn_m.v)
        currents['trn_c'] += self.md_trn.update(net.md.spikes, net.trn_c.v)
        currents['trn_c'] += self.il_trn.update(net.il.spikes, net.trn_c.v)

        # TRN -> Relay (feedback inhibition)
        currents['vl'] += self.trn_vl_feed.update(net.trn_m.spikes, net.vl.v)
        currents['va'] += self.trn_va_feed.update(net.trn_m.spikes, net.va.v)
        currents['md'] += self.trn_md_feed.update(net.trn_c.spikes, net.md.v)
        currents['il'] += self.trn_il_feed.update(net.trn_c.spikes, net.il.v)

        # TRN -> Relay (lateral inhibition, cross-channel)
        currents['vl'] += self.trn_vl_lat.update(net.trn_m.spikes, net.vl.v)
        currents['va'] += self.trn_va_lat.update(net.trn_m.spikes, net.va.v)
        currents['md'] += self.trn_md_lat.update(net.trn_c.spikes, net.md.v)
        currents['il'] += self.trn_il_lat.update(net.trn_c.spikes, net.il.v)

        return currents

    def compute_total_currents(self, net, gpi_rates, snr_rates, ctx_rate):
        """
        Compute all synaptic currents for one timestep.
        Sums BG input + cortical input + internal connections.

        net: ThalamusNetwork
        gpi_rates: array of GPi firing rates per channel (Hz)
        snr_rates: array [snr_md_rate, snr_il_rate] (Hz)
        ctx_rate: cortical background firing rate (Hz)
        Returns: dict of population_name -> total current array
        """
        # Get currents from each source
        bg = self.compute_bg_input(net, gpi_rates, snr_rates)
        ctx = self.compute_ctx_input(net, ctx_rate)
        internal = self.compute_internal(net)

        # Sum all currents per population
        total = {}
        for name in ['vl', 'va', 'md', 'il', 'trn_m', 'trn_c']:
            total[name] = np.zeros(self._get_n(name))
            if name in bg:
                total[name] += bg[name]
            if name in ctx:
                total[name] += ctx[name]
            if name in internal:
                total[name] += internal[name]

        return total

    def _get_n(self, name):
        """Get population size by name."""
        sizes = {
            'vl': N_VL, 'va': N_VA,
            'md': N_MD, 'il': N_IL,
            'trn_m': N_TRN_M, 'trn_c': N_TRN_C
        }
        return sizes[name]

    def summary(self):
        """Print connection summary."""
        print("=" * 50)
        print("BioMind-Thalamus Connections")
        print("=" * 50)

        groups = [
            ("BG -> Thalamus (GABA)", [
                ("GPi->VL", self.gpi_vl[0], N_CHANNELS),
                ("GPi->VA", self.gpi_va[0], N_CHANNELS),
                ("SNr->MD", self.snr_md, 1),
                ("SNr->IL", self.snr_il, 1)
            ]),
            ("Cortex -> Thalamus (Glu)", [
                ("Ctx->VL_AMPA", self.ctx_vl_ampa, 1),
                ("Ctx->VL_NMDA", self.ctx_vl_nmda, 1),
                ("Ctx->VA_AMPA", self.ctx_va_ampa, 1),
                ("Ctx->MD_AMPA", self.ctx_md_ampa, 1),
                ("Ctx->IL_AMPA", self.ctx_il_ampa, 1),
                ("Ctx->TRN_M", self.ctx_trn_m, 1),
                ("Ctx->TRN_C", self.ctx_trn_c, 1)
            ]),
            ("Relay -> TRN (Glu)", [
                ("VL->TRN_M", self.vl_trn, 1),
                ("VA->TRN_M", self.va_trn, 1),
                ("MD->TRN_C", self.md_trn, 1),
                ("IL->TRN_C", self.il_trn, 1)
            ]),
            ("TRN -> Relay (GABA)", [
                ("TRN_M->VL_fb", self.trn_vl_feed, 1),
                ("TRN_M->VL_lat", self.trn_vl_lat, 1),
                ("TRN_M->VA_fb", self.trn_va_feed, 1),
                ("TRN_C->MD_fb", self.trn_md_feed, 1),
                ("TRN_C->IL_fb", self.trn_il_feed, 1)
            ])
        ]

        for group_name, pathways in groups:
            print(f"\n  {group_name}:")
            for name, syn, n_ch in pathways:
                n_conn = np.sum(syn.W)
                total_possible = syn.n_pre * syn.n_post
                density = n_conn / total_possible * 100 if total_possible > 0 else 0
                ch_str = f" x{n_ch}ch" if n_ch > 1 else ""
                print(f"    {name:20s}: g={syn.g_max:.4f}  "
                      f"tau={syn.tau:.0f}ms  "
                      f"conn={n_conn:.0f}/{total_possible}{ch_str}  "
                      f"({density:.0f}%)")
        print("x" * 50)