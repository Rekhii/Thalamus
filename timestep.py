import numpy as np
from params import *


def step(net, conn, t, gpi_rates, snr_rates, ctx_rate, neuromod=NEUROMOD_WAKE):
    """
    Advance the entire thalamic network by one timestep (DT ms).

    net: ThalamusNetwork object
    conn: ThalamusConnections object
    t: current simulation time (ms)
    gpi_rates: array of GPi firing rates per channel (Hz), shape (N_CHANNELS,)
    snr_rates: array [snr_md_rate, snr_il_rate] (Hz)
    ctx_rate: cortical background firing rate (Hz)
    neuromod: float 0-1, brainstem arousal level
    """

    # Step 1: Compute all synaptic currents
    currents = conn.compute_total_currents(net, gpi_rates, snr_rates, ctx_rate)

    # Step 2: Update relay populations (with T-channel)
    vl_spikes = net.vl.update(currents['vl'], neuromod)
    va_spikes = net.va.update(currents['va'], neuromod)
    md_spikes = net.md.update(currents['md'], neuromod)
    il_spikes = net.il.update(currents['il'], neuromod)

    # Step 3: Update TRN populations (standard LIF)
    trn_m_spikes = net.trn_m.update(currents['trn_m'])
    trn_c_spikes = net.trn_c.update(currents['trn_c'])

    # Step 4: Record spikes
    net.record_spikes('vl', vl_spikes, t)
    net.record_spikes('va', va_spikes, t)
    net.record_spikes('md', md_spikes, t)
    net.record_spikes('il', il_spikes, t)
    net.record_spikes('trn_m', trn_m_spikes, t)
    net.record_spikes('trn_c', trn_c_spikes, t)