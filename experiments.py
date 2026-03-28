import numpy as np
import matplotlib.pyplot as plt
from params import *
from populations import ThalamusNetwork
from connections import ThalamusConnections
from timestep import step


def experiment_1_baseline(duration_ms=2000):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline Firing Rates")
    print("=" * 60)

    net = ThalamusNetwork()
    conn = ThalamusConnections()
    gpi_rates = np.full(N_CHANNELS, GPi_TONIC_RATE)
    snr_rates = np.array([SNr_TONIC_RATE, SNr_TONIC_RATE])
    ctx_rate = CTX_BACKGROUND_RATE

    n_steps = int(duration_ms / DT)
    print(f"Running {duration_ms} ms ({n_steps} timesteps)...")

    for i in range(n_steps):
        t = i * DT
        step(net, conn, t, gpi_rates, snr_rates, ctx_rate, NEUROMOD_WAKE)

    rates = net.get_firing_rates()
    print("\nFiring rates:")
    for name, rate in rates.items():
        print(f"  {name:6s}: {rate:.1f} Hz")

    checks = [
        ('vl', 2, 5, 'VL suppressed by GPi'),
        ('va', 2, 5, 'VA suppressed by GPi'),
        ('md', 5, 10, 'MD suppressed by SNr'),
        ('il', 8, 15, 'IL baseline'),
        ('trn_m', 15, 25, 'TRN-M spontaneous'),
        ('trn_c', 15, 25, 'TRN-C spontaneous')
    ]
    all_pass = True
    print("\nValidation:")
    for name, lo, hi, desc in checks:
        rate = rates[name]
        passed = lo <= rate <= hi
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}: {rate:.1f} Hz (target: {lo}-{hi})")
        if not passed:
            all_pass = False

    vl_mode = net.vl.get_mode_fractions()
    va_mode = net.va.get_mode_fractions()
    print(f"\n  VL burst-ready: {vl_mode['burst_ready']:.0%}")
    print(f"  VA burst-ready: {va_mode['burst_ready']:.0%}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    names = list(rates.keys())
    values = [rates[n] for n in names]
    colors = ['#2196F3', '#42A5F5', '#7E57C2', '#AB47BC', '#EF5350', '#E53935']
    axes[0].bar(names, values, color=colors)
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title('Experiment 1: Baseline Firing Rates')

    h_data = [net.vl.h, net.va.h, net.md.h, net.il.h]
    h_labels = ['VL', 'VA', 'MD', 'IL']
    axes[1].violinplot(h_data, positions=range(4), showmeans=True)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(h_labels)
    axes[1].set_ylabel('h (T-channel de-inactivation)')
    axes[1].set_title('T-channel State: 0=tonic, 1=burst-ready')
    axes[1].set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('exp1_baseline.png', dpi=150)
    plt.show()
    net.summary()
    return all_pass


def experiment_2_disinhibition(duration_ms=3000):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Disinhibition Response (Burst-Tonic)")
    print("=" * 60)

    net = ThalamusNetwork()
    conn = ThalamusConnections()
    snr_rates = np.array([SNr_TONIC_RATE, SNr_TONIC_RATE])
    ctx_rate = CTX_BACKGROUND_RATE

    n_steps = int(duration_ms / DT)
    print(f"Running {duration_ms} ms ({n_steps} timesteps)...")

    t_record = []
    h_ch0_record = []
    h_ch1_record = []
    vl_v_sample = []
    spike_times_ch0 = []
    sample_neuron = 5

    for i in range(n_steps):
        t = i * DT
        gpi_rates = np.full(N_CHANNELS, GPi_TONIC_RATE)
        if 1000.0 <= t < 1500.0:
            gpi_rates[0] = 0.0
        step(net, conn, t, gpi_rates, snr_rates, ctx_rate, NEUROMOD_WAKE)

        if i % 10 == 0:
            t_record.append(t)
            h_ch0_record.append(np.mean(net.vl.h[:N_VL_PER_CH]))
            h_ch1_record.append(np.mean(net.vl.h[N_VL_PER_CH:2*N_VL_PER_CH]))
            vl_v_sample.append(net.vl.v[sample_neuron])

        if np.any(net.vl.spikes[:N_VL_PER_CH]):
            spike_times_ch0.append(t)

    onset_spikes = [s for s in spike_times_ch0 if 1000.0 <= s <= 1020.0]
    n_onset_spikes = len(onset_spikes)
    print(f"\nSpikes in first 20 ms after disinhibition: {n_onset_spikes}")

    ch_rates = net.get_channel_firing_rates('vl', N_VL_PER_CH)
    print(f"\nOverall VL channel rates:")
    for ch in range(N_CHANNELS):
        print(f"  Channel {ch}: {ch_rates[ch]:.1f} Hz")

    print("\nValidation:")
    burst_ok = n_onset_spikes >= 3
    print(f"  [{'PASS' if burst_ok else 'FAIL'}] Burst at onset: "
          f"{n_onset_spikes} spikes (target: >=3)")
    ch0_elevated = ch_rates[0] > ch_rates[1] + 1
    print(f"  [{'PASS' if ch0_elevated else 'FAIL'}] Ch0 elevated vs Ch1: "
          f"{ch_rates[0]:.1f} vs {ch_rates[1]:.1f} Hz")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(t_record, h_ch0_record, 'b-', label='Channel 0 (disinhibited)', linewidth=2)
    axes[0].plot(t_record, h_ch1_record, 'r--', label='Channel 1 (control)', linewidth=1.5)
    axes[0].axvspan(1000, 1500, alpha=0.15, color='green', label='GPi ch0 OFF')
    axes[0].set_ylabel('Mean h')
    axes[0].set_title('Experiment 2: T-channel De-inactivation')
    axes[0].legend()
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].plot(t_record, vl_v_sample, 'k-', linewidth=0.5)
    axes[1].axvspan(1000, 1500, alpha=0.15, color='green')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_title(f'VL Neuron {sample_neuron} (Channel 0) Membrane Potential')
    axes[1].axhline(V_THRESH, color='r', linestyle=':', alpha=0.5, label='Threshold')
    axes[1].legend()

    ch0_spike_hist = net.spike_history['vl']
    for idx, times in ch0_spike_hist.items():
        if idx < N_VL_PER_CH:
            axes[2].scatter(times, [idx]*len(times), s=0.5, c='blue', alpha=0.5)
        elif idx < 2*N_VL_PER_CH:
            axes[2].scatter(times, [idx]*len(times), s=0.5, c='red', alpha=0.5)
    axes[2].axvspan(1000, 1500, alpha=0.15, color='green')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Neuron Index')
    axes[2].set_title('Spike Raster: Blue=Ch0, Red=Ch1')
    plt.tight_layout()
    plt.savefig('exp2_disinhibition.png', dpi=150)
    plt.show()
    return burst_ok and ch0_elevated


def experiment_3_trn_lateral(duration_ms=3000):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: TRN Lateral Inhibition")
    print("=" * 60)

    net = ThalamusNetwork()
    conn = ThalamusConnections()
    snr_rates = np.array([SNr_TONIC_RATE, SNr_TONIC_RATE])
    ctx_rate = CTX_BACKGROUND_RATE

    n_steps = int(duration_ms / DT)
    print(f"Running {duration_ms} ms ({n_steps} timesteps)...")

    phase1_counts = np.zeros((N_CHANNELS,))
    phase2_counts = np.zeros((N_CHANNELS,))
    trn_phase1_count = 0
    trn_phase2_count = 0

    for i in range(n_steps):
        t = i * DT
        gpi_rates = np.full(N_CHANNELS, GPi_TONIC_RATE)
        if t >= 1000.0:
            gpi_rates[0] = 0.0
        step(net, conn, t, gpi_rates, snr_rates, ctx_rate, NEUROMOD_WAKE)

        for ch in range(N_CHANNELS):
            s, e = ch*N_VL_PER_CH, (ch+1)*N_VL_PER_CH
            ch_spikes = np.sum(net.vl.spikes[s:e])
            if t < 1000.0:
                phase1_counts[ch] += ch_spikes
            else:
                phase2_counts[ch] += ch_spikes

        trn_spk = np.sum(net.trn_m.spikes)
        if t < 1000.0:
            trn_phase1_count += trn_spk
        else:
            trn_phase2_count += trn_spk

    phase1_s = 1.0
    phase2_s = (duration_ms - 1000.0) / 1000.0
    phase1_rates = phase1_counts / (N_VL_PER_CH * phase1_s)
    phase2_rates = phase2_counts / (N_VL_PER_CH * phase2_s)
    trn_phase1_rate = trn_phase1_count / (N_TRN_M * phase1_s)
    trn_phase2_rate = trn_phase2_count / (N_TRN_M * phase2_s)

    print("\nVL Channel rates by phase:")
    for ch in range(N_CHANNELS):
        print(f"  Channel {ch}: Phase1={phase1_rates[ch]:.1f} Hz  "
              f"Phase2={phase2_rates[ch]:.1f} Hz  "
              f"Change={phase2_rates[ch] - phase1_rates[ch]:+.1f}")
    print(f"\nTRN-M rate: Phase1={trn_phase1_rate:.1f} Hz  Phase2={trn_phase2_rate:.1f} Hz")

    print("\nValidation:")
    ch0_up = phase2_rates[0] > phase1_rates[0] + 10
    print(f"  [{'PASS' if ch0_up else 'FAIL'}] Ch0 elevated: "
          f"{phase1_rates[0]:.1f} -> {phase2_rates[0]:.1f} Hz")
    ch1_down = phase2_rates[1] < phase1_rates[1]
    ch2_down = phase2_rates[2] < phase1_rates[2]
    lat_ok = ch1_down and ch2_down
    print(f"  [{'PASS' if lat_ok else 'FAIL'}] Ch1,2 suppressed: "
          f"Ch1: {phase1_rates[1]:.1f} -> {phase2_rates[1]:.1f}  "
          f"Ch2: {phase1_rates[2]:.1f} -> {phase2_rates[2]:.1f}")
    trn_up = trn_phase2_rate > trn_phase1_rate
    print(f"  [{'PASS' if trn_up else 'FAIL'}] TRN-M increased: "
          f"{trn_phase1_rate:.1f} -> {trn_phase2_rate:.1f} Hz")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    x = np.arange(N_CHANNELS)
    width = 0.35
    axes[0].bar(x - width/2, phase1_rates, width, label='Phase 1 (baseline)', color='#90CAF9')
    axes[0].bar(x + width/2, phase2_rates, width, label='Phase 2 (ch0 open)', color='#1565C0')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'Channel {i}' for i in range(N_CHANNELS)])
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title('Experiment 3: Lateral Inhibition Effect')
    axes[0].legend()
    axes[1].bar(['Phase 1', 'Phase 2'], [trn_phase1_rate, trn_phase2_rate], color=['#EF9A9A', '#C62828'])
    axes[1].set_ylabel('Firing Rate (Hz)')
    axes[1].set_title('TRN-Motor Firing Rate')
    plt.tight_layout()
    plt.savefig('exp3_lateral_inhibition.png', dpi=150)
    plt.show()
    return ch0_up and lat_ok and trn_up


def experiment_4_oscillations(duration_ms=5000):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Oscillation Generation (Sleep Spindles)")
    print("=" * 60)

    net = ThalamusNetwork()
    conn = ThalamusConnections()

    gpi_rates = np.full(N_CHANNELS, GPi_TONIC_RATE * 0.5)
    snr_rates = np.array([SNr_TONIC_RATE * 0.5, SNr_TONIC_RATE * 0.5])
    ctx_rate = CTX_BACKGROUND_RATE * 0.3

    # Sleep-specific parameters (biologically justified):
    # During sleep, cortical slow oscillations produce larger input fluctuations
    # and TRN GABA release is enhanced (extrasynaptic GABA-A activation)
    SLEEP_NOISE = 10.0
    TRN_SLEEP_BOOST = 25

    n_steps = int(duration_ms / DT)
    print(f"Running {duration_ms} ms ({n_steps} timesteps)...")

    t_record = []
    h_mean_record = []
    vl_rate_bins = []
    trn_rate_bins = []

    bin_size_ms = 50.0  # 50 ms bins for spindle-frequency resolution
    bin_steps = int(bin_size_ms / DT)
    vl_bin_count = 0
    trn_bin_count = 0

    for i in range(n_steps):
        t = i * DT

        # Transition to sleep
        if t < 500.0:
            neuromod = NEUROMOD_WAKE - (NEUROMOD_WAKE - NEUROMOD_SLEEP) * (t / 500.0)
            frac = t / 500.0
            for pop in [net.vl, net.va, net.md, net.il]:
                pop.noise_std = NOISE_RELAY + (SLEEP_NOISE - NOISE_RELAY) * frac
        else:
            neuromod = NEUROMOD_SLEEP
            for pop in [net.vl, net.va, net.md, net.il]:
                pop.noise_std = SLEEP_NOISE

        # Boost TRN->relay GABA during sleep
        if t >= 500.0:
            for syn in [conn.trn_vl_feed, conn.trn_va_feed, conn.trn_md_feed, conn.trn_il_feed]:
                syn.g_max = G_TRN_RELAY * TRN_SLEEP_BOOST
            for syn in [conn.trn_vl_lat, conn.trn_va_lat, conn.trn_md_lat, conn.trn_il_lat]:
                syn.g_max = G_TRN_LATERAL * TRN_SLEEP_BOOST

        step(net, conn, t, gpi_rates, snr_rates, ctx_rate, neuromod)

        vl_bin_count += np.sum(net.vl.spikes)
        trn_bin_count += np.sum(net.trn_m.spikes)

        if (i + 1) % bin_steps == 0:
            t_record.append(t)
            h_mean_record.append(net.vl.get_h_mean())
            vl_rate_bins.append(vl_bin_count / (N_VL * bin_size_ms / 1000.0))
            trn_rate_bins.append(trn_bin_count / (N_TRN_M * bin_size_ms / 1000.0))
            vl_bin_count = 0
            trn_bin_count = 0

    # Individual neuron h spread (neurons at different burst cycle phases)
    h_spread = np.max(net.vl.h) - np.min(net.vl.h)

    # Spectral analysis on VL rate bins after sleep onset
    sleep_start_idx = int(500.0 / bin_size_ms)
    vl_sleep = np.array(vl_rate_bins[sleep_start_idx:])
    peak_freq = 0
    freqs = np.array([0])
    power = np.array([0])
    valid = np.array([False])

    if len(vl_sleep) > 10:
        vl_centered = vl_sleep - np.mean(vl_sleep)
        freqs = np.fft.rfftfreq(len(vl_centered), d=bin_size_ms / 1000.0)
        power = np.abs(np.fft.rfft(vl_centered)) ** 2
        valid = freqs > 2.0
        if np.any(valid):
            peak_idx = np.argmax(power[valid])
            peak_freq = freqs[valid][peak_idx]

    print(f"\nPeak oscillation frequency: {peak_freq:.1f} Hz")

    h_arr = np.array(h_mean_record[sleep_start_idx:])
    h_mean_range = np.max(h_arr) - np.min(h_arr) if len(h_arr) > 0 else 0
    print(f"h population mean range: {h_mean_range:.3f}")
    print(f"h individual spread (snapshot): {h_spread:.3f}")

    print("\nValidation:")
    freq_ok = 4.0 <= peak_freq <= 14.0
    print(f"  [{'PASS' if freq_ok else 'FAIL'}] Peak frequency in sleep oscillation range: "
          f"{peak_freq:.1f} Hz (target: 4-14 Hz)")
    h_oscillates = h_spread > 0.3
    print(f"  [{'PASS' if h_oscillates else 'FAIL'}] h individual spread: "
          f"{h_spread:.2f} (target: >0.3)")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    axes[0].plot(t_record, h_mean_record, 'b-', linewidth=1)
    axes[0].axvspan(0, 500, alpha=0.1, color='yellow', label='Wake->Sleep transition')
    axes[0].set_ylabel('Mean h')
    axes[0].set_title('Experiment 4: T-channel De-inactivation During Sleep')
    axes[0].legend()

    axes[1].plot(t_record, vl_rate_bins, 'b-', linewidth=0.8, label='VL')
    axes[1].plot(t_record, trn_rate_bins, 'r-', linewidth=0.8, alpha=0.7, label='TRN-M')
    axes[1].set_ylabel('Rate (Hz)')
    axes[1].set_title('Population Firing Rates (50 ms bins)')
    axes[1].legend()

    zoom_start = int(2000.0 / bin_size_ms)
    zoom_end = int(3000.0 / bin_size_ms)
    if zoom_end <= len(vl_rate_bins):
        t_zoom = t_record[zoom_start:zoom_end]
        vl_zoom = vl_rate_bins[zoom_start:zoom_end]
        trn_zoom = trn_rate_bins[zoom_start:zoom_end]
        axes[2].plot(t_zoom, vl_zoom, 'b-', linewidth=1.5, label='VL')
        axes[2].plot(t_zoom, trn_zoom, 'r-', linewidth=1.5, alpha=0.7, label='TRN-M')
        axes[2].set_ylabel('Rate (Hz)')
        axes[2].set_title('Zoomed: 2000-3000 ms (Spindle Window)')
        axes[2].legend()

    if len(freqs) > 1 and np.any(valid):
        axes[3].plot(freqs[valid], power[valid], 'k-', linewidth=1)
        axes[3].axvspan(4, 14, alpha=0.2, color='green', label='Sleep oscillation range')
        axes[3].set_xlabel('Frequency (Hz)')
        axes[3].set_ylabel('Power')
        axes[3].set_title(f'Power Spectrum (peak: {peak_freq:.1f} Hz)')
        axes[3].set_xlim(0, 30)
        axes[3].legend()

    plt.tight_layout()
    plt.savefig('exp4_oscillations.png', dpi=150)
    plt.show()
    return freq_ok and h_oscillates


def experiment_5_bg_integration(duration_ms=5000):
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: BG-Thalamus Integration")
    print("=" * 60)

    net = ThalamusNetwork()
    conn = ThalamusConnections()
    snr_rates = np.array([SNr_TONIC_RATE, SNr_TONIC_RATE])
    ctx_rate = CTX_BACKGROUND_RATE

    n_steps = int(duration_ms / DT)
    print(f"Running {duration_ms} ms ({n_steps} timesteps)...")

    t_record = []
    h_records = {ch: [] for ch in range(N_CHANNELS)}
    rate_records = {ch: [] for ch in range(N_CHANNELS)}
    bin_size_ms = 5.0
    bin_steps = int(bin_size_ms / DT)
    ch_bin_counts = np.zeros(N_CHANNELS)

    for i in range(n_steps):
        t = i * DT
        gpi_rates = np.full(N_CHANNELS, GPi_TONIC_RATE)
        if 1500.0 <= t < 3000.0:
            gpi_rates[0] = 10.0
            gpi_rates[1] = GPi_TONIC_RATE + 15.0
            gpi_rates[2] = GPi_TONIC_RATE + 15.0
        elif 3000.0 <= t < 4000.0:
            gpi_rates[1] = 10.0
            gpi_rates[0] = GPi_TONIC_RATE + 15.0
            gpi_rates[2] = GPi_TONIC_RATE + 15.0

        step(net, conn, t, gpi_rates, snr_rates, ctx_rate, NEUROMOD_WAKE)

        for ch in range(N_CHANNELS):
            s, e = ch*N_VL_PER_CH, (ch+1)*N_VL_PER_CH
            ch_bin_counts[ch] += np.sum(net.vl.spikes[s:e])

        if (i + 1) % bin_steps == 0:
            t_record.append(t)
            for ch in range(N_CHANNELS):
                s, e = ch*N_VL_PER_CH, (ch+1)*N_VL_PER_CH
                h_records[ch].append(np.mean(net.vl.h[s:e]))
                rate_records[ch].append(ch_bin_counts[ch] / (N_VL_PER_CH * bin_size_ms / 1000.0))
            ch_bin_counts[:] = 0

    print("\nPhase analysis:")
    phases = [
        ('Phase 1 (baseline)', 0, 1500),
        ('Phase 2 (action 0)', 1500, 3000),
        ('Phase 3 (action 1)', 3000, 4000),
        ('Phase 4 (no action)', 4000, 5000)
    ]
    for phase_name, t_start, t_end in phases:
        idx_s = int(t_start / bin_size_ms)
        idx_e = int(t_end / bin_size_ms)
        print(f"\n  {phase_name}:")
        for ch in range(N_CHANNELS):
            pr = rate_records[ch][idx_s:idx_e]
            ph = h_records[ch][idx_s:idx_e]
            print(f"    Ch{ch}: rate={np.mean(pr):.1f} Hz  h={np.mean(ph):.2f}")

    print("\nValidation:")
    p2_s, p2_e = int(1500/bin_size_ms), int(3000/bin_size_ms)
    p2_ch0 = np.mean(rate_records[0][p2_s:p2_e])
    p2_ch1 = np.mean(rate_records[1][p2_s:p2_e])
    ch0_wins_p2 = p2_ch0 > p2_ch1 + 5
    print(f"  [{'PASS' if ch0_wins_p2 else 'FAIL'}] Phase 2 ch0 dominant: ch0={p2_ch0:.1f} ch1={p2_ch1:.1f}")

    p3_s, p3_e = int(3000/bin_size_ms), int(4000/bin_size_ms)
    p3_ch0 = np.mean(rate_records[0][p3_s:p3_e])
    p3_ch1 = np.mean(rate_records[1][p3_s:p3_e])
    ch1_wins_p3 = p3_ch1 > p3_ch0 + 5
    print(f"  [{'PASS' if ch1_wins_p3 else 'FAIL'}] Phase 3 ch1 dominant: ch1={p3_ch1:.1f} ch0={p3_ch0:.1f}")

    p4_s, p4_e = int(4000/bin_size_ms), int(5000/bin_size_ms)
    p4_rates = [np.mean(rate_records[ch][p4_s:p4_e]) for ch in range(N_CHANNELS)]
    all_low = all(r < 8 for r in p4_rates)
    print(f"  [{'PASS' if all_low else 'FAIL'}] Phase 4 all suppressed: {[f'{r:.1f}' for r in p4_rates]}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    colors = ['#1565C0', '#E53935', '#2E7D32']
    for ch in range(N_CHANNELS):
        axes[0].plot(t_record, rate_records[ch], color=colors[ch], linewidth=1, label=f'Channel {ch}', alpha=0.8)
    axes[0].axvspan(1500, 3000, alpha=0.1, color='blue', label='Action 0')
    axes[0].axvspan(3000, 4000, alpha=0.1, color='red', label='Action 1')
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title('Experiment 5: BG-Thalamus Integration - VL Channel Rates')
    axes[0].legend()

    for ch in range(N_CHANNELS):
        axes[1].plot(t_record, h_records[ch], color=colors[ch], linewidth=1.5, label=f'Channel {ch}')
    axes[1].axvspan(1500, 3000, alpha=0.1, color='blue')
    axes[1].axvspan(3000, 4000, alpha=0.1, color='red')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Mean h')
    axes[1].set_title('T-channel State Per Channel')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('exp5_integration.png', dpi=150)
    plt.show()
    return ch0_wins_p2 and ch1_wins_p3 and all_low