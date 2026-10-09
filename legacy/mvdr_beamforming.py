import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import inv

from charts_utils.core.io import load_gpu_corr_frame, gpu_corr_to_visibilities
from charts_utils.core.gpu_frames import resolve_gpu_frame_timestamps
from charts_utils.cal.algorithms import utvec2mat, apply_gains, form_beam
from charts_utils.cal.coordinates import CHARTSArray, fringestop_time, _reference_delays
from charts_utils.core.kotekan import (
    KOTEKAN_GPU_N_FREQ, KOTEKAN_GPU_N_RECEIVERS, KOTEKAN_GPU_N_POLARIZATIONS,
    KOTEKAN_GPU_FRAME_CADENCE_S, KOTEKAN_GPU_FREQ_START_MHZ, KOTEKAN_GPU_FREQ_STOP_MHZ
)

def compute_mvdr_weights(R_in: np.ndarray, a_target: np.ndarray, diag_loading: float = 1e-2) -> np.ndarray:
    """
    Computes the Minimum Variance Distortionless Response (MVDR) weights.
    Applies Diagonal Loading to ensure numerical stability during inversion.
    """
    n_ant = R_in.shape[0]
    mean_power = np.abs(np.trace(R_in)) / n_ant
    R_reg = R_in + (diag_loading * mean_power) * np.eye(n_ant)
    
    R_inv = inv(R_reg)
    num = R_inv @ a_target
    den = a_target.conj().T @ num
    return num / den

def get_steering_vector(array: CHARTSArray, timestamp_unix: float, freq_hz: float, target: str = "sun") -> np.ndarray:
    """Computes the frequency-dependent steering vector 'a' towards the specified spatial coordinate."""
    delays_tau = _reference_delays(timestamp_unix, array, target)
    return np.exp(-1j * 2.0 * np.pi * freq_hz * delays_tau).reshape(-1, 1)

if __name__ == "__main__":
    data_dir = "../corr_24h_2608"
    gains_file = "static_gains_24h.npy"
    
    if not os.path.exists(gains_file):
        raise FileNotFoundError(f"Missing calibration file: {gains_file}. Run Pass 1 first.")
        
    all_files = sorted(glob(os.path.join(data_dir, "*.bin")))
    plot_files = all_files[:100]  # Process 100 frames (~5 seconds)
    n_frames = len(plot_files)
    n_prod = 32 * 33 // 2 
    
    freqs_mhz = np.linspace(
        KOTEKAN_GPU_FREQ_START_MHZ, KOTEKAN_GPU_FREQ_STOP_MHZ, 
        KOTEKAN_GPU_N_FREQ, endpoint=False
    )
    charts_array = CHARTSArray.default(n_ant=32)
    static_gains = np.load(gains_file)
    
    timestamps_plot, _ = resolve_gpu_frame_timestamps(
        plot_files, frame_cadence=KOTEKAN_GPU_FRAME_CADENCE_S
    )
    
    # ==============================================================================
    # 1. LOAD AND CALIBRATE ALL DATA 
    # ==============================================================================
    print("--- Loading and calibrating data tensor ---")
    vis_raw_all = np.zeros((KOTEKAN_GPU_N_FREQ, n_prod, n_frames), dtype=np.complex64)
    
    for t_idx, file_path in enumerate(plot_files):
        corr_raw = load_gpu_corr_frame(
            file_path, KOTEKAN_GPU_N_FREQ, KOTEKAN_GPU_N_RECEIVERS, KOTEKAN_GPU_N_POLARIZATIONS
        )
        vis_raw_all[:, :, t_idx] = gpu_corr_to_visibilities(corr_raw, KOTEKAN_GPU_N_POLARIZATIONS)
        
    # ==============================================================================
    # ANTENNA HEALTH MASKING (Using Autocorrelations)
    # ==============================================================================
    from charts_utils.cal.algorithms import get_auto_indices
    
    print("--- Running Antenna Health Diagnostics ---")
    # 1. Get the flat indices corresponding to the auto-correlations (V_00, V_11, etc.)
    auto_indices = get_auto_indices(32)
    
    # 2. Extract the time-averaged auto-correlation power for all 32 antennas 
    # across a stable channel (e.g., the middle of the band)
    mid_channel = KOTEKAN_GPU_N_FREQ // 2
    auto_powers = np.mean(np.abs(vis_raw_all[mid_channel, auto_indices, :]), axis=1)
    
    # 3. Use median statistics to find functional antennas
    median_auto = np.median(auto_powers)
    mad_auto = np.median(np.abs(auto_powers - median_auto))
    
    # Define functional bounds: Not completely dead (too low) and not saturating/unstable (too high)
    lower_bound = median_auto - (3 * 1.4826 * mad_auto)
    upper_bound = median_auto + (3 * 1.4826 * mad_auto)
    
    # Create a boolean mask of good antennas (True = Healthy, False = Dead/Unstable)
    good_antennas = (auto_powers > max(0, lower_bound)) & (auto_powers < upper_bound)
    functional_count = np.sum(good_antennas)
    
    print(f"  > Detected {functional_count}/32 functional antennas based on auto-correlation power.")
    print(f"  > Active Antennas: {np.where(good_antennas)[0]}")
    
    # 4. Zero-out the data for bad antennas in the visibility array
    # Any baseline (cross-correlation) involving a bad antenna must be set to 0.
    bad_antennas = np.where(~good_antennas)[0]
    
    # We iterate over the upper triangle logic to find products associated with bad antennas
    for ant_a in range(32):
        for ant_b in range(ant_a, 32):
            if ant_a in bad_antennas or ant_b in bad_antennas:
                # Calculate the flat index for this baseline
                flat_idx = ant_a * 32 - ant_a * (ant_a - 1) // 2 + (ant_b - ant_a)
                # Zero out the raw visibility data for all frequencies and times
                vis_raw_all[:, flat_idx, :] = 0.0 + 0.0j

    # CRITICAL FIX FOR MVDR: Because we are inserting zeros, the 32x32 covariance 
    # matrices will become strictly singular. We must increase the diagonal loading 
    # factor in compute_mvdr_weights (e.g., diag_loading=1e-2) to ensure invertibility, 
    # or mathematically shrink the matrix to K x K functional antennas before inversion.

    # Apply gains using the library's native function. 
    # Reshape gains to (n_freq, n_ant, 1) to broadcast over the time axis.
    gains_3d = static_gains[..., np.newaxis]
    vis_cal_all = apply_gains(vis_raw_all, gains_3d)

    # ==============================================================================
    # 2. CONVENTIONAL BEAMFORMING (Fast search)
    # Phase the visibilities and form the beam to find the transient.
    # ==============================================================================
    print("--- Running Conventional Beamforming ---")
    # We use a copy to avoid modifying the unphased calibrated data we need for MVDR
    vis_phased_all = fringestop_time(
        vis_cal_all.copy(), timestamps_plot, freqs_mhz, charts_array, 'sun', inplace=True
    )
    # form_beam collapses the phased visibilities into a 2D dynamic spectrum (n_freq, n_time)
    cb_spectrum = form_beam(vis_phased_all)
    
    # ==============================================================================
    # 3. DYNAMIC TRANSIENT LOCALIZATION
    # ==============================================================================
    print("--- Isolating RFI and detecting transient temporal window ---")
    
    # Identify RFI-inflicted channels via Median Absolute Deviation (MAD) of the temporal median.
    # Using the temporal median prevents a high-power transient (FRB) from inflating the 
    # noise floor estimation and being erroneously masked as RFI.
    channel_medians = np.median(cb_spectrum, axis=1)
    overall_median = np.median(channel_medians)
    mad_pwr = np.median(np.abs(channel_medians - overall_median))
    
    # Mask channels that are significantly above the noise floor (e.g., > 5 robust sigmas)
    rfi_threshold = overall_median + 5 * 1.4826 * mad_pwr
    is_clean_channel = (channel_medians < rfi_threshold)
    
    print(f"  > Detected {np.sum(~is_clean_channel)} RFI contaminated channels.")
    
    # Collapse clean channels to create a 1D temporal lightcurve
    clean_lightcurve = np.mean(cb_spectrum[is_clean_channel, :], axis=0)
    
    # Localize the FRB in time (Peak detection + Windowing)
    t_peak = int(np.argmax(clean_lightcurve))
    window_size = 5  # Margin of frames around the peak
    frb_start = max(0, t_peak - window_size)
    frb_end = min(n_frames, t_peak + window_size + 1)
    
    print(f"  > Transient detected around frame {t_peak}. Bounding box: [{frb_start}:{frb_end}]")
    
    # Create the off-pulse mask to estimate R_i+n
    off_pulse_mask = np.ones(n_frames, dtype=bool)
    off_pulse_mask[frb_start:frb_end] = False
    off_pulse_indices = np.where(off_pulse_mask)[0]

    # ==============================================================================
    # 4. R_i+n ESTIMATION & MVDR BEAMFORMING
    # ==============================================================================
    print("--- Estimating background covariance and computing MVDR ---")
    
    # Since V_ab = x_a * conj(x_b) inherently represents the outer product, 
    # the time-averaged covariance matrix is simply the mean of the visibilities.
    mean_bg_vis = np.mean(vis_cal_all[:, :, off_pulse_indices], axis=2)
    
    mvdr_spectrum = np.zeros((KOTEKAN_GPU_N_FREQ, n_frames))
    
    for f_idx, freq_mhz in enumerate(freqs_mhz):
        # 4.a: Reconstruct the RFI+Noise Hermitian matrix for this frequency
        R_in = utvec2mat(32, mean_bg_vis[f_idx])
        freq_hz = freq_mhz * 1e6
        
        for t_idx in range(n_frames):
            a_target = get_steering_vector(charts_array, timestamps_plot[t_idx], freq_hz, "sun")
            
            # 4.b: Compute robust weights (Notice we feed R_in, NOT the current frame data)
            w_mvdr = compute_mvdr_weights(R_in, a_target)
            
            # 4.c: Reconstruct current frame's observation matrix and apply weights
            R_obs = utvec2mat(32, vis_cal_all[f_idx, :, t_idx])
            mvdr_spectrum[f_idx, t_idx] = np.real(w_mvdr.conj().T @ R_obs @ w_mvdr)[0, 0]
            
        if (f_idx + 1) % 100 == 0:
            print(f"  Processed MVDR for {f_idx + 1}/{KOTEKAN_GPU_N_FREQ} frequency channels...")

    # ==============================================================================
    # 5. VISUALIZATION
    # ==============================================================================
    print("--- Generating Plot ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # We clip negative values to a small floor to safely apply log10
    cb_power = 10 * np.log10(np.clip(cb_spectrum, 1e-10, None))
    mvdr_power = 10 * np.log10(np.clip(mvdr_spectrum, 1e-10, None))
    
    im1 = ax1.imshow(cb_power, aspect='auto', origin='lower', 
                     extent=[0, n_frames, freqs_mhz[0], freqs_mhz[-1]], cmap='inferno')
    ax1.set_title("Conventional Beamforming (RFI Present)")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_xlabel("Time (Frames)")
    
    # Overlay the temporal bounding box on the CB plot to verify our dynamic detection
    ax1.axvline(x=frb_start, color='white', linestyle='--', alpha=0.7)
    ax1.axvline(x=frb_end, color='white', linestyle='--', alpha=0.7)
    
    im2 = ax2.imshow(mvdr_power, aspect='auto', origin='lower', 
                     extent=[0, n_frames, freqs_mhz[0], freqs_mhz[-1]], cmap='inferno')
    ax2.set_title("MVDR Beamforming (RFI Mitigated)")
    ax2.set_xlabel("Time (Frames)")
    
    plt.colorbar(im2, ax=[ax1, ax2], label="Power (dB)")
    plt.savefig("rfi_comparison_mvdr_dynamic.png", dpi=150, bbox_inches='tight')
    print("Plot exported as 'rfi_comparison_mvdr_dynamic.png'.")