import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import inv

# CHARTS core and calibration imports
from charts_utils.core.io import load_gpu_corr_frame, gpu_corr_to_visibilities
from charts_utils.core.gpu_frames import resolve_gpu_frame_timestamps
from charts_utils.cal.algorithms import utvec2mat, solve_gain
from charts_utils.cal.coordinates import CHARTSArray, _reference_delays
from charts_utils.core.kotekan import (
    KOTEKAN_GPU_N_FREQ, KOTEKAN_GPU_N_RECEIVERS, KOTEKAN_GPU_N_POLARIZATIONS,
    KOTEKAN_GPU_FRAME_CADENCE_S, KOTEKAN_GPU_FREQ_START_MHZ, KOTEKAN_GPU_FREQ_STOP_MHZ
)

def compute_mvdr_weights(R_obs, a_target, diag_loading_factor=1e-5):
    """
    Computes the Minimum Variance Distortionless Response (MVDR) beamformer weights.
    """
    n_ant = R_obs.shape[0]
    mean_power = np.real(np.trace(R_obs)) / n_ant
    R_reg = R_obs + (diag_loading_factor * mean_power) * np.eye(n_ant)
    
    R_inv = inv(R_reg)
    numerator = R_inv @ a_target
    denominator = np.real(a_target.conj().T @ numerator)
    return numerator / denominator

def get_steering_vector(array, timestamp_unix, freq_hz, target="sun"):
    """
    Computes the steering vector 'a' towards a celestial target.
    """
    delays_tau = _reference_delays(timestamp_unix, array, target)
    a_target = np.exp(-1j * 2.0 * np.pi * freq_hz * delays_tau).reshape(-1, 1)
    return a_target

# ==========================================
# MAIN TWO-PASS PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    data_dir = "../corr_24h_2608"
    all_files = sorted(glob(os.path.join(data_dir, "*.bin")))
    
    if not all_files:
        raise FileNotFoundError(f"No .bin files found in {data_dir}")
    
    # Configuration limits to protect HPC Login Nodes
    # Pass 1: ~1200 frames (approx 60s) to integrate the covariance matrix
    #n_integration_frames = min(1200, len(all_files)) 
    #integration_files = all_files[:n_integration_frames]
    
    # Configuration limits to protect HPC Login Nodes
    # Pass 1: ~200 frames for a quick debug integration
    n_integration_frames = min(200, len(all_files))
    integration_files = all_files[:n_integration_frames]



    # Pass 2: 50 frames to plot the dynamic spectrum visualization
    plot_files = all_files[:50]
    
    # Shared Array/Frequency Setup
    freqs_mhz = np.linspace(
        KOTEKAN_GPU_FREQ_START_MHZ, KOTEKAN_GPU_FREQ_STOP_MHZ, 
        KOTEKAN_GPU_N_FREQ, endpoint=False
    )
    charts_array = CHARTSArray.default(n_ant=32)
    
    # ---------------------------------------------------------
    # PASS 1: INTEGRATION AND STATIC GAIN EXTRACTION
    # ---------------------------------------------------------
    print(f"--- PASS 1: Integrating {n_integration_frames} frames ---")
    
    # Accumulator for the phased covariance matrices [672 channels, 32x32 matrix]
    integrated_R = np.zeros((KOTEKAN_GPU_N_FREQ, 32, 32), dtype=np.complex64)
    
    timestamps_integ, _ = resolve_gpu_frame_timestamps(
        integration_files, frame_cadence=KOTEKAN_GPU_FRAME_CADENCE_S
    )
    
    for t_idx, file_path in enumerate(integration_files):
        corr_raw = load_gpu_corr_frame(
            path=file_path, n_freq=KOTEKAN_GPU_N_FREQ,
            n_recv=KOTEKAN_GPU_N_RECEIVERS, n_pol=KOTEKAN_GPU_N_POLARIZATIONS
        )
        visibilities = gpu_corr_to_visibilities(corr_raw, n_pol=KOTEKAN_GPU_N_POLARIZATIONS)
        obs_time = timestamps_integ[t_idx]
        
        for f_idx, freq_mhz in enumerate(freqs_mhz):
            freq_hz = freq_mhz * 1e6
            R_obs = utvec2mat(32, visibilities[f_idx, :])
            a_sun = get_steering_vector(charts_array, obs_time, freq_hz, target="sun")
            
            # Phase to the target and accumulate
            phase_matrix = a_sun.conj() @ a_sun.T
            integrated_R[f_idx] += (R_obs * phase_matrix)
            
        if (t_idx + 1) % 100 == 0:
            print(f"  Processed {t_idx + 1}/{n_integration_frames} integration frames...")
            
    # Calculate the time-average of the accumulated matrices
    integrated_R /= n_integration_frames
    
    print("--- PASS 1: Solving WALS static gains ---")
    # Array to hold the static complex gains for all 672 channels
    static_gains = np.zeros((KOTEKAN_GPU_N_FREQ, 32), dtype=np.complex64)
    
    for f_idx in range(KOTEKAN_GPU_N_FREQ):
        vis_phased_1d = integrated_R[f_idx][np.triu_indices(32)]
        vis_phased_3d = vis_phased_1d.reshape(1, -1, 1)
        
        # Solve gains on the ultra-stable, RFI-washed integrated matrix
        resultado_wals = solve_gain(vis_phased_3d)
        static_gains[f_idx] = resultado_wals[0][0, :, 0]
        
    # ---------------------------------------------------------
    # PASS 2: APPLY STATIC GAINS AND COMPUTE DYNAMIC SPECTRUM
    # ---------------------------------------------------------
    print(f"--- PASS 2: Processing {len(plot_files)} frames for visualization ---")
    
    conventional_spectrum = np.zeros((KOTEKAN_GPU_N_FREQ, len(plot_files)))
    mvdr_spectrum = np.zeros((KOTEKAN_GPU_N_FREQ, len(plot_files)))
    
    timestamps_plot, _ = resolve_gpu_frame_timestamps(
        plot_files, frame_cadence=KOTEKAN_GPU_FRAME_CADENCE_S
    )
    
    for t_idx, file_path in enumerate(plot_files):
        corr_raw = load_gpu_corr_frame(
            path=file_path, n_freq=KOTEKAN_GPU_N_FREQ,
            n_recv=KOTEKAN_GPU_N_RECEIVERS, n_pol=KOTEKAN_GPU_N_POLARIZATIONS
        )
        visibilities = gpu_corr_to_visibilities(corr_raw, n_pol=KOTEKAN_GPU_N_POLARIZATIONS)
        obs_time = timestamps_plot[t_idx]
        
        for f_idx, freq_mhz in enumerate(freqs_mhz):
            freq_hz = freq_mhz * 1e6
            R_obs = utvec2mat(32, visibilities[f_idx, :])
            a_sun = get_steering_vector(charts_array, obs_time, freq_hz, target="sun")
            
            # Phase raw data to target
            phase_matrix = a_sun.conj() @ a_sun.T
            R_phased = R_obs * phase_matrix
            
            # Apply the pre-calculated STATIC gains for this channel
            channel_gains = static_gains[f_idx]
            gain_matrix = np.outer(channel_gains, channel_gains.conj())
            R_cal = R_phased / gain_matrix
            
            # Mathematical steering vector is now simply a column of 1s
            a_calibrated = np.ones((32, 1), dtype=np.complex64)
            
            # Conventional Beamforming
            w_cb = a_calibrated / 32
            power_cb = np.real(w_cb.conj().T @ R_cal @ w_cb)[0, 0]
            conventional_spectrum[f_idx, t_idx] = power_cb
            
            # MVDR Beamforming
            w_mvdr = compute_mvdr_weights(R_cal, a_calibrated)
            power_mvdr = np.real(w_mvdr.conj().T @ R_cal @ w_mvdr)[0, 0]
            mvdr_spectrum[f_idx, t_idx] = power_mvdr

    # ---------------------------------------------------------
    # PLOTTING
    # ---------------------------------------------------------
    print("--- GENERATING PLOT ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    im1 = ax1.imshow(10 * np.log10(conventional_spectrum), aspect='auto', origin='lower', 
                    extent=[0, len(plot_files), freqs_mhz[0], freqs_mhz[-1]], cmap='inferno')
    ax1.set_title("Conventional Beamforming (Dirty)")
    ax1.set_ylabel("Frequency (MHz)")
    ax1.set_xlabel("Time (Frames)")
    
    im2 = ax2.imshow(10 * np.log10(mvdr_spectrum), aspect='auto', origin='lower', 
                    extent=[0, len(plot_files), freqs_mhz[0], freqs_mhz[-1]], cmap='inferno')
    ax2.set_title("MVDR (Mitigated + Calibrated)")
    ax2.set_xlabel("Time (Frames)")
    
    plt.colorbar(im2, ax=[ax1, ax2], label="Power (dB)")
    plt.savefig("rfi_comparison_integrated.png", dpi=150, bbox_inches='tight')
    print("Plot exported as 'rfi_comparison_integrated.png'.")