import os
import numpy as np
from glob import glob

from charts_utils.core.io import load_gpu_corr_frame, gpu_corr_to_visibilities
from charts_utils.core.gpu_frames import resolve_gpu_frame_timestamps
from charts_utils.cal.algorithms import solve_gain
from charts_utils.cal.coordinates import CHARTSArray, fringestop_time
from charts_utils.core.kotekan import (
    KOTEKAN_GPU_N_FREQ, KOTEKAN_GPU_N_RECEIVERS, KOTEKAN_GPU_N_POLARIZATIONS,
    KOTEKAN_GPU_FRAME_CADENCE_S, KOTEKAN_GPU_FREQ_START_MHZ, KOTEKAN_GPU_FREQ_STOP_MHZ
)

def compute_24h_calibration(data_dir: str, output_file: str = "static_gains_24h.npy"):
    """
    Integrates GPU correlation frames to extract static instrumental gains 
    using the Weighted Alternating Least Squares (WALS) algorithm.
    
    Optimized for HPC environments by keeping data as upper-triangular vectors
    to prevent memory overflow on compute nodes.
    """
    all_files = sorted(glob(os.path.join(data_dir, "*.bin")))
    if not all_files:
        raise FileNotFoundError(f"No .bin files found in {data_dir}.")
    
# Downsample: process 1 frame every 100 frames to accelerate the 24h integration
    # This reduces the I/O load on the GPFS parallel file system while maintaining 
    # statistical validity for the WALS calibration.
    #integration_files = all_files[::100]
    #n_integration_frames = len(integration_files)
# Load all available files for integration without downsampling
    integration_files = all_files
    n_integration_frames = len(integration_files)

    # Define frequency axis in MHz
    freqs_mhz = np.linspace(
        KOTEKAN_GPU_FREQ_START_MHZ, KOTEKAN_GPU_FREQ_STOP_MHZ, 
        KOTEKAN_GPU_N_FREQ, endpoint=False
    )
    
    # Initialize standard 32-element CHARTS array topology
    charts_array = CHARTSArray.default(n_ant=32)

    print(f"--- Starting integration of {n_integration_frames} frames ---")
    
    # Resolve timestamps from the FPGA metadata headers
    timestamps_integ, _ = resolve_gpu_frame_timestamps(
        integration_files, frame_cadence=KOTEKAN_GPU_FRAME_CADENCE_S
    )

    # A 32-antenna array produces 528 baselines (upper triangle including auto-correlations)
    n_prod = 32 * 33 // 2 
    
    # Direct vectorial accumulator (saves RAM by keeping data in 1D baseline format)
    integrated_vis = np.zeros((KOTEKAN_GPU_N_FREQ, n_prod), dtype=np.complex64)

    for t_idx, file_path in enumerate(integration_files):
        # 1. Load the raw 32-bit integer GPU correlation binary
        corr_raw = load_gpu_corr_frame(
            path=file_path, n_freq=KOTEKAN_GPU_N_FREQ,
            n_recv=KOTEKAN_GPU_N_RECEIVERS, n_pol=KOTEKAN_GPU_N_POLARIZATIONS
        )
        
        # 2. Reorder into upper-triangle visibilities following V_ab = x_a * conj(x_b)
        vis = gpu_corr_to_visibilities(corr_raw, n_pol=KOTEKAN_GPU_N_POLARIZATIONS)
        
        # 3. Reshape for fringestop_time which expects a (n_freq, n_prod, n_time) tensor
        vis_3d = vis[..., np.newaxis]
        obs_time = np.array([timestamps_integ[t_idx]])
        
        # 4. Apply inverse geometric phase to phase center on the Sun
        # fringestop_time interprets unitless frequency arrays as MHz internally
        vis_phased = fringestop_time(
            timestream=vis_3d, 
            times=obs_time, 
            freq=freqs_mhz, 
            array_obj=charts_array, 
            src_name='sun', 
            inplace=False
        )
        
        # 5. Accumulate the phased visibilities
        integrated_vis += vis_phased[..., 0]
        
        if (t_idx + 1) % 50 == 0:
            print(f"  Processed {t_idx + 1}/{n_integration_frames} frames...")
            
    # Compute the time-average of the accumulated visibilities
    integrated_vis /= n_integration_frames
    
    print("--- Extracting instrumental gains (WALS) ---")
    
    # solve_gain natively operates on (n_freq, n_prod, n_time) tensors
    integrated_vis_3d = integrated_vis[..., np.newaxis]
    
    # Extract gains. The returned tuple structure from solve_gain is (evalue, gain, gain_error).
    # We omit optional parameters (frequencies, source_name) to use default scaling.
    wals_result = solve_gain(integrated_vis_3d)
    gains_3d = wals_result[1] 
    
    # Extract static gains (shape: n_freq, n_ant) and save to disk
    static_gains = gains_3d[:, :, 0]
    np.save(output_file, static_gains)
    print(f"--- Calibration completed. Gains exported to '{output_file}' ---")

if __name__ == "__main__":
    compute_24h_calibration("../corr_24h_2608")