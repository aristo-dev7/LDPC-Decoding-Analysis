import numpy as np
import os
import matplotlib.pyplot as plt
from decoders import min_sum_decoding

def simulate_ber(H, max_snr_db=10, trials=1000):
    """
    Simulates Bit Error Rate (BER) vs SNR for Min-Sum LDPC decoding.
    """
    print(f"Running Monte Carlo Simulation ({trials} trials per SNR point)...")
    snr_db = np.arange(0, max_snr_db + 1, 1)
    ber_list = []
    
    cols = H.shape[1]
    
    for snr in snr_db:
        # Convert SNR (dB) to linear scale
        snr_lin = 10**(snr/10)
        
        # Calculate noise variance for BPSK
        noise_var = 1 / (2 * snr_lin)
        sigma = np.sqrt(noise_var)
        
        errors = 0
        for _ in range(trials):
            # Transmit all 0 bits (which map to +1 in BPSK)
            tx = np.ones(cols)
            
            # Add AWGN
            noise = np.random.normal(0, sigma, cols)
            rx = tx + noise
            
            # Decode using Min-Sum (verbose=False to keep console clean)
            decoded, _ = min_sum_decoding(rx, H, noise_var, iterations=5, verbose=False)
            
            # Count bit errors (any bit flipped to 1 is an error)
            errors += np.sum(decoded != 0)
            
        current_ber = errors / (trials * cols)
        ber_list.append(current_ber)
        print(f"SNR: {snr:2d} dB | BER: {current_ber:.5f}")
    
    # Generate the Plot
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_db, ber_list, 'b-o', linewidth=2, markersize=6)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs SNR for LDPC Code (Min-Sum Algorithm)', fontsize=14)
    
    # Save the plot to the directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, 'ber_vs_snr.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to:\n{save_path}")
    
    # Display the plot
    plt.show()