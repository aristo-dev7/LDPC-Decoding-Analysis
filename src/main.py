import numpy as np
import sys
import os
from decoders import hard_decision, min_sum_decoding
from simulation import simulate_ber

class DualLogger(object):
    """
    A custom logger that writes `print()` statements to both the terminal 
    and a specified text file simultaneously.
    """
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "w")

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.log_file.write(message)  # Write to file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def run_q1():
    print("="*50)
    print(" Q.1: LDPC DECODING (SINGLE VECTOR)")
    print("="*50)
    
    # Given H Matrix
    H = np.array([
        [0, 1, 0, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 0]
    ])
    
    # Given Received Vector & Noise Variance
    y = np.array([0.2, 1.1, -0.3, 0.9, 0.7, -0.1, 1.2, 0.4])
    noise_var = 2
    
    print("\n[ Executing Hard Decision Decoding ]")
    hd_out, hd_check = hard_decision(y, H)
    print(f"Final Decoded Output: {hd_out}")
    print(f"Parity Satisfied:     {hd_check}\n")
    
    print("\n[ Executing Min-Sum Algorithm Decoding ]")
    # Setting verbose=True here fulfills the requirement to show intermediate messages
    ms_out, ms_check = min_sum_decoding(y, H, noise_var, iterations=3, verbose=True)
    
    print("\n--- Final Q.1 Comparison ---")
    print(f"Hard Decision Result: {hd_out} | Satisfied: {hd_check}")
    print(f"Min-Sum Result:       {ms_out} | Satisfied: {ms_check}")
    print("\n")

def run_q2():
    print("="*50)
    print(" Q.2: BER VS SNR SIMULATION")
    print("="*50)
    
    H = np.array([
        [0, 1, 0, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 0]
    ])
    
    # Execute the plotting function
    simulate_ber(H, max_snr_db=10, trials=1000)

if __name__ == "__main__":
    # 1. Dynamically find the results folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Define the log file path
    log_path = os.path.join(results_dir, 'decoding_log.txt')
    
    # 3. Activate the Dual Logger
    sys.stdout = DualLogger(log_path)
    
    print(f"=== PROJECT LOGGING INITIATED ===")
    print(f"Saving all terminal output to: {log_path}\n")
    
    # 4. Run the assignment code
    run_q1()
    run_q2()
    
    print(f"\n=== EXECUTION COMPLETE ===")
    print(f"Results and plots successfully saved to the /results/ directory.")