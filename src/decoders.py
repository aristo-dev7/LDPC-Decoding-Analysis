import numpy as np

def hard_decision(y, H):
    """
    Performs hard decision bit-flipping decoding based on received vector y.
    """
    # Map > 0 to 0 (BPSK +1), and <= 0 to 1 (BPSK -1)
    x_hat = (y <= 0).astype(int)
    
    # Calculate parity check: H * x_hat (mod 2)
    parity = np.dot(H, x_hat) % 2
    is_satisfied = np.all(parity == 0)
    
    return x_hat, is_satisfied

def min_sum_decoding(y, H, noise_var, iterations=3, verbose=False):
    """
    Implements Soft-Decision LDPC decoding using the Min-Sum Algorithm (MSA).
    """
    rows, cols = H.shape
    
    # Initial Log-Likelihood Ratios (LLR)
    L_ci = 2 * y / noise_var
    
    # Initialize message matrices
    M_vc = np.zeros((rows, cols)) # Variable to Check node messages
    for r in range(rows):
        for c in range(cols):
            if H[r, c] == 1:
                M_vc[r, c] = L_ci[c]

    M_cv = np.zeros((rows, cols)) # Check to Variable node messages
    
    if verbose:
        print(f"--- Initial LLRs ---\n{np.round(L_ci, 3)}")
    
    for iter in range(iterations):
        # 1. Check Node Update (M_c->v)
        for r in range(rows):
            connected_vars = np.where(H[r, :] == 1)[0]
            for c in connected_vars:
                # Gather messages from all connected variable nodes EXCEPT current 'c'
                others = [M_vc[r, v] for v in connected_vars if v != c]
                
                # Multiply signs and find the minimum absolute value
                sign = np.prod(np.sign(others))
                min_val = np.min(np.abs(others))
                M_cv[r, c] = sign * min_val
        
        # 2. Variable/ Bit Node Update (M_v->c)
        L_total = np.zeros(cols)
        for c in range(cols):
            connected_checks = np.where(H[:, c] == 1)[0]
            
            # Total LLR is Initial LLR + sum of messages from connected check nodes
            L_total[c] = L_ci[c] + np.sum(M_cv[connected_checks, c])
            
            # Send updated messages back to check nodes (excluding the message from that specific check node)
            for r in connected_checks:
                M_vc[r, c] = L_total[c] - M_cv[r, c]

        # 3. Hard Decision & Parity Check
        x_hat = (L_total <= 0).astype(int)
        parity = np.dot(H, x_hat) % 2
        success = np.all(parity == 0)
        
        if verbose:
            print(f"\n--- Iteration {iter+1} ---")
            print(f"Check to Variable (M_cv):\n{np.round(M_cv, 2)}")
            print(f"Total LLRs: {np.round(L_total, 2)}")
            print(f"Decoded Bits: {x_hat}")
            print(f"Parity Satisfied: {success}")
        
        if success:
            if verbose:
                print("\nParity check satisfied!")
            # Ensures it runs for at least 3 iterations (iter is 0-indexed, so 2 means 3rd iteration)
            if iter >= 2: 
                if verbose:
                    print("Completed minimum 3 iterations. Stopping.")
                break
            else:
                if verbose:
                    print("Continuing to satisfy the 'at least 3 iterations' requirement...")
            
    return x_hat, success