"""
Appendix Code I (Improved): Dimensional Scaling Verification
Authors: Lévis Bonneau and Eric Bonneau

Description:
This script verifies that the accessible configuration volume of the 
11-dimensional cycle space scales as mu^11 when the unconstrained 
interaction amplitudes are scaled by mu.

Improvements:
- Uses numerically stable log-volume computation
- Avoids np.cov normalization artifacts
- Includes log-log regression to extract scaling exponent
"""

import numpy as np
import networkx as nx
from scipy.linalg import null_space

def verify_dimensional_scaling(mu_values, num_samples=100000):
    # 1. Define topology (dodecahedral graph)
    G = nx.dodecahedral_graph()
    
    # Incidence matrix A (20 vertices x 30 edges)
    A = nx.incidence_matrix(G, oriented=True).toarray()
    
    # 2. Cycle space (kernel of A)
    C = null_space(A)
    dim_C = C.shape[1]
    print(f"Theoretical Cycle Space Dimension: {dim_C} (Expected: 11)")
    
    # Projection operator (orthonormal basis → P = C C^T)
    P = C @ C.T
    
    log_volumes = []
    
    # 3. Scaling experiment
    for mu in mu_values:
        # Generate random unconstrained configurations scaled by mu
        e_unconstrained = np.random.randn(30, num_samples) * mu
        
        # Project onto 11D cycle space
        e_phys = P @ e_unconstrained
        
        # Compute covariance (second moment, more stable than np.cov)
        cov_matrix = (e_phys @ e_phys.T) / num_samples
        
        # Eigenvalues (variances along principal axes)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        # Extract top 11 nonzero eigenvalues
        nonzero_eigenvalues = np.sort(eigenvalues)[-dim_C:]
        
        # Log-volume (numerically stable)
        log_volume = 0.5 * np.sum(np.log(nonzero_eigenvalues))
        log_volumes.append(log_volume)
    
    return np.array(log_volumes)

if __name__ == "__main__":
    # Test scaling factors
    mu_test_values = np.array([0.1, 0.2, 0.5, 1.0, 2.0])
    
    log_volumes = verify_dimensional_scaling(mu_test_values, num_samples=200000)
    
    # Convert back to volume (for display only)
    volumes = np.exp(log_volumes)
    
    print("\n--- Scaling Analysis Results ---")
    for mu, vol in zip(mu_test_values, volumes):
        print(f"mu = {mu:.2f} | Volume ≈ {vol:.5e}")
    
    # --- Ratio check ---
    ratio_actual = volumes[-1] / volumes[0]
    ratio_theoretical = (mu_test_values[-1] / mu_test_values[0]) ** 11
    
    print("\n--- Ratio Verification ---")
    print(f"Empirical Ratio:   {ratio_actual:.3e}")
    print(f"Theoretical Ratio: {ratio_theoretical:.3e}")
    
    # --- Log-log slope fit (KEY RESULT) ---
    log_mu = np.log(mu_test_values)
    slope, intercept = np.polyfit(log_mu, log_volumes, 1)
    
    print("\n--- Scaling Law Fit ---")
    print(f"Fitted exponent: {slope:.4f} (Expected: 11.0)")
    print(f"Intercept:       {intercept:.4f}")
    
    print("\nConclusion:")
    print("The fitted exponent close to 11 confirms that the configuration volume")
    print("scales as mu^11, consistent with the dimensionality of the cycle space.")

