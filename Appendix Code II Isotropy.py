"""
Appendix Code II: 11D Physical Manifold Isotropic Analysis
Authors: Lévis Bonneau and Eric Bonneau
Description: 
This script analyzes the active 11D cycle space (the physical manifold) 
to verify that the dynamically allowed interactions exhibit statistically 
isotropic behavior. This geometric isotropy mirrors the observed 
isotropic nature of the gravitational field, supporting the mu11 scaling.
"""

import numpy as np
import networkx as nx
from scipy.linalg import null_space

def analyze_cycle_space_isotropy(num_samples=10000000):
    # 1. Define topology and matrices
    G = nx.dodecahedral_graph()
    A = nx.incidence_matrix(G, oriented=True).toarray()
    
    # C represents the 11 basis vectors of the cycle space
    C = null_space(A) 
    
    # 2. Projection operator P onto the 11D cycle space
    # P = C @ C^+ (Standard projection onto column space)
    P = C @ np.linalg.pinv(C)

    # 3. Generate random unconstrained configurations (representing raw potential)
    e_unconstrained = np.random.randn(30, num_samples)
    
    # 4. Project the raw potential strictly into the allowed 11D physical manifold
    e_11d = P @ e_unconstrained
    
    # 5. Statistical analysis of the 11D space
    # Check for mean zero (no net directional bias)
    mean_11d = np.mean(e_11d, axis=1)
    max_mean_deviation = np.max(np.abs(mean_11d))
    
    # Calculate covariance to check for directional preference (anisotropy)
    cov_11d = np.cov(e_11d)
    eigenvalues = np.linalg.eigvalsh(cov_11d)
    
    # The physical cycle space has dimension 11
    # If the 11D space is purely isotropic, its 11 active eigenvalues should be identical
    nonzero_evals = np.sort(eigenvalues)[-11:]
    
    variance_ratio = np.max(nonzero_evals) / np.min(nonzero_evals)
    
    return max_mean_deviation, nonzero_evals, variance_ratio

if __name__ == "__main__":
    max_drift, evals, isotropy_ratio = analyze_cycle_space_isotropy()
    
    print("--- 11D Physical Manifold Isotropic Analysis ---")
    print(f"Maximum mean deviation from zero: {max_drift:.5e} (Expected ~0)")
    print(f"Isotropy Variance Ratio (Max/Min Eigenvalue): {isotropy_ratio:.5f} (Expected ~1.0)")

    print("\nEigenvalues of the 11D covariance matrix (top 5):")
    print(evals[-5:])

    print(f"\nMean eigenvalue: {np.mean(evals):.5f}")
    print(f"Eigenvalue std deviation: {np.std(evals):.5f} (Lower = more isotropic)")

    print("\nAll 11 eigenvalues:")
    print(evals)

    print("\nConclusion: A variance ratio close to 1.0 indicates statistical isotropy.")
    print("This result is consistent with the absence of preferred directions, ")
    print("a necessary condition for compatibility with observed macroscopic isotropy.")