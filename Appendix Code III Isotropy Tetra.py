"""
Appendix Code III (Final): Robust Isotropy Analysis in Tetrahedral Cycle Space
Authors: Lévis Bonneau and Eric Bonneau

Description:
This script analyzes whether projection onto the tetrahedral cycle space
preserves or induces isotropy under different input conditions.

Three tests are performed:
1. Isotropic Gaussian input (baseline)
2. Anisotropic diagonal covariance input
3. Fully random covariance input

This ensures that isotropy is not a trivial consequence of isotropic sampling.
"""

import numpy as np
import networkx as nx
from scipy.linalg import null_space

# -------------------------------
# Utility: Compute isotropy stats
# -------------------------------
def compute_isotropy(e_projected, dim_N):
    num_samples = e_projected.shape[1]

    # Mean drift
    mean_vec = np.mean(e_projected, axis=1)
    max_drift = np.max(np.abs(mean_vec))

    # Covariance (faster form)
    cov = (e_projected @ e_projected.T) / num_samples

    # Eigenvalues
    evals = np.linalg.eigvalsh(cov)
    nonzero = np.sort(evals)[-dim_N:]

    ratio = np.max(nonzero) / np.min(nonzero)

    return max_drift, nonzero, ratio


# -------------------------------
# Main analysis function
# -------------------------------
def analyze_tetrahedral_isotropy(num_samples=500000):
    # 1. Tetrahedral graph (K4)
    G = nx.complete_graph(4)
    A = nx.incidence_matrix(G, oriented=True).toarray()

    # 2. Cycle space (dimension = 3)
    C = null_space(A)
    dim_N = C.shape[1]

    # 3. Projection operator (orthonormal basis)
    P = C @ C.T

    print("\n--- TETRAHEDRAL ISOTROPY ANALYSIS ---")
    print(f"Cycle space dimension: {dim_N}\n")

    # ===============================
    # TEST 1: Isotropic Gaussian input
    # ===============================
    e_iso = np.random.randn(6, num_samples)
    e_proj_iso = P @ e_iso

    drift1, evals1, ratio1 = compute_isotropy(e_proj_iso, dim_N)

    print("Test 1: Isotropic Gaussian Input")
    print(f"Max mean drift: {drift1:.3e}")
    print(f"Eigenvalues: {evals1}")
    print(f"Variance ratio: {ratio1:.5f}\n")

    # ===============================
    # TEST 2: Anisotropic diagonal input
    # ===============================
    diag_cov = np.diag([1, 2, 3, 4, 5, 6])
    e_aniso = np.random.multivariate_normal(
        mean=np.zeros(6), cov=diag_cov, size=num_samples
    ).T

    e_proj_aniso = P @ e_aniso

    drift2, evals2, ratio2 = compute_isotropy(e_proj_aniso, dim_N)

    print("Test 2: Anisotropic Diagonal Input")
    print(f"Max mean drift: {drift2:.3e}")
    print(f"Eigenvalues: {evals2}")
    print(f"Variance ratio: {ratio2:.5f}\n")

    # ===============================
    # TEST 3: Random covariance input
    # ===============================
    A_rand = np.random.randn(6, 6)
    cov_rand = A_rand @ A_rand.T  # SPD matrix

    e_rand = np.random.multivariate_normal(
        mean=np.zeros(6), cov=cov_rand, size=num_samples
    ).T

    e_proj_rand = P @ e_rand

    drift3, evals3, ratio3 = compute_isotropy(e_proj_rand, dim_N)

    print("Test 3: Random Covariance Input")
    print(f"Max mean drift: {drift3:.3e}")
    print(f"Eigenvalues: {evals3}")
    print(f"Variance ratio: {ratio3:.5f}\n")

    # -------------------------------
    # Interpretation
    # -------------------------------
    print("--- INTERPRETATION ---")

    def interpret(r):
        if abs(r - 1.0) < 0.02:
            return "Isotropic"
        elif r < 1.2:
            return "Weak anisotropy"
        else:
            return "Anisotropic"

    print(f"Test 1: {interpret(ratio1)}")
    print(f"Test 2: {interpret(ratio2)}")
    print(f"Test 3: {interpret(ratio3)}")

    print("\nConclusion:")
    print("The projection does not introduce directional bias")
    print("and does not enforce isotropy.")
    print("It preserves the statistical structure of the input within the admissible subspace.")
    print("Isotropy is preserved under isotropic sampling,")
    print("but is not enforced under anisotropic inputs.")
    print("Thus, the cycle space behaves as a neutral geometric subspace.")
    print("Anisotropic components aligned with the admissible subspace are preserved,")
    print("while incompatible components are suppressed.")

    return evals1, evals2, evals3

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    evals1, evals2, evals3 = analyze_tetrahedral_isotropy()

    print("\nEigenvalues for Figure 3:")
    print("Isotropic:", evals1)
    print("Diagonal:", evals2)
    print("Random:", evals3)

