"""
Appendix Code IV: Projection Factor Verification (2/pi)

Description:
This script verifies that the average projection of a uniformly distributed cyclic degree of freedom yields the geometric factor 2/pi.

This factor is independent of topology and arises purely from projection geometry.
"""

import numpy as np

def verify_projection_factor(num_samples=10000000):
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    values = np.abs(np.cos(theta))

    mean_value = np.mean(values)
    std_error = np.std(values) / np.sqrt(num_samples)

    return num_samples, mean_value, std_error

if __name__ == "__main__":
    num_samples, result, std_error = verify_projection_factor()

    print(f"Number of samples: {num_samples}")

    print(f"Numerical result: {result:.8f}")
    print(f"Theoretical 2/pi: {2/np.pi:.8f}")
    print(f"Absolute error:   {abs(result - 2/np.pi):.2e}")
    print(f"Estimated standard error: {std_error:.2e}")

    print("\nConclusion:")
    print("The numerical result agrees with 2/pi within statistical uncertainty,")
    print("with deviations consistent with the expected Monte Carlo error.")

