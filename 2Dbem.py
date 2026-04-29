import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def y_top(x, a=1.0, A=0.25, lam=2.5, phase=0.0):
    """Geometry of the corrugated top plate."""
    return a + A * np.sin(2 * np.pi * x / lam + phase)

def build_panels(n=120, a=1.0, A=0.25, lam=2.5, phase=0.0, x_domain=(-4, 4)):
    """Creates panel geometry for both plates."""
    x = np.linspace(x_domain[0], x_domain[1], n)
    top = np.column_stack([x, y_top(x, a, A, lam, phase)])
    bot = np.column_stack([x, np.zeros_like(x)])

    starts = np.vstack([top[:-1], bot[:-1]])
    ends = np.vstack([top[1:], bot[1:]])
    centers = 0.5 * (starts + ends)

    n_top = len(top) - 1
    n_bot = len(bot) - 1
    return starts, ends, centers, n_top, n_bot

def build_matrix_vectorized(starts, ends, centers, n_quad=16):
    """
    Builds the BEM matrix using vectorized quadrature and 
    analytical self-terms for the diagonal log-singularity.
    """
    N = len(centers)
    A_mat = np.zeros((N, N))
    
    vecs = ends - starts
    lengths = np.linalg.norm(vecs, axis=1)
    
    # Gaussian quadrature nodes/weights
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    t_quad = 0.5 * (nodes + 1)
    w_quad = 0.5 * weights

    for j in range(N):
        # Local panel j geometry
        p_start = starts[j]
        p_vec = vecs[j]
        L = lengths[j]
        
        # Analytic self-term for i == j (integral of -1/2pi * ln|r-rp|)
        # Integral of -1/2pi * ln|x| from -L/2 to L/2
        A_mat[j, j] = (L / (2 * np.pi)) * (1 - np.log(L / 2))
        
        # Interaction with other panels i != j
        # Quadrature points on panel j
        pts = p_start[None, :] + t_quad[:, None] * p_vec[None, :]
        
        for i in range(N):
            if i == j: continue
            dists = np.linalg.norm(centers[i] - pts, axis=1)
            # Standard Green's function integration
            G = -(1 / (2 * np.pi)) * np.log(dists)
            A_mat[i, j] = L * np.sum(w_quad * G)
            
    return A_mat

def solve_bem(a=1.0, phase=0.0, n=100, A_amp=0.25, lam=2.5):
    """Solves the BEM system and returns energy and geometry."""
    starts, ends, centers, n_top, n_bot = build_panels(n=n, a=a, A=A_amp, lam=lam, phase=phase)
    M = build_matrix_vectorized(starts, ends, centers)
    
    # Potentials: top = 1, bottom = -1 (delta V = 2)
    b = np.concatenate([np.ones(n_top), -np.ones(n_bot)])
    sigma = np.linalg.solve(M, b)
    
    # Energy in the electrostatic analogy
    energy = 0.5 * sigma @ (M @ sigma)
    return energy, sigma, centers, n_top

def calculate_pfa(a, A_amp, lam, x_domain=(-4, 4)):
    """Proximity Force Approximation: integrates local 1/h(x)."""
    func = lambda x: 1.0 / (a + A_amp * np.sin(2 * np.pi * x / lam))
    val, _ = quad(func, x_domain[0], x_domain[1])
    return val

def run_casimir_analysis():
    # Parameters
    A_amp = 0.3
    lam = 2.0
    n_panels = 80
    a_vals = np.linspace(0.8, 2.0, 10)
    
    print("--- Running Vertical Sweep (BEM vs PFA) ---")
    bem_energies = []
    pfa_energies = []
    
    for a in a_vals:
        E_bem, _, _, _ = solve_bem(a=a, n=n_panels, A_amp=A_amp, lam=lam)
        E_pfa = calculate_pfa(a, A_amp, lam)
        bem_energies.append(E_bem)
        pfa_energies.append(E_pfa)
        print(f"a: {a:.2f} | BEM: {E_bem:.4f} | PFA: {E_pfa:.4f}")

    # Forces
    bem_force = -np.gradient(bem_energies, a_vals)
    pfa_force = -np.gradient(pfa_energies, a_vals)

    print("\n--- Running Lateral Force Sweep ---")
    phases = np.linspace(0, 2*np.pi, 15)
    lat_energies = []
    for p in phases:
        E, _, _, _ = solve_bem(a=1.2, phase=p, n=n_panels, A_amp=A_amp, lam=lam)
        lat_energies.append(E)
    
    lat_force = -np.gradient(lat_energies, phases)

    # Plotting
    os.makedirs("img", exist_ok=True)
    
    # Plot 1: BEM vs PFA
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(a_vals, bem_force, 'bo-', label="BEM (Full Geometry)")
    plt.plot(a_vals, pfa_force, 'r--', label="PFA (Local Approximation)")
    plt.xlabel("Separation (a)")
    plt.ylabel("Vertical Force")
    plt.title("Beyond-PFA Effects")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Lateral Force
    plt.subplot(1, 2, 2)
    plt.plot(phases / np.pi, lat_force, 'g-o')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("Phase Shift (units of $\pi$)")
    plt.ylabel("Lateral Force ($F_x$)")
    plt.title("Non-Trivial Lateral Casimir Force")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("img/casimir_results.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_casimir_analysis()