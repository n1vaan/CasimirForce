import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn
from scipy.integrate import simpson
import os

H_BAR = 1.054e-34
C = 2.998e8

def y_top(x, a=1.0, A=0.25, lam=2.5, phase=0.0):
    return a + A * np.sin(2 * np.pi * x / lam + phase)

def build_panels(n=50, a=1.0, A=0.25, lam=2.5, phase=0.0):
    x = np.linspace(-3, 3, n)
    top = np.column_stack([x, y_top(x, a, A, lam, phase)])
    bot = np.column_stack([x, np.zeros_like(x)])
    starts = np.vstack([top[:-1], bot[:-1]])
    ends = np.vstack([top[1:], bot[1:]])
    centers = 0.5 * (starts + ends)
    return starts, ends, centers

def calculate_casimir_energy(a, A_amp, lam, n_panels=40):
    starts, ends, centers = build_panels(n_panels, a, A_amp, lam)
    N = len(centers)
    
    # Adjusted frequency range: lower frequencies carry the force at larger distances
    xi_points = np.logspace(8, 14, 25) 
    log_dets = []
    
    for xi in xi_points:
        kappa = xi / C
        M = np.zeros((N, N))
        vecs = ends - starts
        lengths = np.linalg.norm(vecs, axis=1)
        
        for j in range(N):
            for i in range(N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if i == j:
                    # Normalized diagonal: we set this to 1 to focus on INTERACTION
                    M[i, j] = 1.0 
                else:
                    # Interaction kernel
                    M[i, j] = lengths[j] * kn(0, kappa * dist)
        
        # log det(I + K) represents the interaction energy density
        (sign, logdet) = np.linalg.slogdet(M)
        log_dets.append(logdet)
    
    energy_integral = simpson(y=log_dets, x=xi_points)
    return (H_BAR / (2 * np.pi)) * energy_integral

def run_experiment():
    # Use a tighter range to see the curve clearly
    a_vals = np.linspace(0.5, 2.5, 100)
    print("Starting Lifshitz-BEM frequency integration...")
    energies = []
    for a in a_vals:
        E = calculate_casimir_energy(a, 0.3, 2.0)
        energies.append(E)
        print(f"Distance a={a:.2f} completed.")
        
    forces = -np.gradient(energies, a_vals)
    
    plt.figure(figsize=(8, 5))
    # Plot absolute value on log scale to see the power-law decay
    plt.plot(a_vals, np.abs(forces), 'o-', color='darkmagenta', label='|Lifshitz-BEM Force|')
    
    plt.yscale('log')
    plt.xlabel("Separation $a$")
    plt.ylabel("Force Magnitude (N)")
    plt.title("Refined Lifshitz-BEM Force Profile")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()

    # --- SAVE LOGIC ---
    os.makedirs("img", exist_ok=True) # Create folder if it doesn't exist
    save_path = os.path.join("img", "lifshitzbem.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {save_path}")
    # ------------------

    plt.show()

def run_lateral_experiment():
    # Keep distance constant, sweep phase
    a_fixed = 1.0
    phases = np.linspace(0, 2 * np.pi, 40)
    print(f"Starting Lateral Force sweep at a={a_fixed}...")
    
    energies = []
    for p in phases:
        # FIXED: Changed A_amp=0.3 to A=0.3 to match the function definition
        starts, ends, centers = build_panels(n=60, a=a_fixed, A=0.3, lam=2.0, phase=p)
        
        xi_points = np.logspace(8, 14, 20)
        log_dets = []
        for xi in xi_points:
            kappa = xi / C
            N = len(centers)
            M = np.eye(N) 
            vecs = ends - starts
            lengths = np.linalg.norm(vecs, axis=1)
            for j in range(N):
                for i in range(N):
                    if i != j:
                        dist = np.linalg.norm(centers[i] - centers[j])
                        M[i, j] = lengths[j] * kn(0, kappa * dist)
            _, logdet = np.linalg.slogdet(M)
            log_dets.append(logdet)
        
        E = (H_BAR / (2 * np.pi)) * simpson(y=log_dets, x=xi_points)
        energies.append(E)
        print(f"Phase {p/(np.pi):.2f}π completed.")

    lat_force = -np.gradient(energies, phases)

    plt.figure(figsize=(8, 5))
    plt.plot(phases / np.pi, lat_force, color='forestgreen', lw=2, marker='o', markersize=4)
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.xlabel("Phase Shift (units of π)")
    plt.ylabel("Lateral Force $F_x$ (N)")
    plt.title("Refined Lifshitz-BEM Lateral Force")
    plt.grid(True, alpha=0.3)

    os.makedirs("img", exist_ok=True)
    save_path = os.path.join("img", "lifshitz_lateral.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Lateral graph saved as {save_path}")
    plt.show()

def plot_data_driven_vectors(a_vals, v_forces, phases, l_forces, A, lam):
    """
    Uses the actual calculated sweep data to visualize the 
    force vector acting on the corrugated plate.
    """
    os.makedirs("img", exist_ok=True)
    
    # 1. Select a specific index to visualize (e.g., the midpoint of your sweep)
    # Or you can loop this to create frames for an animation
    idx_a = len(a_vals) // 4  # Pick a close separation for visibility
    idx_p = len(phases) // 4  # Pick a phase shift
    
    a = a_vals[idx_a]
    phi = phases[idx_p]
    
    # Get the ACTUAL calculated data points
    fy = v_forces[idx_a]
    fx = l_forces[idx_p]
    
    # 2. Setup Geometry
    starts, ends, centers = build_panels(n=80, a=a, A=A, lam=lam, phase=phi)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the plates
    plt.plot(centers[:40, 0], centers[:40, 1], 'k-', lw=3, label="Top Plate")
    plt.plot(centers[40:, 0], centers[40:, 1], 'k-', lw=3, label="Bottom Plate")
    
    # 3. Apply the Data-Driven Vector
    # We place the resultant force vector at the center of the corrugated plate
    origin_x = np.mean(centers[:40, 0])
    origin_y = np.mean(centers[:40, 1])
    
    # Normalize for visualization scale (Casimir forces are tiny!)
    scale_factor = 1e26  # Adjust this so the arrow is visible
    
    plt.quiver(origin_x, origin_y, fx * scale_factor, fy * scale_factor, 
               color='crimson', angles='xy', scale_units='xy', scale=1,
               width=0.01, label=f"Net Force Vector (Data Driven)")

    # 4. Labeling with actual data values
    plt.annotate(f"Fx: {fx:.2e} N\nFy: {fy:.2e} N", 
                 xy=(origin_x, origin_y), xytext=(20, 20),
                 textcoords='offset points', color='crimson', weight='bold')

    plt.title(f"Visualizing Force Data at a={a:.2f}, phase={phi/np.pi:.2f}π")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.legend()
    
    plt.savefig("img/casimir_data_vector.png", dpi=300)
    plt.show()
    
if __name__ == "__main__":
    # Parameters for the project
    a_val = 1.0
    amp = 0.3
    wavelength = 2.0
    
    # 1. Run the vertical force sweep
    run_experiment()
    
    # 2. Run the lateral force sweep
    run_lateral_experiment()
    
