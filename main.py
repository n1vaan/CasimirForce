import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GEOMETRY
# ============================================================

def y_bottom(x):
    """
    Flat bottom plate at y = 0.
    """
    return np.zeros_like(x)


def y_top(x, a=1.0, A=0.2, lam=2.0):
    """
    Corrugated top surface:
        y = a + A * sin(2*pi*x/lam)

    Parameters
    ----------
    x : array
        x-coordinates
    a : float
        Mean separation between the two surfaces
    A : float
        Corrugation amplitude
    lam : float
        Corrugation wavelength
    """
    return a + A * np.sin(2 * np.pi * x / lam)


# ============================================================
# 2. WORLDLINE GENERATION
# ============================================================

def generate_closed_loop(n_points=200, loop_scale=0.3, rng=None):
    """
    Generate a random closed loop in 2D.

    Idea:
    - Make random steps
    - Accumulate into a path
    - Remove drift so it closes approximately
    - Center it at the origin

    Returns
    -------
    loop : ndarray of shape (n_points, 2)
        The loop coordinates
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random 2D steps
    steps = rng.normal(size=(n_points, 2))

    # Build path
    path = np.cumsum(steps, axis=0)

    # Remove linear drift so final point returns near start
    drift = np.linspace(0, 1, n_points)[:, None] * path[-1]
    loop = path - drift

    # Center loop at origin
    loop -= loop.mean(axis=0)

    # Normalize size, then scale
    rms = np.sqrt(np.mean(np.sum(loop**2, axis=1)))
    if rms > 0:
        loop = loop / rms

    loop *= loop_scale
    return loop


def place_loop(loop, center_x, center_y):
    """
    Translate a loop to a chosen center.
    """
    placed = loop.copy()
    placed[:, 0] += center_x
    placed[:, 1] += center_y
    return placed


# ============================================================
# 3. INTERSECTION TESTING
# ============================================================

def classify_loop(loop, a=1.0, A=0.2, lam=2.0):
    """
    Classify whether a loop intersects:
    - bottom plate
    - top plate
    - both

    For the bottom plate at y=0:
        hit if any point has y <= 0

    For the top corrugated plate:
        hit if any point has y >= y_top(x)

    Returns
    -------
    hit_bottom : bool
    hit_top : bool
    hit_both : bool
    """
    x = loop[:, 0]
    y = loop[:, 1]

    hit_bottom = np.any(y <= y_bottom(x))
    hit_top = np.any(y >= y_top(x, a=a, A=A, lam=lam))
    hit_both = hit_bottom and hit_top

    return hit_bottom, hit_top, hit_both


# ============================================================
# 4. MONTE CARLO ENERGY PROXY
# ============================================================

def estimate_energy_proxy(
    n_loops=2000,
    n_points=200,
    loop_scale=0.3,
    x_domain=(-4, 4),
    y_domain=(-0.5, 2.5),
    a=1.0,
    A=0.2,
    lam=2.0,
    seed=0,
    store_examples=False,
    max_examples=25
):
    """
    Estimate a simple Casimir interaction energy proxy.

    We randomly place closed loops in space.
    A loop contributes to the interaction if it intersects BOTH surfaces.

    Energy proxy:
        E_proxy = - (fraction of loops that hit both surfaces)

    This is not yet a fully normalized Casimir energy, but it is a good
    first numerical quantity for studying trends with geometry.

    Parameters
    ----------
    n_loops : int
        Number of Monte Carlo loops to sample.
    n_points : int
        Number of points per loop.
    loop_scale : float
        Typical size of each loop.
    x_domain : tuple
        Range of x positions where loop centers are placed.
    y_domain : tuple
        Range of y positions where loop centers are placed.
    a : float
        Mean separation of the corrugated top surface.
    A : float
        Corrugation amplitude.
    lam : float
        Corrugation wavelength.
    seed : int
        Random seed for reproducibility.
    store_examples : bool
        If True, store a few sample loops from each category.
    max_examples : int
        Maximum number of example loops to store per category.

    Returns
    -------
    energy_proxy : float
        Estimated interaction energy proxy.
    stats : dict
        Dictionary containing counts, fractions, uncertainty estimate,
        and optionally sample loops.
    """
    rng = np.random.default_rng(seed)

    count_none = 0
    count_bottom_only = 0
    count_top_only = 0
    count_both = 0

    example_loops = {
        "none": [],
        "bottom_only": [],
        "top_only": [],
        "both": []
    }

    for _ in range(n_loops):
        loop = generate_closed_loop(
            n_points=n_points,
            loop_scale=loop_scale,
            rng=rng
        )

        cx = rng.uniform(*x_domain)
        cy = rng.uniform(*y_domain)
        loop = place_loop(loop, cx, cy)

        hit_bottom, hit_top, hit_both = classify_loop(loop, a=a, A=A, lam=lam)

        if hit_both:
            count_both += 1
            if store_examples and len(example_loops["both"]) < max_examples:
                example_loops["both"].append(loop.copy())

        elif hit_bottom:
            count_bottom_only += 1
            if store_examples and len(example_loops["bottom_only"]) < max_examples:
                example_loops["bottom_only"].append(loop.copy())

        elif hit_top:
            count_top_only += 1
            if store_examples and len(example_loops["top_only"]) < max_examples:
                example_loops["top_only"].append(loop.copy())

        else:
            count_none += 1
            if store_examples and len(example_loops["none"]) < max_examples:
                example_loops["none"].append(loop.copy())

    frac_none = count_none / n_loops
    frac_bottom_only = count_bottom_only / n_loops
    frac_top_only = count_top_only / n_loops
    frac_both = count_both / n_loops

    # Simple Monte Carlo energy proxy
    energy_proxy = -frac_both

    # Binomial standard error for frac_both
    sigma_frac_both = np.sqrt(frac_both * (1 - frac_both) / n_loops)
    sigma_energy = sigma_frac_both

    stats = {
        "count_none": count_none,
        "count_bottom_only": count_bottom_only,
        "count_top_only": count_top_only,
        "count_both": count_both,
        "frac_none": frac_none,
        "frac_bottom_only": frac_bottom_only,
        "frac_top_only": frac_top_only,
        "frac_both": frac_both,
        "sigma_frac_both": sigma_frac_both,
        "sigma_energy": sigma_energy,
    }

    if store_examples:
        stats["example_loops"] = example_loops

    return energy_proxy, stats

# ============================================================
# 5. VISUALIZATION OF GEOMETRY + SAMPLE LOOPS
# ============================================================

def plot_geometry_and_loops(
    n_show=60,
    n_points=200,
    loop_scale=0.3,
    x_domain=(-4, 4),
    y_domain=(-0.5, 2.5),
    a=1.0,
    A=0.2,
    lam=2.0,
    seed=1
):
    """
    Plot the geometry and sample loops.
    Loops are colored by whether they hit:
    - neither surface
    - one surface
    - both surfaces
    """
    rng = np.random.default_rng(seed)

    x_plot = np.linspace(x_domain[0], x_domain[1], 1000)
    yb = y_bottom(x_plot)
    yt = y_top(x_plot, a=a, A=A, lam=lam)

    plt.figure(figsize=(10, 6))

    # Surfaces
    plt.plot(x_plot, yb, linewidth=2, label="Bottom plate")
    plt.plot(x_plot, yt, linewidth=2, label="Top corrugated plate")

    for _ in range(n_show):
        loop = generate_closed_loop(n_points=n_points, loop_scale=loop_scale, rng=rng)
        cx = rng.uniform(*x_domain)
        cy = rng.uniform(*y_domain)
        loop = place_loop(loop, cx, cy)

        hit_bottom, hit_top, hit_both = classify_loop(loop, a=a, A=A, lam=lam)

        if hit_both:
            alpha = 1
            lw = 1.5
            zorder = 3
        elif hit_bottom or hit_top:
            alpha = 0.5
            lw = 1.0
            zorder = 2
        else:
            alpha = 0.1
            lw = 0.8
            zorder = 1

        plt.plot(loop[:, 0], loop[:, 1], alpha=alpha, linewidth=lw, zorder=zorder)

    plt.xlim(x_domain)
    plt.ylim(y_domain)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Corrugated Geometry with Sample Worldlines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# 6. ENERGY VS SEPARATION
# ============================================================

def sweep_separation(
    a_values,
    n_loops=3000,
    n_points=200,
    loop_scale=0.3,
    x_domain=(-4, 4),
    y_domain=(-0.5, 3.0),
    A=0.2,
    lam=2.0,
    seed=10
):
    """
    Compute energy proxy for multiple separations.
    """
    energies = []

    for i, a in enumerate(a_values):
        energy, stats = estimate_energy_proxy(
            n_loops=n_loops,
            n_points=n_points,
            loop_scale=loop_scale,
            x_domain=x_domain,
            y_domain=y_domain,
            a=a,
            A=A,
            lam=lam,
            seed=seed + i
        )
        energies.append(energy)
        print(f"a = {a:.3f}, E_proxy = {energy:.5f}, frac_both = {stats['frac_both']:.5f}")

    return np.array(energies)


def plot_energy_vs_separation(a_values, energies):
    """
    Plot energy proxy as a function of separation.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(a_values, energies, marker='o')
    plt.xlabel("Mean separation a")
    plt.ylabel("Energy proxy")
    plt.title("Energy Proxy vs Separation")
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# 7. MAIN DEMO
# ============================================================

if __name__ == "__main__":
    # Geometry parameters
    A = 0.25      # corrugation amplitude
    lam = 2.5     # corrugation wavelength
    a = 1.0       # mean separation

    # Show sample loops and geometry
    plot_geometry_and_loops(
        n_show=80,
        n_points=250,
        loop_scale=0.28,
        x_domain=(-4, 4),
        y_domain=(-0.6, 2.4),
        a=a,
        A=A,
        lam=lam,
        seed=2
    )

    # Sweep separation
    a_values = np.linspace(0.6, 1.8, 8)
    energies = sweep_separation(
        a_values=a_values,
        n_loops=2500,
        n_points=250,
        loop_scale=0.28,
        x_domain=(-4, 4),
        y_domain=(-0.6, 3.0),
        A=A,
        lam=lam,
        seed=42
    )

    plot_energy_vs_separation(a_values, energies)