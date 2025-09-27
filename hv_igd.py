import json
import numpy as np
from scipy.spatial import distance

def b_dominates_a(a, b):
    """Return True if solution b dominates solution a (minimization)."""
    return all(b_i <= a_i for a_i, b_i in zip(a, b)) and any(b_i < a_i for a_i, b_i in zip(a, b))

def get_non_dominated(pop):
    """Extract non-dominated solutions from population JSON."""
    objs = [(p["gap"], p["runtime"]) 
            for p in pop 
            if np.isfinite(p["gap"]) and np.isfinite(p["runtime"])]
    objs = np.array(objs, dtype=float)
    non_dominated = []
    for i, a in enumerate(objs):
        dominated = False
        for j, b in enumerate(objs):
            if i != j and b_dominates_a(a, b):
                dominated = True
                break
        if not dominated:
            non_dominated.append(a)
    return np.array(non_dominated, dtype=float)

def hypervolume_raw(front, ref_point):
    """Compute hypervolume for 2D minimization (raw values)."""
    # Sort by first objective ascending
    front = front[np.argsort(front[:,0])]
    hv = 0.0
    prev_f1 = ref_point[0]
    # Iterate from right to left to accumulate rectangles
    for f1, f2 in reversed(front):
        width = prev_f1 - f1
        height = ref_point[1] - f2
        if width > 0 and height > 0:
            hv += width * height
        prev_f1 = f1
    return hv

def igd(approx_front, ref_front):
    """Compute IGD: average distance from each ref point to nearest approx point."""
    dists = []
    for r in ref_front:
        d = min(distance.euclidean(r, a) for a in approx_front)
        dists.append(d)
    return np.mean(dists)

if __name__ == "__main__":
    # Load population from JSON file
    files =["outputs/main/reevo_bpp_online_2025-09-26_22-32-25/population_iter11.json",
            "outputs/main/reevo_bpp_online_2025-09-26_22-24-50/population_iter11.json",
            "outputs/main/reevo_bpp_online_2025-09-26_22-16-57/population_iter11.json",
            "outputs/main/map-elites_bpp_online_2025-09-27_10-27-02/population_iter9.json",
            "outputs/main/map-elites_bpp_online_2025-09-27_10-18-05/population_iter9.json",
            "outputs/main/map-elites_bpp_online_2025-09-27_10-10-50/population_iter9.json"]
    for file in files:
        with open(file) as f:
            pop = json.load(f)

        # Get Pareto front (raw values)
        nd_front_raw = get_non_dominated(pop)

        ref_point = np.array([150, 50])    # runtime max = 50

        # Metrics
        hv = hypervolume_raw(nd_front_raw, ref_point)
        # If you have a true reference front, you can compute IGD:
        # igd_value = igd(nd_front_raw, true_ref_front)
        igd_value = np.nan  # placeholder

        # Output
        # print("Non-dominated front (raw):\n", nd_front_raw)
        # print("Reference point (raw):", ref_point)
        print("Hypervolume (raw):", hv)
        # print("IGD:", igd_value)
