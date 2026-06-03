"""Benchmark snn_numba against the original C++ SNN and scikit-learn KDTree.

Measures:
  * index build time
  * batch radius-query time (throughput)
  * accuracy (recall / precision) vs KDTree as ground truth

Run:  uv run python bench/run_bench.py
Output: console tables + CSVs and PNG plots under bench/results/.
"""

from __future__ import annotations

import os

# keep matplotlib's cache inside the project (sandbox-friendly)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tabulate import tabulate

from snn_numba import SNN

import common

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_THREADS = os.cpu_count()
snnomp = common.load_cpp()


# --------------------------------------------------------------------------- #
# method wrappers: each returns (build_fn, query_fn) closures over the data
# --------------------------------------------------------------------------- #
def methods_for(X, Q, r):
    """Build callables for every method on a given dataset/query/radius."""
    methods = {}

    # --- snn_numba (float64) ---
    def build_snn64():
        return SNN(X, dtype=np.float64)

    methods["snn_numba(f64)"] = (
        build_snn64,
        lambda m: m.query_radius(Q, r, return_distance=False),
    )

    # --- snn_numba (float32) ---
    def build_snn32():
        return SNN(X.astype(np.float32), dtype=np.float32)

    methods["snn_numba(f32)"] = (
        build_snn32,
        lambda m: m.query_radius(Q.astype(np.float32), np.float32(r)),
    )

    # --- original C++ (float64 / float32) ---
    if snnomp is not None:
        methods["snn_cpp(f64)"] = (
            lambda: snnomp.SNN_DOUBLE(X, N_THREADS),
            lambda m: m.query_radius_batch(Q, r),
        )
        Xf = X.astype(np.float32)
        Qf = Q.astype(np.float32)
        methods["snn_cpp(f32)"] = (
            lambda: snnomp.SNN_FLOAT(Xf, N_THREADS),
            lambda m: m.query_radius_batch(Qf, np.float32(r)),
        )

    # --- scikit-learn KDTree ---
    methods["sklearn_KDTree"] = (
        lambda: KDTree(X),
        lambda m: m.query_radius(Q, r),
    )
    return methods


def warmup():
    """Trigger Numba JIT compilation so it is excluded from timings."""
    X = common.make_data(2000, 8, seed=42)
    Q = common.make_queries(X, 16, seed=43)
    m = SNN(X)
    m.query_radius(Q, 0.5)
    m.query_radius(Q, 0.5, return_distance=True, sort_results=True)
    m.query_radius(Q, 0.5, count_only=True)
    SNN(X.astype(np.float32), dtype=np.float32).query_radius(
        Q.astype(np.float32), np.float32(0.5)
    )


# --------------------------------------------------------------------------- #
# one benchmark point
# --------------------------------------------------------------------------- #
def bench_point(n, d, m_queries, target_nbrs, n_clusters, seed=0,
                build_repeat=2, query_repeat=4):
    X = common.make_data(n, d, seed=seed, n_clusters=n_clusters)
    Q = common.make_queries(X, m_queries, seed=seed + 100)

    tuner = SNN(X)
    r = common.tune_radius(tuner, Q, target_nbrs)
    avg_nbrs = float(tuner.query_radius(Q, r, count_only=True).mean())

    methods = methods_for(X, Q, r)
    rows = []
    built = {}
    for name, (build_fn, query_fn) in methods.items():
        b_t, model = common.timeit(build_fn, repeat=build_repeat)
        built[name] = model
        q_t, _ = common.timeit(lambda: query_fn(model), repeat=query_repeat)
        rows.append(
            {
                "method": name,
                "n": n,
                "d": d,
                "m": m_queries,
                "radius": round(r, 4),
                "avg_nbrs": round(avg_nbrs, 1),
                "build_s": b_t,
                "query_s": q_t,
                "q_per_s": m_queries / q_t,
            }
        )
    return pd.DataFrame(rows), built, X, Q, r


# --------------------------------------------------------------------------- #
# accuracy vs KDTree
# --------------------------------------------------------------------------- #
def accuracy_vs_kdtree(built, X, Q, r):
    """Recall & precision of each SNN variant against KDTree ground truth."""
    kd = KDTree(X)
    truth = [set(a.tolist()) for a in kd.query_radius(Q, r)]

    def get_results(name, model):
        if name.startswith("snn_numba"):
            rr = r if "f64" in name else np.float32(r)
            QQ = Q if "f64" in name else Q.astype(np.float32)
            return [set(a.tolist()) for a in model.query_radius(QQ, rr)]
        if name.startswith("snn_cpp"):
            if "f64" in name:
                res = model.query_radius_batch(Q, r)
            else:
                res = model.query_radius_batch(Q.astype(np.float32), np.float32(r))
            return [set(np.asarray(a).tolist()) for a in res]
        return None

    rows = []
    for name, model in built.items():
        res = get_results(name, model)
        if res is None:
            continue
        tp = fp = fn = 0
        for got, exp in zip(res, truth):
            tp += len(got & exp)
            fp += len(got - exp)
            fn += len(exp - got)
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        exact = sum(1 for got, exp in zip(res, truth) if got == exp)
        rows.append(
            {
                "method": name,
                "recall": recall,
                "precision": precision,
                "exact_queries": f"{exact}/{len(truth)}",
            }
        )
    return pd.DataFrame(rows)


def fmt(df, cols=None):
    d = df.copy()
    # display build/query in milliseconds with explicit headers
    d["build_ms"] = d["build_s"] * 1e3
    d["query_ms"] = d["query_s"] * 1e3
    for c in ("build_ms", "query_ms"):
        d[c] = d[c].map(lambda x: f"{x:.2f}")
    d["q_per_s"] = d["q_per_s"].map(lambda x: f"{x:,.0f}")
    cols = cols or ["method", "build_ms", "query_ms", "q_per_s", "avg_nbrs"]
    cols = [c.replace("build_s", "build_ms").replace("query_s", "query_ms") for c in cols]
    d = d[cols]
    return tabulate(d, headers="keys", tablefmt="github", showindex=False)


# --------------------------------------------------------------------------- #
def main():
    print(f"Threads: {N_THREADS}   C++ module: {'yes' if snnomp else 'NO'}\n")
    print("Warming up Numba JIT...")
    warmup()

    all_speed = []
    all_acc = []

    # ----- main scenario (README-like): n=100k, d=100, clustered -----
    print("\n" + "=" * 78)
    print("MAIN SCENARIO  n=100,000  d=100  m=1,000 queries  (~100 neighbors)")
    print("=" * 78)
    df, built, X, Q, r = bench_point(
        n=100_000, d=100, m_queries=500, target_nbrs=100, n_clusters=20
    )
    print(fmt(df, ["method", "build_s", "query_s", "q_per_s", "avg_nbrs"]))
    print("\nAccuracy vs sklearn KDTree (ground truth):")
    acc = accuracy_vs_kdtree(built, X, Q, r)
    print(tabulate(acc, headers="keys", tablefmt="github", showindex=False))
    all_speed.append(df)
    acc["scenario"] = "main"
    all_acc.append(acc)

    # ----- scaling over n -----
    print("\n" + "=" * 78)
    print("SCALING OVER n   (d=50, m=500, clustered, ~50 neighbors)")
    print("=" * 78)
    for n in (10_000, 50_000, 200_000):
        df, built, X, Q, r = bench_point(
            n=n, d=50, m_queries=500, target_nbrs=50, n_clusters=15
        )
        print(f"\n-- n={n:,} --")
        print(fmt(df, ["method", "build_s", "query_s", "q_per_s", "avg_nbrs"]))
        all_speed.append(df)
        acc = accuracy_vs_kdtree(built, X, Q, r)
        acc["scenario"] = f"n={n}"
        all_acc.append(acc)

    # ----- scaling over d -----
    print("\n" + "=" * 78)
    print("SCALING OVER d   (n=100,000, m=500, clustered, ~50 neighbors)")
    print("=" * 78)
    for d in (2, 10, 25, 200, 500):
        df, built, X, Q, r = bench_point(
            n=100_000, d=d, m_queries=500, target_nbrs=50, n_clusters=15
        )
        print(f"\n-- d={d} --")
        print(fmt(df, ["method", "build_s", "query_s", "q_per_s", "avg_nbrs"]))
        all_speed.append(df)
        acc = accuracy_vs_kdtree(built, X, Q, r)
        acc["scenario"] = f"d={d}"
        all_acc.append(acc)

    speed = pd.concat(all_speed, ignore_index=True)
    acc = pd.concat(all_acc, ignore_index=True)
    speed.to_csv(os.path.join(RESULTS_DIR, "speed.csv"), index=False)
    acc.to_csv(os.path.join(RESULTS_DIR, "accuracy.csv"), index=False)
    print(f"\nSaved CSVs to {RESULTS_DIR}")

    make_plots(speed)


def make_plots(speed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # query throughput vs n (d=50)
    sub = speed[(speed.d == 50)]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, g in sub.groupby("method"):
            g = g.sort_values("n")
            ax.plot(g.n, g.q_per_s, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (dataset size)")
        ax.set_ylabel("queries / second")
        ax.set_title("Radius-query throughput vs n  (d=50)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "throughput_vs_n.png"), dpi=120)

    # query time vs d (n=100k)
    sub = speed[speed.n == 100_000]
    sub = sub[sub.d.isin([2, 10, 25, 100, 200, 500])]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, g in sub.groupby("method"):
            g = g.sort_values("d")
            ax.plot(g.d, g.query_s * 1e3, marker="o", label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("d (dimensionality)")
        ax.set_ylabel("batch query time (ms)")
        ax.set_title("Radius-query time vs d  (n=100,000, m=500)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "querytime_vs_d.png"), dpi=120)

    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
