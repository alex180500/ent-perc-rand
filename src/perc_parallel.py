import igraph as ig
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

rng = np.random.default_rng()


def simulate_uniform(
    graph: ig.Graph,
    window: float,
    n_probs: int = 100,
    copies: int = 1000,
    low_p: float = 0.4,
    high_p: float = 0.6,
    no_bar: bool = False,
):
    n_edges = graph.ecount()
    lcc = np.zeros(n_probs, dtype="float32")
    all_prob = np.linspace(max(low_p, window), min(high_p, 1 - window), n_probs)
    for idx, p in tqdm(enumerate(all_prob), total=n_probs, disable=no_bar):
        is_removed = rng.random((copies, n_edges))
        link_prob = rng.uniform(p - window, p + window, (copies, n_edges))
        links_removed = is_removed > link_prob
        for rep in range(copies):
            links_to_remove = np.flatnonzero(links_removed[rep])
            work = graph.copy()
            work.delete_edges(links_to_remove)
            lcc[idx] += max(work.components().sizes())
        lcc[idx] /= copies
    return all_prob, lcc / graph.vcount()


def run_parallel(
    graph: ig.Graph,
    windows: np.ndarray,
    name: str,
    n_probs: int = 100,
    copies: int = 100,
):
    parallel = partial(
        simulate_uniform,
        graph.copy(),
        low_p=0,
        high_p=1,
        n_probs=n_probs,
        copies=copies,
        no_bar=True,
    )
    with multiprocessing.Pool(processes=10) as pool:
        results = pool.imap(parallel, windows)

        prob, lcc = [], []
        for p, temp in tqdm(results, total=len(windows)):
            prob.append(p)
            lcc.append(temp)
    np.savez(name, windows=windows, prob=prob, lcc=lcc)
    return results


if __name__ == "__main__":
    windows = np.linspace(0, 0.2, 10)
    g = ig.Graph.Lattice([100, 100], circular=False)
    res = run_parallel(g, windows, "uniform_test", n_probs=100, copies=200)
