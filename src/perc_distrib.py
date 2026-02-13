import igraph as ig
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
from typing import Callable, Tuple

rng = np.random.default_rng()


def simulate_gauss(
    graph: ig.Graph,
    window: float,
    low_p: float = 0.4,
    high_p: float = 0.6,
    n_probs: int = 100,
    copies: int = 1000,
    no_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    sigma = window / 3
    n_edges = graph.ecount()
    lcc = np.zeros(n_probs, dtype="float32")
    all_prob = np.linspace(max(low_p, window), min(high_p, 1 - window), n_probs)
    for idx, p in tqdm(enumerate(all_prob), total=n_probs, disable=no_bar):
        is_removed = rng.random((copies, n_edges))
        link_prob = rng.normal(p, sigma, (copies, n_edges))
        links_removed = is_removed >= link_prob
        for rep in range(copies):
            links_to_remove = np.flatnonzero(links_removed[rep])
            work = graph.copy()
            work.delete_edges(links_to_remove)
            lcc[idx] += max(work.components().sizes())
        lcc[idx] /= copies
    return all_prob, lcc / graph.vcount()


def simulate_bimodal(
    graph: ig.Graph,
    window: float,
    low_p: float = 0.4,
    high_p: float = 0.6,
    n_probs: int = 100,
    copies: int = 1000,
    no_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    n_edges = graph.ecount()
    lcc = np.zeros(n_probs, dtype="float32")
    all_prob = np.linspace(max(low_p, window), min(high_p, 1 - window), n_probs)
    for idx, p in tqdm(enumerate(all_prob), total=n_probs, disable=no_bar):
        is_removed = rng.random((copies, n_edges))
        link_prob = rng.choice([p - window, p + window], (copies, n_edges))
        # link_prob = rng.choice([p - window], (copies, n_edges))
        links_removed = is_removed >= link_prob
        for rep in range(copies):
            links_to_remove = np.flatnonzero(links_removed[rep])
            work = graph.copy()
            work.delete_edges(links_to_remove)
            lcc[idx] += max(work.components().sizes())
        lcc[idx] /= copies
    return all_prob, lcc / graph.vcount()


def simulate_bimodal_weight(
    graph: ig.Graph,
    window: float,
    low_p: float = 0.4,
    high_p: float = 0.6,
    n_probs: int = 100,
    copies: int = 1000,
    no_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    n_edges = graph.ecount()
    lcc = np.zeros(n_probs, dtype="float32")
    all_prob = np.linspace(max(low_p, window), min(high_p, 1 - window), n_probs)
    for idx, p in tqdm(enumerate(all_prob), total=n_probs, disable=no_bar):
        is_removed = rng.random((copies, n_edges))
        link_prob = rng.choice(
            [p - window, p + window], (copies, n_edges), p=(2 / 3, 1 / 3)
        )
        links_removed = is_removed >= link_prob
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
    func: Callable,
    name: str | None = None,
    n_probs: int = 100,
    copies: int = 100,
) -> np.ndarray:
    parallel = partial(
        func,
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
    if name:
        np.savez(name, windows=windows, prob=prob, lcc=lcc)
    return results


if __name__ == "__main__":
    windows = np.linspace(0, 0.2, 10)
    g = ig.Graph.Lattice([100, 100], circular=False)
    res = run_parallel(
        g, windows, simulate_bimodal_weight, "bimodal_skew", n_probs=100, copies=200
    )
