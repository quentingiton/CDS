#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Percolation threshold demo (site percolation on a square grid with 4-neighborhood).
Self-contained: numpy + matplotlib only.

Features
--------
1) Monte Carlo estimate of percolation probability as a function of p (open-site probability).
2) Visualization of a single grid and the "connected-to-top" open cluster;
   If this cluster touches the bottom, the system percolates.

Usage
-----
$ python percolation_demo.py
This will produce two figures (saved as PNGs):
- percolation_curve.png
- percolation_example.png

You can tune N, trials, and p_values in the __main__ block.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class UnionFind:
    parent: np.ndarray
    rank: np.ndarray

    @classmethod
    def make(cls, n):
        parent = np.arange(n, dtype=int)
        rank = np.zeros(n, dtype=int)
        return cls(parent, rank)

    def find(self, x):
        # path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def grid_index(i, j, N):
    return i * N + j

def sample_grid(N, p, rng):
    # True=open; False=closed
    return rng.random((N, N)) < p

def percolates_top_to_bottom(open_grid):
    """Check percolation using Union-Find with virtual top & bottom nodes."""
    N = open_grid.shape[0]
    uf = UnionFind.make(N*N + 2)
    TOP = N*N
    BOT = N*N + 1

    # Union neighbors
    for i in range(N):
        for j in range(N):
            if not open_grid[i, j]:
                continue
            idx = grid_index(i, j, N)
            # connect to virtual nodes if on borders
            if i == 0:
                uf.union(idx, TOP)
            if i == N-1:
                uf.union(idx, BOT)
            # 4-neighborhood
            if i > 0 and open_grid[i-1, j]:
                uf.union(idx, grid_index(i-1, j, N))
            if i < N-1 and open_grid[i+1, j]:
                uf.union(idx, grid_index(i+1, j, N))
            if j > 0 and open_grid[i, j-1]:
                uf.union(idx, grid_index(i, j-1, N))
            if j < N-1 and open_grid[i, j+1]:
                uf.union(idx, grid_index(i, j+1, N))

    return uf.find(TOP) == uf.find(BOT)

def connected_to_top(open_grid):
    """Return a boolean mask of sites that are both open and connected to the top row (BFS)."""
    N = open_grid.shape[0]
    visited = np.zeros_like(open_grid, dtype=bool)
    from collections import deque
    q = deque()

    # seed with open sites on the top row
    for j in range(N):
        if open_grid[0, j]:
            visited[0, j] = True
            q.append((0, j))

    # BFS
    while q:
        i, j = q.popleft()
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            i2, j2 = i+di, j+dj
            if 0 <= i2 < N and 0 <= j2 < N and not visited[i2, j2] and open_grid[i2, j2]:
                visited[i2, j2] = True
                q.append((i2, j2))

    return visited

def estimate_curve(N=60, trials=60, p_values=None, seed=0):
    if p_values is None:
        p_values = np.linspace(0.4, 0.7, 13)
    rng = np.random.default_rng(seed)
    probs = []
    for p in p_values:
        count = 0
        for _ in range(trials):
            g = sample_grid(N, p, rng)
            if percolates_top_to_bottom(g):
                count += 1
        probs.append(count / trials)
    return p_values, np.array(probs)

def plot_curve(p_values, probs, save_path="percolation_curve.png"):
    plt.figure(figsize=(6,4))
    plt.plot(p_values, probs, marker='o')
    plt.xlabel("p (probabilité qu'une case soit ouverte)")
    plt.ylabel("Probabilité de percolation (haut→bas)")
    plt.title("Percolation de site sur grille carrée (N fixe)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_example(N=60, p=0.59, seed=1, save_path="percolation_example.png"):
    rng = np.random.default_rng(seed)
    g = sample_grid(N, p, rng)
    top_conn = connected_to_top(g)
    does_perc = np.any(top_conn[-1, :])

    # 0=closed, 1=open-not-connected-to-top, 2=connected-to-top
    img = np.zeros((N, N), dtype=int)
    img[g] = 1
    img[top_conn] = 2

    plt.figure(figsize=(5,5))
    # custom colormap-like via listed colors
    # 0: closed (white), 1: open (light gray), 2: connected-to-top (darker tone)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([[1,1,1],[0.8,0.8,0.8],[0.4,0.6,1.0]])
    plt.imshow(img, cmap=cmap, interpolation='nearest', origin='upper')
    plt.xticks([]); plt.yticks([])
    title = f"Exemple à p={p:.2f} — {'PERCOLATION !' if does_perc else 'pas de percolation'}"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_multiple_examples(N, p_list, seeds, save_path_prefix):
    for i, (p, seed) in enumerate(zip(p_list, seeds)):
        plot_example(N=N, p=p, seed=seed, save_path=f"{save_path_prefix}_p{p:.3f}_ex{i+1}.png")

def plot_grid_square(N, p, seeds, save_path):
    """Plot several grids for p=p_c in a 4x4 square."""
    imgs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        g = sample_grid(N, p, rng)
        top_conn = connected_to_top(g)
        does_perc = np.any(top_conn[-1, :])
        img = np.zeros((N, N), dtype=int)
        img[g] = 1
        img[top_conn] = 2
        imgs.append((img, does_perc))
    # Plot 4x4 square
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([[1,1,1],[0.8,0.8,0.8],[0.4,0.6,1.0]])
    fig, axes = plt.subplots(4, 4, figsize=(16,16))
    for ax, (img, does_perc), seed in zip(axes.flat, imgs, seeds):
        ax.imshow(img, cmap=cmap, interpolation='nearest', origin='upper')
        ax.set_xticks([]); ax.set_yticks([])
        title = f"seed={seed}\n{'PERCOLATION!' if does_perc else 'no percolation'}"
        ax.set_title(title)
    # Hide unused axes if seeds < 16
    for ax in axes.flat[len(imgs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



if __name__ == "__main__":
    # 1) Monte Carlo: probability curve
    p_vals, probs = estimate_curve(N=60, trials=60, p_values=np.linspace(0.45, 0.7, 11), seed=2214)
    plot_curve(p_vals, probs, save_path="percolation_curve.png")

    # Estimate p_c as the p where probability is closest to 0.5
    idx_pc = np.argmin(np.abs(probs - 0.5))
    p_c = p_vals[idx_pc]

    # 3) Several examples below, at and above threshold
    seeds = list(range(40, 56))
    plot_grid_square(N=70, p=p_c-0.05, seeds=seeds, save_path="percolation_below_pc_square_16.png")
    plot_grid_square(N=70, p=p_c, seeds=seeds, save_path="percolation_at_pc_square_16.png")
    plot_grid_square(N=70, p=p_c+0.05, seeds=seeds, save_path="percolation_above_pc_square_16.png")
