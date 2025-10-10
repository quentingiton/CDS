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
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from collections import deque

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
    cmap = ListedColormap([[0,0.6,0], [0,0.6,0], [1,0,0]])
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
    cmap = ListedColormap([[0,0.6,0], [0,0.6,0], [1,0,0]])
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


def sample_bond_grid(N, p, rng):
    # Bonds: True=open, False=closed
    # Horizontal bonds: between (i, j) and (i, j+1)
    h_bonds = rng.random((N, N-1)) < p
    # Vertical bonds: between (i, j) and (i+1, j)
    v_bonds = rng.random((N-1, N)) < p
    return h_bonds, v_bonds


def coords_to_mask(N, coords):
    mask = np.zeros((N, N), dtype=bool)
    for (i, j) in coords:
        if 0 <= i < N and 0 <= j < N:
            mask[i, j] = True
    return mask

def rect_mask(N, top, left, height, width):
    mask = np.zeros((N, N), dtype=bool)
    i1 = max(0, top); i2 = min(N, top + height)
    j1 = max(0, left); j2 = min(N, left + width)
    mask[i1:i2, j1:j2] = True
    return mask

def manhattan_ball_mask(N, center, radius):
    ci, cj = center
    ii, jj = np.ogrid[:N, :N]
    return (np.abs(ii - ci) + np.abs(jj - cj)) <= radius

def union_masks(*masks):
    out = None
    for m in masks:
        out = m if out is None else (out | m)
    return out if out is not None else None



def burned_trees(N, h_bonds, v_bonds, ignition="left", rng=None):
    """
    ignition can be:
      - "random", "left", "top" (as before)
      - list/tuple of (i, j) coordinates
      - a boolean mask of shape (N, N)
      - a callable f(i, j) -> bool
      - a dict shape spec:
          {"shape":"rect", "top":..., "left":..., "height":..., "width":...}
          {"shape":"ball", "center":(i0, j0), "radius":R}   (Manhattan ball)
    Returns: burned (bool[N,N]), ignition_pts (list[(i,j)])
    """
    if rng is None:
        rng = np.random.default_rng()

    burned = np.zeros((N, N), dtype=bool)
    q = deque()
    ignition_pts = []

    def ignite_cell(i, j):
        if 0 <= i < N and 0 <= j < N and not burned[i, j]:
            burned[i, j] = True
            q.append((i, j))
            ignition_pts.append((i, j))

    # --- build ignition set (type-checked to avoid ndarray == "random") ---
    if isinstance(ignition, str):
        if ignition == "random":
            ignite_cell(rng.integers(0, N), rng.integers(0, N))
        elif ignition == "left":
            for i in range(N):
                ignite_cell(i, 0)
        elif ignition == "top":
            for j in range(N):
                ignite_cell(0, j)
        else:
            raise ValueError(f"Unknown ignition keyword: {ignition}")

    elif isinstance(ignition, np.ndarray):
        if ignition.shape != (N, N) or ignition.dtype != bool:
            raise ValueError("Boolean ignition mask must have shape (N,N).")
        is_, js_ = np.where(ignition)
        for i0, j0 in zip(is_, js_):
            ignite_cell(i0, j0)

    elif isinstance(ignition, (list, tuple)) and len(ignition) > 0 and isinstance(ignition[0], (list, tuple)):
        # list of (i, j) coordinates
        for (i0, j0) in ignition:
            ignite_cell(i0, j0)

    elif isinstance(ignition, dict):
        shape = ignition.get("shape")
        if shape == "rect":
            m = rect_mask(N, top=ignition["top"], left=ignition["left"],
                          height=ignition["height"], width=ignition["width"])
        elif shape == "ball":
            m = manhattan_ball_mask(N, center=ignition["center"], radius=ignition["radius"])
        else:
            raise ValueError(f"Unknown ignition shape: {shape}")
        is_, js_ = np.where(m)
        for i0, j0 in zip(is_, js_):
            ignite_cell(i0, j0)

    elif callable(ignition):
        for i in range(N):
            for j in range(N):
                if ignition(i, j):
                    ignite_cell(i, j)

    else:
        raise ValueError("Unsupported ignition type.")

    # --- BFS spread on open bonds ---
    while q:
        i, j = q.popleft()
        # S
        if i < N-1 and v_bonds[i, j] and not burned[i+1, j]:
            burned[i+1, j] = True; q.append((i+1, j))
        # N
        if i > 0 and v_bonds[i-1, j] and not burned[i-1, j]:
            burned[i-1, j] = True; q.append((i-1, j))
        # E
        if j < N-1 and h_bonds[i, j] and not burned[i, j+1]:
            burned[i, j+1] = True; q.append((i, j+1))
        # W
        if j > 0 and h_bonds[i, j-1] and not burned[i, j-1]:
            burned[i, j-1] = True; q.append((i, j-1))
    return burned, ignition_pts

# To estimate critical p:
def estimate_bond_pc(N, trials, p_values, seed=0):
    rng = np.random.default_rng(seed)
    burned_props = []
    for p in p_values:
        fracs = []
        for _ in range(trials):
            h_bonds, v_bonds = sample_bond_grid(N, p, rng)
            burned, _ = burned_trees(N, h_bonds, v_bonds, rng=rng)  # or pass ignition=...
            fracs.append(burned.mean())  # scalar fraction burned
        burned_props.append(np.mean(fracs))
    return p_values, burned_props

def estimate_bond_curve(N=70, trials=60, p_values=None, seed=0):
    if p_values is None:
        p_values = np.linspace(0.3, 0.7, 15)
    rng = np.random.default_rng(seed)
    burned_props = []
    for p in p_values:
        fracs = []
        for _ in range(trials):
            h_bonds, v_bonds = sample_bond_grid(N, p, rng)
            burned, _ = burned_trees(N, h_bonds, v_bonds, rng=rng)  # or pass ignition=...
            fracs.append(burned.mean())  # scalar fraction burned
        burned_props.append(np.mean(fracs))
    return p_values, np.array(burned_props)

def plot_bond_curve(p_values, burned_props, save_path="bond_percolation_curve.png"):
    plt.figure(figsize=(6,4))
    plt.plot(p_values, burned_props, marker='o', color='firebrick')
    plt.xlabel("p (probabilité qu'une liaison soit ouverte)")
    plt.ylabel("Proportion d'arbres brûlés")
    plt.title("Percolation de liaison sur grille carrée (N fixe)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def random_blob_mask(N, rng, min_size=20, max_size=200, p_grow=0.6, anchor=None):
    """
    Make a connected random blob (boolean mask) by growing from one seed.
    anchor: None | "left" | "top"  (chooses the seed on that boundary)
    """
    target = int(rng.integers(min_size, max_size + 1))
    mask = np.zeros((N, N), dtype=bool)

    # choose seed
    if anchor == "left":
        i0, j0 = int(rng.integers(0, N)), 0
    elif anchor == "top":
        i0, j0 = 0, int(rng.integers(0, N))
    else:
        i0, j0 = int(rng.integers(0, N)), int(rng.integers(0, N))

    q = deque([(i0, j0)])
    mask[i0, j0] = True

    # grow the blob
    while q and mask.sum() < target:
        i, j = q.popleft()
        for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
            i2, j2 = i + di, j + dj
            if 0 <= i2 < N and 0 <= j2 < N and not mask[i2, j2]:
                if rng.random() < p_grow:
                    mask[i2, j2] = True
                    q.append((i2, j2))

    # if growth stalled early, thicken a bit to reach min_size
    while mask.sum() < min_size:
        is_, js_ = np.where(mask)
        if len(is_) == 0:
            break
        k = int(rng.integers(0, len(is_)))
        i, j = is_[k], js_[k]
        for di, dj in ((1,0), (-1,0), (0,1), (0,-1)):
            i2, j2 = i + di, j + dj
            if 0 <= i2 < N and 0 <= j2 < N and not mask[i2, j2]:
                mask[i2, j2] = True
                break
    return mask



def plot_bond_grid_square(N, p, seeds, save_path,
                          ignition="left",
                          ignition_factory=None,
                          show_ignition=False,      # keeps your scatter option if you like
                          ign_scatter_every=15):
    imgs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        h_bonds, v_bonds = sample_bond_grid(N, p, rng)

        ign_spec = ignition_factory(N, rng) if callable(ignition_factory) else ignition
        burned_mask, ignition_pts = burned_trees(N, h_bonds, v_bonds, ignition=ign_spec, rng=rng)

        # Build a 0/1/2 image:
        # 0 = green (unburned), 1 = orange (burned), 2 = red (ignition)
        img = burned_mask.astype(np.int8)
        if ignition_pts:                      # ignition_pts is a list of (i,j)
            ii, jj = zip(*ignition_pts)       # rows, cols
            img[list(ii), list(jj)] = 2

        imgs.append((img, burned_mask.sum(), ignition_pts))

    cmap = ListedColormap([[0,0.6,0], [1,0.5,0], [1,0,0]])  # green, orange, red
    fig, axes = plt.subplots(4, 4, figsize=(16,16))
    for ax, (img, burned, ignition_pts), seed in zip(axes.flat, imgs, seeds):
        ax.imshow(img, cmap=cmap, interpolation='nearest', origin='upper')
        ax.set_xticks([]); ax.set_yticks([])

        if show_ignition and ignition_pts:    # optional extra markers, not required
            pts = ignition_pts[::max(1, ign_scatter_every)]
            ys, xs = zip(*pts)
            ax.scatter(xs, ys, s=10, c='red', marker='s')

        ax.set_title(f"seed={seed}\nBrûlés: {burned} ({burned/(N*N):.1%})")

    for ax in axes.flat[len(imgs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()




if __name__ == "__main__":
    # 1) Monte Carlo: probability curve
    p_vals, probs = estimate_curve(N=70, trials=60, p_values=np.linspace(0.3, 0.7, 15), seed=2214)
    plot_curve(p_vals, probs, save_path="11_10_25/medias/images/percolation_curve.png")

    # Estimate p_c as the p where probability is closest to 0.5
    idx_pc = np.argmin(np.abs(probs - 0.5))
    p_c = p_vals[idx_pc]

    # 3) Several examples below, at and above threshold
    seeds = list(range(40, 56))
    plot_grid_square(N=70, p=p_c-0.05, seeds=seeds, save_path="11_10_25/medias/images/percolation_below_pc_square_16.png")
    plot_grid_square(N=70, p=p_c, seeds=seeds, save_path="11_10_25/medias/images/percolation_at_pc_square_16.png")
    plot_grid_square(N=70, p=p_c+0.05, seeds=seeds, save_path="11_10_25/medias/images/percolation_above_pc_square_16.png")

    # Bond percolation analysis
    p_vals_bond, burned_props = estimate_bond_curve(N=70, trials=200, p_values=np.linspace(00, 1, 100), seed=2214)
    plot_bond_curve(p_vals_bond, burned_props, save_path="11_10_25/medias/images/bond_percolation_curve.png")

    # Estimate p_c for bond percolation (where >50% trees burn)
    idx_pc_bond = np.argmax(burned_props > 0.5)
    p_c_bond = p_vals_bond[idx_pc_bond]

    factory = lambda N, rng: random_blob_mask(
    N, rng, min_size=20, max_size=250, p_grow=0.65, anchor=None
    )

    seeds = list(range(40, 56))
    plot_bond_grid_square(N=70, p=p_c_bond-0.05, seeds=seeds, save_path="11_10_25/medias/images/bond_below_pc_square_16.png", ignition_factory=factory, show_ignition=False)
    plot_bond_grid_square(N=70, p=p_c_bond, seeds=seeds, save_path="11_10_25/medias/images/bond_at_pc_square_16.png", ignition_factory=factory, show_ignition=False)
    plot_bond_grid_square(N=70, p=p_c_bond+0.05, seeds=seeds, save_path="11_10_25/medias/images/bond_above_pc_square_16.png", ignition_factory=factory, show_ignition=False)