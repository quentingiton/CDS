#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moving-agents SIR toy model (self-contained, only numpy+matplotlib).
- Blue  : susceptible
- Red   : infected
- Green : recovered (immune)

Mechanics
---------
- N agents move in a square [0,1]x[0,1] with reflecting walls.
- One initial infected agent; others are susceptible.
- Infection happens on close contact (distance < radius) with probability beta per time-step.
- Each infected agent recovers after a personal infection_duration (drawn i.i.d.)
- Recovered agents do not get infected again.

How to use
----------
$ python moving_sir.py
This will open a matplotlib window and also save a GIF next to the script if Pillow is installed.

You can tweak parameters in the __main__ block.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

SUS, INF, REC = 0, 1, 2

class MovingSIR:
    def __init__(self,
                 N=150,
                 steps=600,
                 dt=1.0,
                 speed=0.006,
                 infection_radius=0.03,
                 beta=0.25,  # per step, on contact
                 infection_duration_range=(120, 220),  # in steps (min, max) per agent
                 seed=42):
        rng = np.random.default_rng(seed)
        self.N = N
        self.steps = steps
        self.dt = dt
        self.infection_radius = infection_radius
        self.beta = beta
        self.min_dur, self.max_dur = infection_duration_range

        # positions in [0,1]x[0,1]
        self.pos = rng.random((N, 2))
        # random velocities (normalized) times speed
        v = rng.normal(size=(N, 2))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        self.vel = v * speed

        # states & timers
        self.state = np.zeros(N, dtype=int)  # 0=S,1=I,2=R
        patient_zero = rng.integers(0, N)
        self.state[patient_zero] = INF
        # recovery_time = step index when an infected will recover
        self.recovery_time = np.full(N, np.inf)
        self.recovery_time[patient_zero] = rng.integers(self.min_dur, self.max_dur)

        self.rng = rng
        self.t = 0
        self.R0 = self._estimate_R0()

    def _estimate_R0(self):
        avg_contacts = (self.N - 1) * np.pi * (self.infection_radius ** 2)
        avg_infectious_period = (self.min_dur + self.max_dur) / 2
        return avg_contacts * self.beta * avg_infectious_period

    def step(self):
        # Move
        self.pos += self.vel * self.dt

        # Reflect on walls
        for dim in (0, 1):
            too_low = self.pos[:, dim] < 0
            too_high = self.pos[:, dim] > 1
            self.pos[too_low, dim] = -self.pos[too_low, dim]
            self.pos[too_high, dim] = 2 - self.pos[too_high, dim]
            self.vel[too_low | too_high, dim] *= -1

        # Infection
        infected_idx = np.where(self.state == INF)[0]
        susceptible_idx = np.where(self.state == SUS)[0]
        if infected_idx.size > 0 and susceptible_idx.size > 0:
            inf_pos = self.pos[infected_idx]
            sus_pos = self.pos[susceptible_idx]
            # pairwise distances (I x S)
            d2 = np.sum((inf_pos[:, None, :] - sus_pos[None, :, :])**2, axis=2)
            close = d2 < (self.infection_radius ** 2)  # bool matrix
            # For each susceptible, count number of infected neighbors in radius
            k = close.sum(axis=0)
            # Infection probability that at least one transmission occurs
            p_infect = 1 - (1 - self.beta)**k
            # Draw
            u = self.rng.random(size=p_infect.shape[0])
            newly_infected_mask = (u < p_infect) & (k > 0)
            newly_infected_global = susceptible_idx[newly_infected_mask]
            # Set recovery times for new infections
            self.state[newly_infected_global] = INF
            rec_durs = self.rng.integers(self.min_dur, self.max_dur, size=newly_infected_global.size)
            self.recovery_time[newly_infected_global] = self.t + rec_durs

        # Recovery
        to_recover = np.where((self.state == INF) & (self.t >= self.recovery_time))[0]
        if to_recover.size > 0:
            self.state[to_recover] = REC

        self.t += 1

    def run(self, save_mp4_path=None, save_gif_path=None, show=True, interval=30, movie_duration_sec=None):
        fig, ax = plt.subplots(figsize=(6,6), dpi=200)
        scat = ax.scatter(self.pos[:,0], self.pos[:,1], s=25, c=self._colors(), edgecolors='none')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        title = ax.set_title(self._title())

        def update(frame):
            self.step()
            scat.set_offsets(self.pos)
            scat.set_facecolors(self._colors())
            title.set_text(self._title())
            return scat, title
        
        # Calculate fps and steps if movie_duration_sec is given
        if movie_duration_sec is not None:
            fps = max(1, int(1000/interval))
            steps = self.steps
            if movie_duration_sec is not None:
                fps = int(steps / movie_duration_sec)
                interval = int(1000 / fps)

        anim = FuncAnimation(fig, update, frames=self.steps, interval=interval, blit=False)

        # Save GIF if requested
        if save_gif_path is not None:
            try:
                writer = PillowWriter(fps=max(1, int(1000/interval)))
                anim.save(save_gif_path, writer=writer)
                print(f"Saved GIF to {save_gif_path}")
            except Exception as e:
                print("Could not save GIF:", e)

        # Save MP4 if requested
        if save_mp4_path is not None:
            try:
                writer = FFMpegWriter(fps=fps, bitrate=8000)
                anim.save(save_mp4_path, writer=writer)
                print(f"Saved MP4 to {save_mp4_path}")
            except Exception as e:
                print("Could not save MP4:", e)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _colors(self):
        # blue for S, red for I, green for R
        cols = np.empty((self.N, 4))
        cols[self.state == 0] = (0.2, 0.4, 1.0, 0.85)
        cols[self.state == 1] = (1.0, 0.2, 0.2, 0.85)
        cols[self.state == 2] = (0.2, 0.7, 0.2, 0.85)
        return cols

    def _title(self):
        s = np.sum(self.state == 0)
        i = np.sum(self.state == 1)
        r = np.sum(self.state == 2)
        N = self.N
        prop_s = s / N
        prop_i = i / N
        prop_r = r / N
        return (f"t={self.t}   S={prop_s:.2%}   I={prop_i:.2%}   R={prop_r:.2%}   $R_0$={self.R0:.2f}")


if __name__ == "__main__":
    sim = MovingSIR(
        N=160,
        steps=1000,
        dt=1.0,
        speed=0.006,
        infection_radius=0.035,
        beta=0.015,
        infection_duration_range=(220, 360),
        seed=2214
    )
    # Save a short GIF and also show the animation window
    sim.run(save_mp4_path="sir_moving_agents.mp4", show=False, interval=30, movie_duration_sec=20)