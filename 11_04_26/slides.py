from manim import *
from manim_slides import Slide
import random, math, numpy as np
from manim.utils.tex_file_writing import TexTemplate

class RareEarths(Slide):
    def construct(self):
        self.wait_time_between_slides = 0.1

        title = VGroup(
            Text("Les Terres Rares : ", t2w={"[-15:-2]": BOLD}, t2c={"[-16:-2]": YELLOW}),
            Text("La malédiction du mélange", t2c={"mélange": BLUE}),
        ).arrange(DOWN)

        self.play(FadeIn(title))

        self.next_slide()

        self.play(FadeOut(title))

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r"\usepackage{xcolor}")

        # params
        k_num = 39
        ell_num = 11
        ball_r = 0.17
        padding = 0.06

        # lines (one-by-one)
        b1 = Tex(r"$k$ : Nombre de billes bleues",
                tex_template=tex_template,
                tex_to_color_map={r"$k$": BLUE, "bleues": BLUE})

        b2 = Tex(r"$\ell$ : Nombre de billes rouges",
                tex_template=tex_template,
                tex_to_color_map={r"$\ell$": RED, "rouges": RED})
        
        b3_math = MathTex("N", "=", "k", "+", r"\ell", font_size=48)
        
        b3_math[0].set_color(PURPLE)  # "N"
        b3_math[2].set_color(BLUE)    # "k"
        b3_math[4].set_color(RED)     # "\ell"

        b3_text = Tex(": Nombre total de billes", tex_template=tex_template, tex_to_color_map={"total": PURPLE})
        b3 = VGroup(b3_math, b3_text).arrange(RIGHT, aligned_edge=DOWN, buff=0.12)

        bullets = VGroup(b1, b2, b3).arrange(DOWN, aligned_edge=LEFT, buff=0.28).to_edge(LEFT, buff=0.6)

        # open-top box (draw only left, bottom, right edges)
        box_w = 5.0
        box_h = 2.6
        ref_rect = Rectangle(width=box_w, height=box_h, stroke_width=0).to_edge(RIGHT, buff=0.9)
        center = ref_rect.get_center()
        half_w, half_h = box_w / 2, box_h / 2
        top_left     = center + np.array([-half_w,  half_h, 0])
        bottom_left  = center + np.array([-half_w, -half_h, 0])
        bottom_right = center + np.array([ half_w, -half_h, 0])
        top_right    = center + np.array([ half_w,  half_h, 0])

        left_line   = Line(top_left, bottom_left, stroke_width=3, color=WHITE)
        bottom_line = Line(bottom_left, bottom_right, stroke_width=3, color=WHITE)
        right_line  = Line(bottom_right, top_right, stroke_width=3, color=WHITE)

        open_box = VGroup(left_line, bottom_line, right_line)

        # compute placement bounds (after positioning box)
        box_left = open_box.get_left()
        box_right = open_box.get_right()
        box_top = open_box.get_top()
        box_bottom = open_box.get_bottom()
        center_x = (box_left[0] + box_right[0]) / 2

        def generate_non_overlapping(n, x_min, x_max, y_min, y_max, r, max_attempts=500):
            pts = []
            attempts = 0
            min_dist = 2 * r + 0.02
            while len(pts) < n and attempts < max_attempts:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                ok = True
                for (px, py) in pts:
                    if math.hypot(px - x, py - y) < min_dist:
                        ok = False
                        break
                if ok:
                    pts.append((x, y))
                attempts += 1
            if len(pts) < n:  # fallback grid
                cols = max(1, int(math.sqrt(n)))
                rows = int(math.ceil(n / cols))
                xs = np.linspace(x_min, x_max, cols)
                ys = np.linspace(y_min, y_max, rows)
                pts = []
                for i in range(n):
                    xi = xs[i % cols]
                    yi = ys[i // cols]
                    pts.append((float(xi), float(yi)))
            return pts
        
        # --- gravity-based placement + precomputed trajectories ---

        def sample_x_positions(n, x_min, x_max, min_sep, max_attempts=2000):
            xs = []
            attempts = 0
            while len(xs) < n and attempts < max_attempts:
                x = random.uniform(x_min, x_max)
                ok = True
                for xi in xs:
                    if abs(xi - x) < min_sep:
                        ok = False
                        break
                if ok:
                    xs.append(x)
                attempts += 1
            if len(xs) < n:
                xs = list(np.linspace(x_min + min_sep/2, x_max - min_sep/2, n))
            return xs

        def simulate_settling(xs, ball_r, left_x, right_x, bottom_y, top_y, fixed_obstacles=None,
                            g=9.8, dt=1/60, max_time=4.0, restitution=0.25):
            n = len(xs)
            pos = np.array([[x, top_y + random.uniform(0.8, 1.6)] for x in xs], dtype=float)
            vel = np.zeros((n, 2), dtype=float)
            left = left_x + ball_r
            right = right_x - ball_r
            bottom = bottom_y + ball_r

            steps = int(max_time / dt)
            trajs = [[] for _ in range(n)]
            stable_count = 0

            for step in range(steps):
                # integrate
                for i in range(n):
                    vel[i,1] -= g * dt
                    pos[i] += vel[i] * dt
                    # wall collisions
                    if pos[i,0] < left:
                        pos[i,0] = left
                        vel[i,0] = -vel[i,0] * restitution
                        vel[i,1] *= 0.9
                    if pos[i,0] > right:
                        pos[i,0] = right
                        vel[i,0] = -vel[i,0] * restitution
                        vel[i,1] *= 0.9
                    if pos[i,1] < bottom:
                        pos[i,1] = bottom
                        vel[i,1] = -vel[i,1] * restitution
                        vel[i,0] *= 0.6

                # dynamic-dynamic collisions (pairwise)
                for i in range(n):
                    for j in range(i+1, n):
                        d = pos[j] - pos[i]
                        dist = math.hypot(d[0], d[1])
                        min_dist = 2 * ball_r
                        if dist < 1e-8:
                            # tiny jitter
                            theta = random.random() * 2 * math.pi
                            d = np.array([math.cos(theta), math.sin(theta)])
                            dist = 1e-8
                        overlap = min_dist - dist
                        if overlap > 0:
                            nrm = d / dist
                            pos[i] -= nrm * overlap * 0.5
                            pos[j] += nrm * overlap * 0.5
                            rv = vel[j] - vel[i]
                            vn = rv.dot(nrm)
                            if vn < 0:
                                J = -(1 + restitution) * vn / 2.0
                                vel[i] -= J * nrm
                                vel[j] += J * nrm

                # collisions with fixed obstacles (treated as immobile circles)
                if fixed_obstacles:
                    for i in range(n):
                        for sb in fixed_obstacles:
                            d = pos[i] - sb
                            dist = math.hypot(d[0], d[1])
                            min_dist = 2 * ball_r
                            if dist < 1e-8:
                                d = np.array([0.0, 1.0])
                                dist = 1e-8
                            overlap = min_dist - dist
                            if overlap > 0:
                                nrm = d / dist
                                pos[i] += nrm * overlap
                                vn = vel[i].dot(nrm)
                                if vn < 0:
                                    J = -(1 + restitution) * vn
                                    vel[i] += J * nrm

                vel *= 0.999  # small damping
                for i in range(n):
                    trajs[i].append(pos[i].copy())

                max_speed = np.max(np.linalg.norm(vel, axis=1))
                max_overlap = 0
                for i in range(n):
                    for j in range(i+1, n):
                        dist = np.linalg.norm(pos[i] - pos[j])
                        max_overlap = max(max_overlap, max(0, 2 * ball_r - dist))

                if max_speed < 0.01 and max_overlap < 0.001:
                    stable_count += 1
                    if stable_count > int(0.25 / dt):  # ~0.25s of stability
                        break
                else:
                    stable_count = 0

            final = [trajs[i][-1].copy() for i in range(n)]
            return trajs, final, len(trajs[0]) * dt

        # pick horizontal starts with minimum separation
        full_x_min = box_left[0] + ball_r + padding
        full_x_max = box_right[0] - ball_r - padding
        min_sep = 2 * ball_r + 0.02
        all_xs = sample_x_positions(k_num + ell_num, full_x_min, full_x_max, min_sep)
        random.shuffle(all_xs)
        blue_xs = all_xs[:k_num]
        red_xs = all_xs[k_num:]

        def make_circle(color):
            return Circle(radius=ball_r).set_fill(color, 1).set_stroke(width=0)

        # simulate blue balls (they interact with each other)
        blue_trajs, blue_final, blue_sim_time = simulate_settling(
            blue_xs, ball_r, box_left[0], box_right[0], box_bottom[1], box_top[1], fixed_obstacles=None
        )

        # create blue circles and animate along precomputed trajectories
        blue_balls = VGroup()
        for i, x in enumerate(blue_xs):
            start = blue_trajs[i][0]
            c = make_circle(BLUE).move_to(np.array([start[0], start[1], 0]))
            blue_balls.add(c)

        def traj_updater(traj):
            L = len(traj)
            def updater(mob, alpha):
                t = alpha * (L - 1)
                i0 = int(t)
                i1 = min(i0 + 1, L - 1)
                f = t - i0
                p = traj[i0] * (1 - f) + traj[i1] * f
                mob.move_to(np.array([p[0], p[1], 0]))
            return updater

        # blue_anims = [UpdateFromAlphaFunc(blue_balls[i], traj_updater(blue_trajs[i])) for i in range(len(blue_balls))]
        # self.play(LaggedStart(*blue_anims, lag_ratio=0.02), run_time=max(0.4, blue_sim_time))
        # self.next_slide()

        # freeze blue final positions for red simulation
        blue_final_positions = [np.array(p) for p in blue_final]

        # simulate red balls with blue final positions as static obstacles
        red_trajs, red_final, red_sim_time = simulate_settling(
            red_xs, ball_r, box_left[0], box_right[0], box_bottom[1], box_top[1],
            fixed_obstacles=blue_final_positions
        )

        # create red circles and animate
        red_balls = VGroup()
        for i, x in enumerate(red_xs):
            start = red_trajs[i][0]
            c = make_circle(RED).move_to(np.array([start[0], start[1], 0]))
            red_balls.add(c)

        # red_anims = [UpdateFromAlphaFunc(red_balls[i], traj_updater(red_trajs[i])) for i in range(len(red_balls))]
        # self.play(LaggedStart(*red_anims, lag_ratio=0.02), run_time=max(0.4, red_sim_time))
        # self.next_slide()

        # draw the box, reveal first line, animate blue balls
        self.play(Create(open_box))
        self.next_slide()

        # reveal first line and make blue balls appear, then animate their fall
        self.play(Write(b1), FadeIn(blue_balls))
        blue_anims = [UpdateFromAlphaFunc(blue_balls[i], traj_updater(blue_trajs[i]))
                    for i in range(len(blue_balls))]
        self.play(LaggedStart(*blue_anims, lag_ratio=0.02), run_time=max(0.4, blue_sim_time))
        self.next_slide()

        # reveal second line and handle red balls the same way
        self.play(Write(b2), FadeIn(red_balls))
        red_anims = [UpdateFromAlphaFunc(red_balls[i], traj_updater(red_trajs[i]))
                    for i in range(len(red_balls))]
        self.play(LaggedStart(*red_anims, lag_ratio=0.02), run_time=max(0.4, red_sim_time))
        self.next_slide()

        # reveal total
        self.play(Write(b3))
        self.next_slide()