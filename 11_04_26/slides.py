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
        k_num = 9
        ell_num = 1
        ball_r = 0.17
        padding = 0.06

        # lines (one-by-one)
        k_tex = MathTex("k", font_size=48).set_color(BLUE)
        b1_text = Tex(": Nombre de billes bleues",
                    tex_template=tex_template,
                    tex_to_color_map={"bleues": BLUE})
        b1 = VGroup(k_tex, b1_text).arrange(RIGHT, buff=0.12).to_edge(LEFT, buff=0.6)

        ell_tex = MathTex(r"\ell", font_size=48).set_color(RED)
        b2_text = Tex(": Nombre de billes rouges",
                    tex_template=tex_template,
                    tex_to_color_map={"rouges": RED})
        b2 = VGroup(ell_tex, b2_text).arrange(RIGHT, buff=0.12).to_edge(LEFT, buff=0.6)
        
        b3_f1 = MathTex("N", font_size=48).set_color(PURPLE)
        b3_f2 = MathTex("=", font_size=48)
        b3_f3 = MathTex("k", font_size=48).set_color(BLUE)
        b3_f4  = MathTex("+", font_size=48)
        b3_f5 = MathTex(r"\ell", font_size=48).set_color(RED)

        b3_text = Tex(": Nombre total de billes",
                    tex_template=tex_template,
                    tex_to_color_map={"total": PURPLE})
        
        b3_unchanged = VGroup(b3_f1, b3_f2).arrange(RIGHT, buff=0.12)
        
        b3_transformed = VGroup(b3_f3, b3_f4, b3_f5, b3_text).arrange(RIGHT, buff=0.12)

        b3 = VGroup(b3_unchanged, b3_transformed).arrange(RIGHT, buff=0.12).to_edge(LEFT, buff=0.6)

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
                            g=9, dt=1/60, max_time=4.0, restitution=0.25):
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

        red_final_positions = [np.array(p) for p in red_final]

        # red_anims = [UpdateFromAlphaFunc(red_balls[i], traj_updater(red_trajs[i])) for i in range(len(red_balls))]
        # self.play(LaggedStart(*red_anims, lag_ratio=0.02), run_time=max(0.4, red_sim_time))
        # self.next_slide()

        # draw the box, reveal first line, animate blue balls
        self.play(Create(open_box))
        self.next_slide()

        # prepare trackers + updaters
        trackers_b, trackers_r = [ValueTracker(0.0) for _ in range(len(blue_balls))], [ValueTracker(0.0) for _ in range(len(red_balls))]
        Ls_b, Ls_r = [len(traj) for traj in blue_trajs], [len(traj) for traj in red_trajs]

        def make_tracker_updater(traj, tracker, L):
            def updater(mob):
                t = tracker.get_value() * (L - 1)
                i0 = int(t)
                i1 = min(i0 + 1, L - 1)
                f = t - i0
                p = traj[i0] * (1 - f) + traj[i1] * f
                mob.move_to(np.array([p[0], p[1], 0]))
            return updater

        for i, mob in enumerate(blue_balls):
            mob.add_updater(make_tracker_updater(blue_trajs[i], trackers_b[i], Ls_b[i]))
        for i, mob in enumerate(red_balls):
            mob.add_updater(make_tracker_updater(red_trajs[i], trackers_r[i], Ls_r[i]))

        nb, nr = len(blue_balls), len(red_balls)
        single_run_b, single_run_r = blue_sim_time, red_sim_time
        fade_time = 0.6
        write_time = 1.6
        lag_ratio = 0.05

        initial_ys_b, initial_ys_r = [blue_trajs[i][0][1] for i in range(nb)], [red_trajs[i][0][1] for i in range(nr)]
        order_b, order_r = sorted(range(nb), key=lambda i: initial_ys_b[i]), sorted(range(nr), key=lambda i: initial_ys_r[i])

        # Build ordered tracker animations and play
        ordered_anims_b, ordered_anims_r = [trackers_b[i].animate.set_value(1.0) for i in order_b], [trackers_r[i].animate.set_value(1.0) for i in order_r]

        # run Write + FadeIn first, then animate trackers (no Succession nesting)
        self.play(
            Write(b1, run_time=write_time),
            FadeIn(blue_balls, run_time=fade_time),
            Succession(
                Wait(fade_time),
                LaggedStart(*ordered_anims_b, lag_ratio=lag_ratio, run_time=single_run_b)
                )
            )

        # finalize: place static copies and remove updaters
        static_blue = VGroup()
        for p in blue_final_positions:
            c = make_circle(BLUE).move_to(np.array([p[0], p[1], 0]))
            static_blue.add(c)
        self.add(static_blue)
        for m in blue_balls:
            m.clear_updaters()
        self.remove(blue_balls)
        blue_balls = static_blue

        self.next_slide()

        self.play(
            Write(b2, run_time=write_time),
            FadeIn(red_balls, run_time=fade_time),
            Succession(
                Wait(fade_time),
                LaggedStart(*ordered_anims_r, lag_ratio=lag_ratio, run_time=single_run_r)
                )
            )
        
        static_red = VGroup()
        for p in red_final_positions:
            c = make_circle(RED).move_to(np.array([p[0], p[1], 0]))
            static_red.add(c)
        self.add(static_red)
        for m in red_balls:
            m.clear_updaters()
        self.remove(red_balls)
        red_balls = static_red

        self.next_slide()

        # reveal total
        self.play(Write(b3))
        self.next_slide()

        bb1 = MathTex("=", f"{k_num}", font_size=48)
        bb1[0].set_color(WHITE)      # "="
        bb1[1].set_color(BLUE)       # k 
        bb1.next_to(k_tex, RIGHT, buff=0.12)
        bb1.align_to(b1_text, DOWN)  # match baseline with the old text
        bb1.set_opacity(0)
        self.add(bb1)

        bb2 = MathTex("=", f"{ell_num}", font_size=48)
        bb2[0].set_color(WHITE)      # "="
        bb2[1].set_color(RED)       # ell 
        bb2.next_to(ell_tex, RIGHT, buff=0.12)
        bb2.align_to(b2_text, DOWN)  # match baseline with the old text
        bb2.set_opacity(0)
        self.add(bb2)

        bb3 = MathTex(f"{k_num + ell_num}", font_size=48)
        bb3[0].set_color(PURPLE)      # k+ell
        bb3.next_to(b3_unchanged, RIGHT, buff=0.12)
        bb3.align_to(b3_transformed, DOWN)  # match baseline with the old text
        bb3.set_opacity(0)
        self.add(bb3)

        self.play(
            ReplacementTransform(b1_text, bb1),
            bb1.animate.set_opacity(1.0)
                  )
        self.play(
            ReplacementTransform(b2_text, bb2),
            bb2.animate.set_opacity(1.0)
                  )
        self.play(
            ReplacementTransform(b3_transformed, bb3),
            bb3.animate.set_opacity(1.0)
                  )
        self.next_slide()

        box_group = VGroup(open_box, static_blue, static_red)
        box_group.generate_target()
        box_group.target.to_edge(UP, buff=0.6)

        bullets.generate_target()
        bullets.target.to_edge(UP, buff=0.6)
        bullets.target.set_x(-config["frame_width"] / 4)
        bullets.target.set_y(box_group.target.get_center()[1])

        self.play(
            MoveToTarget(box_group),
            MoveToTarget(bullets)
        )
        
        self.next_slide()

        props1 = Tex("Concentration de boules rouges :",
                    tex_template=tex_template,
                    tex_to_color_map={"rouges": RED})
        props2 = MathTex(r"c = \dfrac{\ell}{N}", font_size=48,
                         tex_to_color_map={r"\ell": RED, "N": PURPLE})
        props3 = MathTex(fr"= \dfrac{{{ell_num}}}{{{ell_num+k_num}}}",
                         font_size=48,
                         tex_to_color_map={
                             str(ell_num): RED,
                             str(ell_num+k_num): PURPLE
                         })
        
        props = VGroup()