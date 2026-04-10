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
        k_num = 2
        ell_num = 9
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
                            g=9, dt=1/60, max_time=4.0, restitution=0.25, collision_iters=3):
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

                for _ in range(collision_iters):
                    
                    # A. Dynamic-dynamic collisions (pairwise)
                    for i in range(n):
                        for j in range(i+1, n):
                            d = pos[j] - pos[i]
                            dist = math.hypot(d[0], d[1])
                            min_dist = 2 * ball_r
                            
                            if dist < min_dist: # Overlap detected!
                                if dist < 1e-8:
                                    theta = random.random() * 2 * math.pi
                                    d = np.array([math.cos(theta), math.sin(theta)])
                                    dist = 1e-8
                                
                                overlap = min_dist - dist
                                nrm = d / dist
                                
                                # Push them apart mathematically
                                pos[i] -= nrm * overlap * 0.6
                                pos[j] += nrm * overlap * 0.6
                                
                                # Bounce velocity
                                rv = vel[j] - vel[i]
                                vn = rv.dot(nrm)
                                if vn < 0:
                                    J = -(1 + restitution) * vn / 2.0
                                    vel[i] -= J * nrm
                                    vel[j] += J * nrm

                    # B. Collisions with fixed obstacles
                    if fixed_obstacles:
                        for i in range(n):
                            for sb in fixed_obstacles:
                                d = pos[i] - sb
                                dist = math.hypot(d[0], d[1])
                                min_dist = 2 * ball_r
                                
                                if dist < min_dist: # Overlap detected!
                                    if dist < 1e-8:
                                        d = np.array([0.0, 1.0])
                                        dist = 1e-8
                                    
                                    overlap = min_dist - dist
                                    nrm = d / dist
                                    
                                    # Push the falling ball OUT of the static ball
                                    pos[i] += nrm * overlap * 1.1
                                    
                                    # Bounce velocity
                                    vn = vel[i].dot(nrm)
                                    if vn < 0:
                                        J = -(1 + restitution) * vn
                                        vel[i] += J * nrm
                    
                    # C. Strict Boundary Clamp (Moved inside the iteration loop!)
                    # If the balls pushing each other forced someone through the wall, 
                    # this strictly snaps them back inside before the frame ends.
                    for i in range(n):
                        if pos[i,0] < left: pos[i,0] = left
                        if pos[i,0] > right: pos[i,0] = right
                        if pos[i,1] < bottom: pos[i,1] = bottom

                # 2. Damping and History
                vel *= 0.999  # small damping
                for i in range(n):
                    trajs[i].append(pos[i].copy())

                # 3. Check for stability
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
        # bb1.set_opacity(0)
        # self.add(bb1)

        bb2 = MathTex("=", f"{ell_num}", font_size=48)
        bb2[0].set_color(WHITE)      # "="
        bb2[1].set_color(RED)       # ell 
        bb2.next_to(ell_tex, RIGHT, buff=0.12)
        bb2.align_to(b2_text, DOWN)  # match baseline with the old text
        # bb2.set_opacity(0)
        # self.add(bb2)

        bb3 = MathTex(f"{k_num + ell_num}", font_size=48)
        bb3[0].set_color(PURPLE)      # k+ell
        bb3.next_to(b3_unchanged, RIGHT, buff=0.12)
        bb3.align_to(b3_transformed, DOWN)  # match baseline with the old text
        # bb3.set_opacity(0)
        # self.add(bb3)

        self.play(
            LaggedStart(*[Transform(b1_text, bb1), Transform(b2_text, bb2), Transform(b3_transformed, bb3)], lag_ratio=1)
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

        props1 = Tex("Concentration de boules bleues :",
                    tex_template=tex_template,
                    tex_to_color_map={"bleues": BLUE})
        props20 = MathTex("c = ", font_size=48)
        props21 = MathTex(
            r"\dfrac{",
            "k",
            "}{",
            "N",
            "}",
            font_size=48,
            tex_to_color_map={"k": BLUE, "N": PURPLE}
        )
        props22 = MathTex(
            r"\dfrac{",
            f"{k_num}",
            "}{",
            f"{ell_num+k_num}",
            "}",
            font_size=48,
            tex_to_color_map={
                str(k_num): BLUE,
                str(ell_num+k_num): PURPLE
            }
        )
        props23 = MathTex(
            f"{(k_num/(ell_num+k_num)):.2f}",
            font_size=48
        )
        
        txt_group = VGroup(props1, props20).arrange(RIGHT, buff=0.5)
        props = VGroup(txt_group, props21).arrange(RIGHT, buff=0.12).move_to(DOWN * 1.5)

        props22.next_to(props20, RIGHT, buff=0.12)
        props23.next_to(props20, RIGHT, buff=0.12)

        math_gp = VGroup(props20, props23)

        self.play(Write(props))
        self.next_slide()
        self.play(ReplacementTransform(props21, props22))
        self.next_slide()
        self.play(
            FadeOut(props22, shift= UP * 0.3),
            FadeIn(props23, shift= UP * 0.3)
        )
        
        self.next_slide()
        
        bullets.generate_target()
        bullets.target.set_x(bullets.get_center()[0])
        bullets.target.set_y(bullets.get_center()[1])

        bullets.target.shift(LEFT * 0.5)

        # c.next_to(bullets.target[0], RIGHT, buff=0.8)


        self.play(
            MoveToTarget(bullets),
            FadeOut(props1),
            VGroup(math_gp, props23).animate.next_to(bullets.target[0], RIGHT, buff=0.8)
        )
        # self.play(Transform(b4, ))

        self.next_slide()

        N = k_num + ell_num
        c = k_num / N
        E = 1 / c

        expect1 = Tex("Nombre moyen de tirages avant succès :",
                        tex_template=tex_template,
                        )
        expect2 = MathTex(
            "E(",
            "c",
            ") =",
            r"\dfrac{1}{",
            "c",
            "}",
            font_size=48
        )
        expect3 = MathTex(
            "E(",
            f"{c:.2f}",
            ") =",
            f"{E:.2f}",
            font_size = 48
        )

        exp_gp = VGroup(expect1, expect2).arrange(DOWN, buff=0.12).move_to(DOWN * 1.5)
        expect3.move_to(expect2, aligned_edge=LEFT)

        self.play(Write(exp_gp))

        self.next_slide()

        self.play(
            ReplacementTransform(expect2[0], expect3[0]),
            ReplacementTransform(expect2[1], expect3[1]),
            ReplacementTransform(expect2[2], expect3[2]), 
            FadeOut(VGroup(expect2[3], expect2[4], expect2[5]), shift=UP*0.3),
            FadeIn(expect3[3], shift=UP*0.3)
        )

        self.next_slide()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        center = RIGHT * 1.5
        half_w, half_h = box_w / 2, box_h / 2
        
        top_left     = center + np.array([-half_w,  half_h, 0])
        bottom_left  = center + np.array([-half_w, -half_h, 0])
        bottom_right = center + np.array([ half_w, -half_h, 0])
        top_right    = center + np.array([ half_w,  half_h, 0])

        left_line   = Line(top_left, bottom_left, stroke_width=3, color=WHITE)
        bottom_line = Line(bottom_left, bottom_right, stroke_width=3, color=WHITE)
        right_line  = Line(bottom_right, top_right, stroke_width=3, color=WHITE)

        open_box = VGroup(left_line, bottom_line, right_line)
        self.play(Wait(0.5))
        self.play(Create(open_box))

        box_top_y = top_left[1]
        box_bot_y = bottom_left[1]
        left_x = top_left[0]
        right_x = top_right[0]

        self.next_slide()

        k_tracker = ValueTracker(0)
        ell_tracker = ValueTracker(0)
        self.add(k_tracker, ell_tracker)
        
        # --- Top Text: k, ell, N ---
        left_k = VGroup(MathTex("k =", tex_to_color_map={"k": BLUE}), Integer(0, color=BLUE)).arrange(RIGHT)
        left_k[1].add_updater(lambda m: m.set_value(k_tracker.get_value()).next_to(left_k[0], RIGHT, buff=0.2))
        
        left_ell = VGroup(MathTex(r"\ell =", tex_to_color_map={r"\ell": RED}), Integer(0, color=RED)).arrange(RIGHT)
        left_ell[1].add_updater(lambda m: m.set_value(int(ell_tracker.get_value())).next_to(left_ell[0], RIGHT, buff=0.2))
        
        left_N = VGroup(MathTex("N =", tex_to_color_map={"N": PURPLE}), Integer(0, color=PURPLE)).arrange(RIGHT)
        left_N[1].add_updater(lambda m: m.set_value(int(k_tracker.get_value() + ell_tracker.get_value())).next_to(left_N[0], RIGHT, buff=0.2))
        
        # Arrange horizontally on top
        left_stats = VGroup(left_k, left_ell, left_N).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(open_box, LEFT, buff=1)

        # --- Bottom Text: c, E(c) ---
        bot_c = VGroup(MathTex("c ="), DecimalNumber(0, num_decimal_places=4)).arrange(RIGHT)
        bot_c[1].add_updater(lambda m: m.set_value(
            k_tracker.get_value() / max(0.001, (k_tracker.get_value() + ell_tracker.get_value()))
        ).next_to(bot_c[0], RIGHT, buff=0.2))

        bot_E = VGroup(MathTex("E(c) ="), DecimalNumber(0, num_decimal_places=2)).arrange(RIGHT)
        bot_E[1].add_updater(lambda m: m.set_value(
            (k_tracker.get_value() + ell_tracker.get_value()) / max(1.0, k_tracker.get_value())
        ).next_to(bot_E[0], RIGHT, buff=0.2))

        # The bottom stats remain horizontally arranged under the box
        bot_stats = VGroup(bot_c, bot_E).arrange(RIGHT, buff=2).next_to(open_box, DOWN, buff=0.5)

        # Helper to animate trajectories
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
        
        def traj_and_fade_updater(traj):
            L = len(traj)
            final_pos = np.array([traj[-1][0], traj[-1][1], 0])
            def updater(mob, alpha):
                if alpha < 0.15: mob.set_opacity(alpha / 0.15)
                else: mob.set_opacity(1.0)
                
                if alpha >= 0.999:
                    mob.move_to(final_pos)
                    return
                
                t = alpha * (L - 1)
                i0 = int(t)
                i1 = min(i0 + 1, L - 1)
                f = t - i0
                p = traj[i0] * (1 - f) + traj[i1] * f
                mob.move_to(np.array([p[0], p[1], 0]))
            return updater

        b_trajs1, b_final1, _ = simulate_settling([random.uniform(-1, 0)], ball_r, left_x, right_x, box_bot_y, box_top_y)
        b_trajs2, b_final2, _ = simulate_settling([random.uniform(-1, 0)], ball_r, left_x, right_x, box_bot_y, box_top_y)
        r_trajs, r_final, _ = simulate_settling([random.uniform(0, 1)], ball_r, left_x, right_x, box_bot_y, box_top_y, fixed_obstacles=b_final1+b_final2)
        
        b0_balls = VGroup(
            make_circle(BLUE).move_to(np.array([b_trajs1[0][0][0], b_trajs1[0][0][1], 0])),
            make_circle(BLUE).move_to(np.array([b_trajs2[0][0][0], b_trajs2[0][0][1], 0])), 
            make_circle(RED).move_to(np.array([r_trajs[0][0][0], r_trajs[0][0][1], 0]))
        )
        
        # Fade in high above
        self.play(FadeIn(b0_balls, shift=DOWN * 0.5))
        
        # Let them fall
        self.play(
            UpdateFromAlphaFunc(b0_balls[0], traj_updater(b_trajs1[0])),
            UpdateFromAlphaFunc(b0_balls[1], traj_updater(b_trajs2[0])),
            UpdateFromAlphaFunc(b0_balls[2], traj_updater(r_trajs[0])),
            run_time=1.5
        )
        
        # Accumulate fixed obstacles for the physics engine
        fixed_obstacles = b_final1 + b_final2 + r_final

        # 4 & 5) Display the counters and snap them to 1
        k_tracker.set_value(1)
        ell_tracker.set_value(1)
        self.play(Write(left_stats), Write(bot_stats))

        self.next_slide()

        total_remaining = 207
        manual_count = 5
        
        # 1. Generate all random X positions
        xs = [random.uniform(left_x + ball_r, right_x - ball_r) for _ in range(total_remaining)]
        
        all_trajs = []
        for x in xs:
            # Drop a single ball onto the pile
            traj, final, _ = simulate_settling(
                [x], ball_r, left_x, right_x, box_bot_y, box_top_y, 
                fixed_obstacles=fixed_obstacles
            )
            all_trajs.append(traj[0])
            # Freeze it exactly where it lands so the next ball can hit it!
            fixed_obstacles.extend(final)
            
        # 3. Create the Mobjects (they start invisible above the box)
        batch_balls = VGroup()
        for i in range(total_remaining):
            start_pos = all_trajs[i][0]
            ball = make_circle(RED).move_to(np.array([start_pos[0], start_pos[1], 0]))
            ball.start_y = start_pos[1] # <--- Custom attribute!
            batch_balls.add(ball)
        
        # Add the threshold counter updater
        base_ell = ell_tracker.get_value()
        def counter_updater(tracker, balls=batch_balls, base=base_ell):
            # Only counts balls whose Y-coordinate has physically dropped below the box top
            dropped_in = sum(1 for b in balls if b.get_y() < b.start_y - 1e-3)
            tracker.set_value(base + dropped_in)
            
        ell_tracker.add_updater(counter_updater)

        for i in range(manual_count):
            ball = batch_balls[i]
            traj = all_trajs[i]
            
            self.play(
                AnimationGroup(
                    FadeIn(ball, run_time=0.1),
                    UpdateFromAlphaFunc(ball, traj_and_fade_updater(traj), run_time=0.8) # Fixed runtime looks great for gravity
                )
            )
            self.next_slide()

        auto_anims = []
        for i in range(manual_count, total_remaining):
            ball = batch_balls[i]
            traj = all_trajs[i]
            
            anim = AnimationGroup(
                FadeIn(ball, run_time=0.1),
                UpdateFromAlphaFunc(ball, traj_and_fade_updater(traj), run_time=0.8)
            )
            auto_anims.append(anim)
            
        # Because the physics are totally independent now, LaggedStart works flawlessly!
        # lag_ratio=0.1 means the next ball starts when the current one is 10% done dropping.
        self.play(
            LaggedStart(*auto_anims, lag_ratio=0.1)
        )
        
        # Cleanup and finalize
        ell_tracker.remove_updater(counter_updater)
        ell_tracker.set_value(base_ell + total_remaining)
        
        self.wait(0.25)
        self.next_slide()

        self.play(FadeOut(batch_balls), FadeOut(b0_balls))

        scale_f = 0.8

        box_w = 5.5 * scale_f
        box_h = 3.5 * scale_f
        ball_r = 0.17 * scale_f

        k_start = 47
        ell_fixed = 103
        k_tracker.set_value(0)
        ell_tracker.set_value(0)
        etot_tracker = ValueTracker(0)
        self.add(etot_tracker)

        self.play(
            VGroup(left_stats, VGroup(open_box, bot_stats)).animate.to_edge(UP, buff=0.6).scale(scale_f)
        )

        box_top_y = open_box.get_top()[1]
        box_bot_y = open_box.get_bottom()[1]
        left_x = open_box.get_left()[0]
        right_x = open_box.get_right()[0]
        

        etot_label = MathTex(
            r"E_\mathrm{tot} =",
            font_size=48,
            tex_to_color_map={r"E_\mathrm{tot}": YELLOW}
        )
        etot_num = DecimalNumber(0, num_decimal_places=2, color=YELLOW)
        etot_num.add_updater(lambda m: m.set_value(etot_tracker.get_value()).next_to(etot_label, RIGHT, buff=0.2))

        etot_group = VGroup(etot_label, etot_num).arrange(RIGHT).move_to(DOWN * 1.5)

        refill_fixed = []
        refill_data = []

        colors = [BLUE]*k_start + [RED]*ell_fixed
        random.shuffle(colors)

        for col in colors:
            x_pos = random.uniform(left_x+ball_r,right_x-ball_r)
            t, f, _ = simulate_settling([x_pos], ball_r, left_x, right_x, box_bot_y, box_top_y, 
                                        fixed_obstacles=refill_fixed)
            refill_data.append({"color": col, "traj": t[0], "final": f[0]})
            refill_fixed.extend(f)

        refill_data_sorted = sorted(refill_data, key=lambda d: d["final"][1])

        refill_balls = VGroup()
        for data in refill_data_sorted:
            start_pos = data["traj"][0]
            ball = make_circle(data["color"]).move_to(np.array([data["traj"][0][0], data["traj"][0][1], 0]))
            ball.start_y = start_pos[1] 
            ball.color_id = data["color"]
            ball.set_opacity(0)
            refill_balls.add(ball)

        def refill_counter(tracker):
            b_in = sum(1 for b in refill_balls if b.get_y() < b.start_y - 1e-3 and b.color_id == BLUE)
            r_in = sum(1 for b in refill_balls if b.get_y() < b.start_y - 1e-3 and b.color_id == RED)
            k_tracker.set_value(b_in)
            ell_tracker.set_value(r_in)

        self.add(k_tracker, ell_tracker)
        k_tracker.add_updater(refill_counter)

        refill_anim = [
            UpdateFromAlphaFunc(
                refill_balls[i],
                traj_and_fade_updater(refill_data_sorted[i]["traj"]),
                run_time=0.8
            )
            for i in range(k_start+ell_fixed)
        ]

        self.play(
            LaggedStart(
                *refill_anim,
                lag_ratio=0.05
            ),
            run_time=5
        )

        self.next_slide()
        self.play(Write(etot_group))

        k_tracker.remove_updater(refill_counter)
        k_tracker.set_value(k_start)
        ell_tracker.set_value(ell_fixed)

        self.next_slide()

        blue_indices = [i for i, d in enumerate(refill_data_sorted)
                        if d["color"]==BLUE]
        random.shuffle(blue_indices)

        for step, idx in enumerate(blue_indices):
            curr_k = k_tracker.get_value()
            curr_ell = ell_tracker.get_value()
            energy_increment = (curr_k + curr_ell) / curr_k

            ball_to_extract = refill_balls[idx]

            rt = max(0.15, 0.6 * (0.90 ** step))

            self.play(
                ball_to_extract.animate.move_to(UP * 5).set_opacity(0),
                k_tracker.animate.set_value(curr_k - 1),
                etot_tracker.animate.set_value(etot_tracker.get_value() + energy_increment),
                run_time=rt,
                rate_func=linear
            )

            if step < 3:
                self.wait(0.2)
                self.next_slide()

        bot_E[1].clear_updaters()
        infty_symbol = MathTex(r"\infty", font_size=48).next_to(bot_E[0], RIGHT, buff=0.2)

        self.play(ReplacementTransform(bot_E[1], infty_symbol))

        self.wait(1)
        self.next_slide()