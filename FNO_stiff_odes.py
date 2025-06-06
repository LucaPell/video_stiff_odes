from manim import *

background = "white"
if background == "white":
    config.background_color = "#a0a0a0"
    Tex.set_default(color=BLACK)
    MathTex.set_default(color=BLACK)
    Text.set_default(color=BLACK)
    color_i_app = "PURE_BLUE"
    color_Q = "#A2142F"
    color_P = "#177245"
    color_L = "#0072BD"
    color_rectangle_FNO = "#177245"
    color_brace = "BLACK"
    color_rectangle_red = "#B2264A"
    color_rectangle_blue = "#1E489F"
    color_arrow = "BLACK"
    color_affine_transf = "PURE_BLUE"
    color_affine_kernel = "PURE_RED"
    color_kernel_function = "PURE_BLUE"
    color_fourier = "#49A88F"

Text.set_default(font="Noto Sans", font_size=50)
config.tex_template = TexTemplate(
    preamble=r"""
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{noto}  % Requires Noto Sans installed in LaTeX
        \renewcommand{\familydefault}{\sfdefault}  % Force sans-serif
        \newcommand{\Na}{\text{Na}}
        \newcommand{\Cal}{\text{Ca}}
        \newcommand{\K}{\text{K}}
        \newcommand{\Ca}{\text{Ca}^{2+}} 
        \newcommand{\CaMKtrap}{\text{CaMK}_{\text{trap}}} 
        \newcommand{\CaMKbound}{\text{CaMK}_{\text{bound}}} 
        \newcommand{\CaMK}{\text{CaMK}}
        \newcommand{\diff}{\text{diff}}
        """
)


class FNO_stiff_odes(MovingCameraScene):
    def construct(self):
        # Set camera orientation for better viewing
        # self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        etichette_font = 30
        text_font = 28
        title_font = 36
        label_font_small = 28
        label_font_big = 34
        d_input = 1
        n_points = 10
        d_v = 6
        k_max = 3
        d_output = 4
        self.camera.background_color = WHITE
        nome = (
            Tex("Luca Pellegrini (PhD Student Unipv/USI)", font_size=10)
            .to_edge(DOWN + LEFT)
            .shift(0.5 * DOWN)
        )

        image = ImageMobject("titolo_bianco.png")

        self.play(FadeIn(image))
        self.wait(20)
        self.play(FadeOut(image))

        ##=========================================================================================##
        #### Our Objective

        objective_title = Text("Our objective", font_size=title_font, color=WHITE)
        objective_title.to_edge(UP)
        rect = Rectangle(
            color="#B2264A",
            fill_color="#B2264A",
            fill_opacity=1,
            height=1.1,
            width=config.frame_width,
        ).move_to(objective_title.get_center() + 0.2 * UP)

        objective = MathTex(
            r"\text{Use Neural Operator methods for learning the dynamics of ionics models}",
            font_size=text_font,
        )
        objective.move_to(2 * UP)
        objective_title.shift(0.2 * UP)

        rectangle_objective = Rectangle(color=color_rectangle_red, height=0.2)
        rectangle_objective.surround(objective)
        ode = MathTex(
            r"&C_m\frac{dV}{dt} + I_{ion}(V,\vec{w},\vec{c}\,) =",
            r"I_{app}",
            r"&& t\in[0,T],\\"
            r"&\frac{dw_j}{dt} = \alpha_j(V)(1-w_j) + \beta_j(V)w_j, && j=1,...,M,\\"
            r"&\frac{dc_k}{dt} = - \frac{I_{c_k}(V,w) A_{cap}}{V_{c_k} z_{c_k}}F, && k =1,...,S.",
            font_size=text_font,
        )

        # ode = MathTex(
        #     r"\displaystyle & \frac{dV}{dt} = \bar{g}_{Na}m^3h(V-V_{Na})+ \bar{g}_K n^4(V-V_K )+\bar{g}_L(V-V_L) +",
        #     r"I_{app}\\",
        #     r" & \frac{dm}{dt} = \alpha_m(V)(1-m)-\beta_m(V)m \\ & \frac{dh}{dt} = \alpha_h(V)(1-h)-\beta_h(V)h \\ & \frac{dn}{dt} = \alpha_n(V)(1-n)-\beta_n(V)n",
        #     font_size=text_font,
        # )

        ode.next_to(objective, 2 * DOWN)
        brace_ode = Brace(ode, direction=([-1, 0, -1]), color=color_brace)
        self.play(
            Create(nome),
            FadeIn(rect),
            Write(objective_title),
            Create(objective),
            Create(rectangle_objective),
        )
        self.wait(7)
        self.play(
            Write(ode),
            Write(brace_ode),
        )

        NO_ode = MathTex(
            r"\mathcal{G}^\dagger :  \mathbb{R}^+&\times [0,T] \rightarrow \mathcal{D}    \\",
            r" &I_{app}",
            r"\mapsto (V,\vec{w},\vec{c}\,)",
            font_size=text_font,
        )
        NO_ode.next_to(ode, 4 * DOWN)
        self.wait(5)
        self.play(Create(NO_ode[0]))

        # action potential
        T_eval = np.linspace(0, 30, 5040)
        sol = np.load("hh_sol_ap.npy")
        Iapp = np.load("I_app_ap.npy")
        V = sol[0, :]
        m = sol[1, :]
        h = sol[2, :]
        n = sol[3, :]

        axes_scale = 0.7
        x_length, y_length = (
            2.5 * axes_scale,
            2.5 * axes_scale,
        )

        ax_I_app = Axes(
            x_range=[0, 30, 10],
            y_range=[0, 15, 1],
            x_length=x_length,
            y_length=y_length,
            x_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            y_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            tips=False,
            color=BLACK,
        ).next_to(NO_ode, LEFT, buff=1.5)

        ax_V = Axes(
            x_range=[0, 30, 10],
            y_range=[-20, 120, 10],
            x_length=x_length,
            y_length=y_length,
            x_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            y_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            tips=False,
            color=BLACK,
        )

        ax_m = Axes(
            x_range=[0, 30, 10],
            y_range=[0, 1, 0.2],
            x_length=x_length,
            y_length=y_length,
            x_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            y_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            tips=False,
            color=BLACK,
        )

        ax_h = Axes(
            x_range=[0, 30, 10],
            y_range=[0, 1, 0.2],
            x_length=x_length,
            y_length=y_length,
            x_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            y_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            tips=False,
            color=BLACK,
        )

        ax_n = Axes(
            x_range=[0, 30, 10],
            y_range=[0, 1, 0.2],
            x_length=x_length,
            y_length=y_length,
            x_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            y_axis_config={
                "include_ticks": False,
                "font_size": 20,
                "color": GRAY,
            },
            tips=False,
            color=BLACK,
        )

        axes_group = (
            VGroup(ax_V, ax_m, ax_h, ax_n)
            .arrange_in_grid(
                rows=2,
                cols=2,
                buff=0.8,
                flow_order="dr",
            )
            .scale(axes_scale)
            .center()
        ).next_to(NO_ode, RIGHT, buff=1.5)

        label_scale = 0.5 * axes_scale

        x_label_I_app = (
            Tex(r"Time (ms)", color=BLACK)
            .scale(label_scale)
            .next_to(ax_I_app.x_axis, 1.2 * DOWN, buff=0.1)
        )
        y_label_I_app = (
            MathTex(r"I_{app}", color=color_i_app)
            .scale(label_scale)
            .next_to(ax_I_app.y_axis, LEFT, buff=0.1)
            .rotate(90 * DEGREES)
        )
        x_label_V = (
            Tex(r"Time (ms)", color=BLACK)
            .scale(label_scale)
            .next_to(ax_V.x_axis, 1.2 * DOWN, buff=0.1)
        )
        y_label_V = (
            Tex(r"V", color=BLACK)
            .scale(label_scale)
            .next_to(ax_V.y_axis, LEFT, buff=0.1)
            .rotate(90 * DEGREES)
        )

        x_label_m = (
            Tex(r"Time (ms)", color=BLACK)
            .scale(label_scale)
            .next_to(ax_m.x_axis, 1.2 * DOWN, buff=0.1)
        )
        y_label_m = (
            Tex(r"m", color=BLACK)
            .scale(label_scale)
            .next_to(ax_m.y_axis, LEFT, buff=0.1)
            .rotate(90 * DEGREES)
        )

        x_label_h = (
            Tex(r"Time (ms)", color=BLACK)
            .scale(label_scale)
            .next_to(ax_h.x_axis, 1.2 * DOWN, buff=0.1)
        )
        y_label_h = (
            Tex(r"h", color=BLACK)
            .scale(label_scale)
            .next_to(ax_h.y_axis, LEFT, buff=0.1)
            .rotate(90 * DEGREES)
        )

        x_label_n = (
            Tex(r"Time (ms)", color=BLACK)
            .scale(label_scale)
            .next_to(ax_n.x_axis, 1.2 * DOWN, buff=0.1)
        )
        y_label_n = (
            Tex(r"n", color=BLACK)
            .scale(label_scale)
            .next_to(ax_n.y_axis, LEFT, buff=0.1)
            .rotate(90 * DEGREES)
        )

        graph_I_app = ax_I_app.plot_line_graph(
            T_eval,
            Iapp,
            add_vertex_dots=False,
            line_color=color_i_app,
            stroke_width=3 * axes_scale,
        )
        graph_V = ax_V.plot_line_graph(
            T_eval,
            V,
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=3 * axes_scale,
        )
        graph_m = ax_m.plot_line_graph(
            T_eval,
            m,
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=3 * axes_scale,
        )
        graph_h = ax_h.plot_line_graph(
            T_eval,
            h,
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=3 * axes_scale,
        )
        graph_n = ax_n.plot_line_graph(
            T_eval,
            n,
            add_vertex_dots=False,
            line_color=BLACK,
            stroke_width=3 * axes_scale,
        )
        self.wait(2)
        self.play(
            FadeIn(ax_I_app),
            Write(VGroup(x_label_I_app, y_label_I_app)),
            Create(graph_I_app),
            NO_ode[1].animate.set_color(color_i_app),
            ode[1].animate.set_color(color_i_app),
        )
        self.wait(2)

        self.play(
            Create(NO_ode[2]),
            FadeIn(axes_group),
            Write(VGroup(x_label_m, x_label_n)),
            Write(VGroup(y_label_V, y_label_m, y_label_h, y_label_n)),
            Create(VGroup(*graph_V, *graph_m, *graph_h, *graph_n)),
            run_time=3,
        )
        self.wait(5)
        self.play(
            FadeOut(
                objective_title,
                objective,
                rectangle_objective,
                ode,
                brace_ode,
                NO_ode,
                axes_group,
                ax_I_app,
                y_label_V,
                y_label_m,
                y_label_h,
                y_label_n,
                x_label_I_app,
                y_label_I_app,
                graph_I_app,
                graph_V,
                graph_m,
                graph_h,
                graph_n,
                x_label_m,
                x_label_n,
            )
        )

        ##=========================================================================================##
        #### Introduction to neural operator
        title_NO = Text("Neural Operator", font_size=title_font, color=WHITE)

        title_NO.to_edge(UP)
        title_NO.shift(0.2 * UP)
        NO_system = MathTex(
            r"&(\mathcal{N}_a u)(x) = f(x), && x\in D,\\"
            r"&(\mathcal{B} u) (x)  = g(x), && x\in \partial D.",
            font_size=text_font,
        )
        NO_system.move_to(2 * UP)
        braces_NO = Brace(NO_system, direction=([-1, 0, -1]), color=color_brace)
        # text_1_system_NO = MathTex(
        #     r"\text{Where } \mathbf{L}_a \text{ is a differential operator parametrized by } a\in \mathcal{A} \text{ and } u\in \mathcal{U}.",
        #     font_size=text_font,
        # )
        # text_1_system_NO.next_to(NO_system, 2 * DOWN)
        text_system_NO = MathTex(
            r"\text{Under this setting we define the solution operator } \mathcal{G}^\dagger = \mathcal{N}^{-1}_a",
            font_size=text_font,
        )
        text_system_NO.next_to(NO_system, 2 * DOWN)
        NO_map = MathTex(
            r" \mathcal{G}^\dagger : & \mathcal{A} \rightarrow \mathcal{U} \\"
            r" & a \mapsto u_a",
            font_size=text_font,
        )
        NO_map.next_to(text_system_NO, 2 * DOWN)
        text_NO = MathTex(
            r"\text{A Neural Operator in general can be defined as a composition of operators:}\\",
            font_size=text_font,
        )
        text_NO.next_to(NO_map, 2 * DOWN)
        NO = MathTex(
            r"\mathcal{G}_{\theta} = \mathcal{Q} \circ \mathcal{L}_L\circ ...\circ \mathcal{L}_1\circ \mathcal{P}",
            font_size=text_font,
        )
        NO.next_to(text_NO, 2 * DOWN)

        NO_approx = MathTex(
            r"\mathcal{G}_{\theta}  \approx \mathcal{G}^\dagger ", font_size=text_font
        )
        NO_approx.next_to(NO, 2 * DOWN)
        rectangle_NO = Rectangle(color=color_rectangle_red)
        rectangle_NO.surround(NO_approx)
        self.play(
            Write(title_NO),
        )
        self.wait(7)
        self.play(
            Create(NO_system),
            Write(braces_NO),
        )
        self.wait(8)
        self.play(
            Create(text_system_NO),
            Create(NO_map),
        )
        self.wait(8)
        self.play(
            Create(text_NO),
            Create(NO),
        )
        self.wait(4)
        self.play(
            Create(NO_approx),
            Create(rectangle_NO),
        )
        self.wait(10)
        self.play(
            FadeOut(
                NO_system,
                braces_NO,
                text_system_NO,
                NO_map,
                text_NO,
                NO,
                NO_approx,
                rectangle_NO,
            )
        )

        rectangle_which_NO = Rectangle(color=color_rectangle_blue, height=0.2)
        which_NO = MathTex(
            r"\text{Which Neural Operator architecture is better suited for this task?}",
            font_size=text_font,
        )
        which_NO.move_to(2 * UP)
        # which_NO.next_to(NO_ode, 1.5 * DOWN)
        rectangle_which_NO.surround(which_NO)
        articolo_edo = Tex(
            "Edoardo Centofanti, Massimiliano Ghiotto, and Luca F. Pavarino. Learning the Hodgkin–Huxley model with operator learning techniques. Computer Methods in Applied Mechanics and Engineering 432 (2024): 117381.",
            font_size=20,
        )
        articolo_edo.next_to(rectangle_which_NO, 1.5 * DOWN)
        self.play(
            Create(which_NO),
            Create(rectangle_which_NO),
        )
        self.wait(6)
        self.play(Create(articolo_edo))
        self.wait(12)
        rectangle_FNO = Rectangle(color=color_rectangle_FNO, height=0.3)

        FNO = MathTex(
            r"\text{Fourier Neural Operator}",
            font_size=text_font,
        )
        FNO.next_to(articolo_edo, 4 * DOWN)
        arrow_FNO = Arrow(
            start=config.bottom, end=config.bottom + DOWN, color=color_arrow
        )
        arrow_FNO.move_to(0.2 * UP)
        rectangle_FNO.surround(FNO).scale(1.1)

        articolo_fno = Tex(
            "Zongyi Li , Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895 (2020).",
            font_size=20,
        )
        articolo_fno.next_to(rectangle_FNO, 2 * DOWN)

        self.play(
            Create(arrow_FNO), Create(FNO), Create(rectangle_FNO), Create(articolo_fno)
        )
        self.wait(4)
        self.play(
            FadeOut(title_NO),
            FadeOut(which_NO),
            FadeOut(rectangle_which_NO),
            FadeOut(articolo_edo),
            FadeOut(arrow_FNO),
            FadeOut(FNO),
            FadeOut(rectangle_FNO),
            FadeOut(articolo_fno),
        )
        self.wait(1)
        # =========================================================================================##
        ### add text for the title

        title = Text("Fourier Neural Operator", font_size=title_font, color=WHITE)
        title.to_edge(UP).shift(0.2 * UP)
        self.play(Write(title), run_time=2)
        NO_split = MathTex(
            r"\mathcal{G}_{\theta} = ",
            r"\mathcal{Q}",
            r"\circ ",
            r" \mathcal{L}_L\circ ...\circ \mathcal{L}_1",
            r"\circ",
            r"\mathcal{P}",
            font_size=text_font,
        )
        NO_split.move_to(2 * UP)
        self.wait(1)

        self.play(Create(NO_split))
        self.wait(1)

        arrow_P = Arrow(start=config.bottom + DOWN, end=config.bottom, color=color_P)
        arrow_P.next_to(NO_split[5], DOWN)
        arrow_Q = Arrow(start=config.bottom + DOWN, end=config.bottom, color=color_Q)
        arrow_Q.next_to(NO_split[1], DOWN)
        text_Q = Tex("Projection", font_size=text_font, color=color_Q)
        text_Q.next_to(arrow_Q, DOWN)
        text_P = Tex("Lifting", font_size=text_font, color=color_P)
        text_P.next_to(arrow_P, DOWN)

        self.play(
            NO_split[5].animate.set_color(color_P), Create(arrow_P), Create(text_P)
        )
        self.wait(5)
        self.play(
            NO_split[1].animate.set_color(color_Q), Create(arrow_Q), Create(text_Q)
        )
        self.wait(3)
        self.play(FadeOut(arrow_P), FadeOut(text_P), FadeOut(arrow_Q), FadeOut(text_Q))

        brace_L = Brace(NO_split[3], direction=([0, -1, -1]), color=color_L)
        text_L = Tex(
            "Composition of L Integral Operators", font_size=text_font, color=color_L
        )
        text_L.next_to(brace_L, DOWN)
        self.play(
            Write(brace_L),
            Create(text_L),
            NO_split[3].animate.set_color(color_L),
        )
        self.wait(4)
        self.play(FadeOut(brace_L, text_L))
        integral_op = MathTex(
            r"\mathcal{L}_t",
            r"(v)(x)=",
            r"\sigma \Big(",
            r"W_t v(x) + b_t",
            r"+(",
            r" \mathcal{K}_t(a,\theta_t)",
            r"v)(x)" r" \Big)",
            font_size=text_font,
        )
        integral_op.next_to(NO_split, 2 * DOWN)
        self.wait(1)
        # brace_affine = Brace(
        #     integral_op[3], direction=([0, -1, -1]), color=color_affine_transf
        # )
        # text_affine = Tex(
        #     "Pointwise linear transformation",
        #     color=color_affine_transf,
        #     font_size=text_font,
        # )
        # text_affine.next_to(brace_affine, DOWN)

        self.play(
            integral_op[0].animate.set_color(color_L),
            Create(integral_op),
            # Create(brace_affine),
            # Create(text_affine),
            # integral_op[3].animate.set_color(color_affine_transf),
        )
        brace_kernel = Brace(
            integral_op[5], direction=([0, -1, -1]), color=color_affine_kernel
        )
        text_kernel = Tex(
            "Integral kernel operator",
            color=color_affine_kernel,
            font_size=text_font,
        )
        text_kernel.next_to(brace_kernel, DOWN)
        self.play(
            # FadeOut(brace_affine),
            # FadeOut(text_affine),
            Write(brace_kernel),
            Create(text_kernel),
            integral_op[5].animate.set_color(color_affine_kernel),
        )
        self.wait(3)
        # self.play(FadeOut(brace_kernel, text_kernel))
        text_kernel_FNO = MathTex(
            r"\mathcal{K}_t(\theta_t)",
            r"=(",
            r"\kappa_{t,\theta_t}",
            r"*v)(x)",
            r"=\mathcal{F}^{-1}(",
            r"\mathcal{F}(\kappa_{t,\theta_t})(k)",
            r"\cdot \mathcal{F}(v)(k))(x)",
            font_size=text_font,
        ).move_to(DOWN)
        arrow_kernel_function = Arrow(
            start=config.bottom + DOWN, end=config.bottom, color=color_kernel_function
        )
        arrow_kernel_function.next_to(text_kernel_FNO[2], DOWN)
        text_kernel_function = Tex(
            r"kernel function", color=color_kernel_function, font_size=text_font
        )
        text_kernel_function.next_to(arrow_kernel_function, DOWN)

        arrow_fourier = Arrow(
            start=config.bottom + DOWN, end=config.bottom, color=color_fourier
        )
        arrow_fourier.next_to(text_kernel_FNO[5], DOWN)
        text_fourier = Tex(
            r"Parametrized by a Matrix", color=color_fourier, font_size=text_font
        )
        text_fourier.next_to(arrow_fourier, DOWN)

        self.play(
            text_kernel_FNO[0].animate.set_color(color_affine_kernel),
            Create(text_kernel_FNO),
        )
        self.wait(1)
        self.play(
            text_kernel_FNO[2].animate.set_color(color_kernel_function),
            Create(arrow_kernel_function),
            Create(text_kernel_function),
        )
        self.wait(2)

        self.play(
            text_kernel_FNO[5].animate.set_color(color_fourier),
            Create(arrow_fourier),
            Create(text_fourier),
        )
        self.wait(2)

        # text_how_to_train = Tex(
        #     r"Neural Operators are trained in a data-driven method.",
        #     font_size=text_font,
        # )
        # text_how_to_train.next_to(text_kernel_FNO, 3 * DOWN)
        # rectangle_how_to_train = Rectangle(color=color_rectangle_red, height=0.2)
        # rectangle_how_to_train.surround(text_how_to_train)

        # text_dataset = MathTex(
        #     r"\textbf{Dataset}\Longrightarrow\{(a^{(i)},u^{(i)})\}",
        #     font_size=text_font,
        # )
        # text_dataset.next_to(text_how_to_train, 2 * DOWN)

        # text_min = MathTex(
        #     r"\min_{\theta\in\Theta} \ \mathbb{E}_{a\sim\mu}\left[ Loss\left(\mathcal{G}^\dagger(a), \mathcal{G}_\theta(a)\right)\right] \approx  \min_{\theta\in\Theta}  \frac{1}{N} \sum_{i=1}^N \, \frac{\sqrt{\sum_{j=1}^{n_{points}} |u^{(i)}(x_j) - \mathcal{G}_\theta(a^{(i)})(x_j)|^2}}{\sqrt{\sum_{j=1}^{n_{points}} |u^{(i)}(x_j)|^2}}",
        #     font_size=text_font,
        # )
        # text_min.next_to(text_dataset, 2 * DOWN)
        self.play(
            FadeOut(
                arrow_kernel_function, text_kernel_function, arrow_fourier, text_fourier
            ),
            # Create(text_how_to_train),
            # Create(rectangle_how_to_train),
            # Create(text_dataset),
            # Create(text_min),
            FadeOut(NO_split),
            FadeOut(integral_op),
            FadeOut(text_kernel_FNO),
            FadeOut(brace_kernel, text_kernel),
            #     FadeOut(text_how_to_train),
            #     FadeOut(rectangle_how_to_train),
            #     FadeOut(text_dataset),
            #     FadeOut(text_min),
        )

        #### add image
        image = VGroup(ax_I_app, x_label_I_app, y_label_I_app, graph_I_app).move_to(
            ORIGIN
        )
        image.scale(2.5).move_to(0.5 * LEFT)
        text = Text("Take an input", font_size=text_font)
        text.to_edge(RIGHT)
        self.play(Create(text), run_time=1)
        self.play(FadeIn(image), run_time=1)
        self.wait(1)

        #### text
        text1 = Text("Evaluate the function \n on a uniform grid", font_size=text_font)
        text1.to_edge(RIGHT)
        self.play(ReplacementTransform(text, text1), run_time=1)

        self.wait(1)
        ##=========================================================================================##

        #### create the tensor
        tensor_size = (n_points, d_input, 1)  # tensor size
        length = 0.3  # side length of each cube
        spacing = 2 * length  # spacing between cubes
        tensor_group1_rotate = self.createTensor(tensor_size, length, spacing)
        tensor_group1_rotate.move_to(ORIGIN).scale(80 / 100)
        self.play(FadeOut(image), FadeIn(tensor_group1_rotate), run_time=2)
        self.wait(1)
        tensor_size = (d_input, n_points, 1)  # tensor size
        tensor_group1 = self.createTensor(tensor_size, length, spacing)
        tensor_group1.move_to(ORIGIN)
        self.play(ReplacementTransform(tensor_group1_rotate, tensor_group1), run_time=1)
        self.wait(1)

        ##=========================================================================================##
        # Move to left
        self.play(tensor_group1.animate.to_edge(LEFT))

        # Change text
        text2 = Text("Apply the FNO", font_size=text_font).to_edge(RIGHT)
        self.play(ReplacementTransform(text1, text2), run_time=1)
        self.wait(1)

        ##=========================================================================================##

        # Load the image
        image = (
            ImageMobject("FNO.png")
            .scale_to_fit_width(0.45 * config.frame_width)
            .move_to(ORIGIN)
        )
        ##=========================================================================================##

        arrow = Arrow(
            start=tensor_group1.get_right(),
            end=image.get_left(),
            buff=0.4,
            stroke_width=4,  # Adjust the width of the arrow
            tip_length=0.2,  # Adjust the length of the arrow tip
            max_tip_length_to_length_ratio=0.1,  # Adjust the tip length to arrow length ratio
            color=color_arrow,
        )
        self.play(GrowArrow(arrow))

        # Animate the image
        self.play(FadeIn(image), run_time=2)
        self.wait()

        # ##=========================================================================================##

        self.play(FadeOut(text2), run_time=1)

        tensor_group_output = self.createTensor(
            (d_output, n_points, 1), length, spacing
        )
        tensor_group_output.move_to(ORIGIN).scale(0.8).to_edge(RIGHT)
        arrow2 = Arrow(
            start=image.get_right(),
            end=tensor_group_output.get_left(),
            buff=0.4,
            stroke_width=4,  # Adjust the width of the arrow
            tip_length=0.2,  # Adjust the length of the arrow tip
            max_tip_length_to_length_ratio=0.1,  # Adjust the tip length to arrow length ratio
            color=color_arrow,
        )
        self.play(GrowArrow(arrow2))
        self.play(FadeIn(tensor_group_output), run_time=1)

        self.wait(2)
        # ##=========================================================================================##

        self.play(
            FadeOut(arrow),
            FadeOut(arrow2),
            FadeOut(image),
            FadeOut(tensor_group1),
            run_time=1,
        )
        self.play(
            tensor_group_output.animate.move_to(ORIGIN),
            run_time=1,
        )
        self.wait(1)

        #### output
        # axes_group_line = (
        #     VGroup(ax_V, ax_m, ax_h, ax_n)
        #     .arrange_in_grid(
        #         rows=1,
        #         cols=4,
        #         buff=0.8,
        #         flow_order="dr",
        #     )
        #     .scale(axes_scale)
        #     .center()
        # ).move_to(ORIGIN)

        # self.play(
        #     FadeIn(axes_group_line),
        #     Write(VGroup(x_label_m, x_label_n)),
        #     Write(VGroup(y_label_V, y_label_m, y_label_h, y_label_n)),
        #     Create(VGroup(*graph_V, *graph_m, *graph_h, *graph_n)),
        #     run_time=3,
        # )
        # image_output_1 = ImageMobject("v_hh.png")
        image_output_1 = VGroup(ax_V, x_label_V, y_label_V, graph_V).move_to(ORIGIN)
        image_output_1.scale(1.5).to_edge(LEFT)
        image_output_2 = VGroup(ax_m, x_label_m, y_label_m, graph_m).move_to(ORIGIN)
        image_output_2.scale(1.5).next_to(image_output_1, 5 * RIGHT)
        image_output_3 = VGroup(ax_h, x_label_h, y_label_h, graph_h).move_to(ORIGIN)
        image_output_3.scale(1.5).next_to(image_output_2, 5 * RIGHT)
        image_output_4 = VGroup(ax_n, x_label_n, y_label_n, graph_n).move_to(ORIGIN)
        image_output_4.scale(1.5).next_to(image_output_3, 5 * RIGHT)

        # animation with arrange
        # self.play(tensor_group_output.animate.scale(0.8).arrange(RIGHT, buff=0.2))

        tensor_group_output_1_single = self.createTensor(
            (1, n_points, 1), length, spacing
        )
        tensor_group_output_1_single.scale(0.5).move_to(image_output_1.get_center())
        tensor_group_output_2_single = self.createTensor(
            (1, n_points, 1), length, spacing
        )
        tensor_group_output_2_single.scale(0.5).move_to(image_output_2.get_center())
        tensor_group_output_3_single = self.createTensor(
            (1, n_points, 1), length, spacing
        )
        tensor_group_output_3_single.scale(0.5).move_to(image_output_3.get_center())
        tensor_group_output_4_single = self.createTensor(
            (1, n_points, 1), length, spacing
        )
        tensor_group_output_4_single.scale(0.5).move_to(image_output_4.get_center())

        self.play(
            ReplacementTransform(
                tensor_group_output,
                VGroup(
                    tensor_group_output_1_single,
                    tensor_group_output_2_single,
                    tensor_group_output_3_single,
                    tensor_group_output_4_single,
                ),
            ),
            run_time=1,
        )

        self.wait(1)

        self.play(
            FadeOut(tensor_group_output_1_single),
            FadeOut(tensor_group_output_2_single),
            FadeOut(tensor_group_output_3_single),
            FadeOut(tensor_group_output_4_single),
            FadeIn(image_output_1),
            FadeIn(image_output_2),
            FadeIn(image_output_3),
            FadeIn(image_output_4),
            run_time=1,
        )

        self.wait(4)
        self.play(
            FadeOut(title),
            FadeOut(image_output_1, image_output_2, image_output_3, image_output_4),
        )

        title_results = Text("Our Results", font_size=title_font, color=WHITE)
        title_results.to_edge(UP).shift(0.2 * UP)

        rectangle_which_model = Rectangle(color=color_rectangle_blue, height=0.2)
        which_model = MathTex(
            r"\text{Which model do we take into consideration?}",
            font_size=text_font,
        )
        which_model.move_to(2 * UP)
        # which_NO.next_to(NO_ode, 1.5 * DOWN)
        rectangle_which_model.surround(which_model)

        self.play(
            Create(title_results), Create(which_model), Create(rectangle_which_model)
        )
        self.wait(10)

        text_FHN = Tex(r"FitzHugh-Nagumo", font_size=text_font)
        text_HH = Tex(r"Hodgkin-Huxley", font_size=text_font)
        # text_HH.next_to(rectangle_which_model, 2 * DOWN)
        # text_FHN.next_to(text_HH, 4 * LEFT)
        text_ORd = Tex(r"O'Hara-Rudy", font_size=text_font)
        # text_ORd.next_to(text_HH, 8 * RIGHT)
        group = VGroup(text_FHN, text_HH, text_ORd)
        group.arrange(RIGHT, buff=7 / 3)
        group.next_to(rectangle_which_model, 2 * DOWN)

        sep_line_FHN = Line(
            start=(text_FHN.get_right() + text_HH.get_left()) / 2,
            end=(text_FHN.get_right() + text_HH.get_left()) / 2 + 5 * DOWN,
            color=BLACK,
            stroke_width=2,
        )

        sep_line_HH = Line(
            start=(text_HH.get_right() + text_ORd.get_left()) / 2,
            end=(text_HH.get_right() + text_ORd.get_left()) / 2 + 5 * DOWN,
            color=BLACK,
            stroke_width=2,
        )

        FHN_system = MathTex(
            r" &\frac{dV}{dt} = bV(V-\beta)(\delta-V) -cw +I_{app}, \\ &\frac{dw}{dt} = e(V-\gamma w).",
            font_size=22,
        )

        HH_system = MathTex(
            r"& C_m\frac{dV}{dt} + I_{ion}(m,h,n)= I_{app},\\  &\frac{dm}{dt} = \alpha_m(V)(1-m)-\beta_m(V)m,\\ &\frac{dh}{dt} = \alpha_h(V)(1-h)-\beta_h(V)h,\\ & \frac{dn}{dt} = \alpha_n(V)(1-n)-\beta_n(V)n.",
            font_size=22,
        )

        ORd_system = MathTex(
            r"&-C_m \frac{dV}{dt} = -\Big(I_{\Na} + I_{to} + I_{\Cal L} + I_{\Cal \Na } + I_{\Cal \K} + I_{\K r} + I_{\K s} + I_{\K 1} + I_{\Na \Cal } + I_{\Na \K} + I_{\Na b} + I_{\Cal b} +I_{\K b} + I_{p\Cal }\Big) + I_{app},\\",
            r"&\frac{d\zeta}{dt} = \frac{\zeta_{\infty} - \zeta}{\tau_{\zeta}},\ \ \ \  \zeta =  \{m,h_{fast},h_{slow},j,h_{\CaMK,slow},j_{\CaMK},m_L,h_L,h_{L,\CaMK},a,i_{fast},i_{slow},a_{\CaMK}\},\\",
            r"&\frac{dn}{dt} = \alpha_n \cdot k_{+2,n} + n\cdot k_{-2,n},\\ &\frac{d \CaMKtrap}{dt} = \alpha_{\CaMK}\cdot \CaMKbound \cdot (\CaMKbound + \CaMKtrap) - \beta_{\CaMK}\cdot \CaMKtrap, \\ &\frac{d[\Na^+]_i}{dt} = - (I_{\Na} + I_{\Na L} + 3 \cdot I_{\Na \Cal,i} + 3 \cdot I_{\Na \K } + I_{\Na b})\cdot \frac{A_{cap}}{F\cdot v_{myo}} + J_{\diff ,\Na} \cdot \frac{v_{ss}}{v_{myo}},\\ &\frac{d[\Na^+]_{ss}}{dt} = - (I_{\Cal \Na } + 3\cdot I_{\Na \Cal ,ss} )\cdot \frac{A_{cap}}{F\cdot v_{ss}} - J_{\diff ,\Na}, \\ &\frac{d[\K ^+]_{i}}{dt} = - (I_{to} + I_{\K r} + I_{\K s} + I_{\K 1} + I_{\K ur} + I_{app} - 2 \cdot I_{\Na \K } )\cdot \frac{A_{cap}}{F\cdot v_{myo}} + J_{\diff ,\K }\cdot \frac{v_{ss}}{v_{myo}}, \\ &\frac{d[\K ^+]_{ss}}{dt} = - I_{\Cal \K }\cdot \frac{A_{cap}}{F\cdot v_{ss}} - J_{\diff ,\K }, \\ &\frac{d[\Cal ^{2+}]_{i}}{dt} = \beta_{\Cal i}\cdot\Big( - (I_{p\Cal }+I_{\Cal b}-2\cdot I_{\Na \Cal ,i})\cdot \frac{A_{cap}}{2\cdot F \cdot v_{myo}}) - J_{up} \cdot \frac{v_{nsr}}{v_{myo}} + J_{\diff ,\Cal } \cdot \frac{v_{ss}}{v_{myo}}\Big), \\ &\frac{d[\Cal ^{2+}]_{ss}}{dt} = \beta_{\Cal ss}\cdot\Big( - (I_{\Cal L}-2\cdot I_{\Na \Cal ,ss})\cdot \frac{A_{cap}}{2\cdot F \cdot v_{ss}}) + J_{rel} \cdot \frac{v_{jsr}}{v_{ss}} - J_{\diff ,\Cal }\Big), \\ &\frac{d[\Cal ^{2+}]_{nsr}}{dt} = J_{up} - J_{tr}\cdot \frac{v_{jsr}}{v_{nsr}}, \\ &\frac{d[\Cal ^{2+}]_{jsr}}{dt} = \beta_{\Cal jsr} \cdot (J_{tr} - J_{rel}).  \\",
            font_size=10,
        )

        ORd_image = ImageMobject("ohara.jpg").scale(0.4)
        ORd_image.next_to(text_ORd, 2 * DOWN)
        HH_system.next_to(text_HH, 2 * DOWN)
        FHN_system.next_to(text_FHN, 2 * DOWN)
        brace_FHN = Brace(FHN_system, direction=([-1, 0, -1]), color=color_brace)
        brace_HH = Brace(HH_system, direction=([-1, 0, -1]), color=color_brace)
        # ORd_system.next_to(text_ORd, 2 * DOWN)
        self.play(Create(text_FHN), Create(FHN_system), Write(brace_FHN))
        self.wait(3)
        self.play(
            Create(text_HH),
            Create(sep_line_FHN),
            Create(HH_system),
            Write(brace_HH),
        )
        self.wait(3)
        self.play(
            Create(text_ORd),
            Create(sep_line_HH),
            FadeIn(ORd_image),
        )
        self.wait(3)
        FHN_title = Tex("FitzHugh-Nagumo visualization", font_size=text_font)

        FHN_results = ImageMobject("FHN_visualization_no_err.png").scale(0.2)
        FHN_results.next_to(text_FHN, DOWN)
        FHN_title.next_to(FHN_results, UP)
        self.camera.frame.save_state()
        # self.play(Create(text_HH), Create(text_FHN), Create(text_ORd))
        self.play(
            self.camera.auto_zoom([FHN_system], margin=2),
            FadeOut(FHN_system, brace_FHN, text_FHN, sep_line_FHN, brace_HH),
            FadeIn(FHN_results, FHN_title),
        )

        self.wait(22)
        self.play(
            Restore(self.camera.frame),
            FadeOut(FHN_results, FHN_title),
            FadeIn(FHN_system, brace_FHN, text_FHN, sep_line_FHN),
            Write(brace_HH),
        )
        self.wait(1)

        self.camera.frame.save_state()
        HH_title = Tex("Hodgkin-Huxley visualization", font_size=text_font)
        HH_results = ImageMobject("hh_visualization_no_err.png").scale(0.2)
        HH_results.next_to(text_HH, 2 * DOWN)
        HH_title.next_to(HH_results, 2 * UP)
        self.camera.frame.save_state()
        # self.play(Create(text_HH), Create(text_HH), Create(text_ORd))
        self.play(
            self.camera.auto_zoom([HH_system], margin=2),
            FadeOut(
                HH_system,
                brace_HH,
                text_HH,
                sep_line_FHN,
                sep_line_HH,
                FHN_system,
                text_FHN,
                text_ORd,
                ORd_image,
            ),
            FadeIn(HH_results, HH_title),
        )
        self.wait(7)
        self.play(
            Restore(self.camera.frame),
            FadeOut(HH_results, HH_title),
            FadeIn(
                HH_system,
                text_HH,
                sep_line_FHN,
                sep_line_HH,
                FHN_system,
                text_FHN,
                text_ORd,
                ORd_image,
            ),
            Write(brace_HH),
        )
        self.wait(1)
        self.camera.frame.save_state()
        ORd_title = Tex("O'Hara-Rudy visualization", font_size=text_font)

        ORd_results = ImageMobject("all_ord.png").scale(0.4)
        ORd_results.next_to(text_ORd, 1.5 * DOWN)
        ORd_title.next_to(ORd_results, 1.5 * UP)

        # self.play(Create(text_ORd), Create(text_ORd), Create(text_ORd))
        self.play(
            self.camera.auto_zoom([ORd_image], margin=2),
            FadeOut(
                text_ORd,
                sep_line_FHN,
                sep_line_HH,
                FHN_system,
                HH_system,
                text_HH,
                text_FHN,
                text_ORd,
                ORd_image,
            ),
            # FadeIn(ORd_title, brace_ORd),
        )

        ORd_system.scale_to_fit_height(self.camera.frame_height * 0.8).next_to(
            ORd_title, DOWN
        )
        brace_ORd = Brace(ORd_system, direction=([-1, 0, -1]), color=color_brace)

        self.play(
            Create(ORd_system),
            Write(brace_ORd),
            Create(ORd_title),
            ORd_system[1].animate.set_color(PURE_RED),
        )
        self.wait(6)

        self.play(FadeOut(ORd_system, brace_ORd), FadeIn(ORd_results))
        self.wait(6)
        self.play(
            Restore(self.camera.frame),
            FadeOut(ORd_results, ORd_title),
            FadeIn(
                sep_line_FHN,
                sep_line_HH,
                FHN_system,
                text_FHN,
                HH_system,
                text_HH,
                text_ORd,
                ORd_image,
            ),
        )

        self.play(FadeOut(which_model, rectangle_which_model))
        constrained = MathTex(
            r"\text{How the number of trainable parameters influence the results?}",
            font_size=text_font,
        )
        constrained.move_to(2 * UP)
        rectangle_which_model.surround(constrained)
        table_FHN = (
            Table(
                [
                    ["Mode", "N. Parameters", "Train err", "Test err"],
                    ["Con.", "0.57M", "0.35%", "0.86%"],
                    ["Uncon.", "8.69M", "0.31%", "0.87%"],
                ],
                include_outer_lines=True,
                line_config={"stroke_color": BLACK, "stroke_width": 1},
            )
            .scale(0.2)
            .next_to(FHN_system, DOWN)
            .to_edge(0.8 * DOWN)
        )
        table_HH = (
            Table(
                [
                    ["Mode", "N. Parameters", "Train err", "Test err"],
                    ["Con.", "0.63M", "1.31%", "2.71%"],
                    ["Uncon.", "13.44M", "0.93%", "2.34%"],
                ],
                include_outer_lines=True,
                line_config={"stroke_color": BLACK, "stroke_width": 1},
            )
            .scale(0.2)
            .next_to(HH_system, DOWN)
            .to_edge(0.8 * DOWN)
        )
        table_ORd = (
            Table(
                [
                    ["Mode", "N. Parameters", "Train err", "Test err"],
                    ["Con.", "0.51M", "1.31%", "2.42%"],
                    ["Uncon.", "12.4M", "0.88%", "2.19%"],
                ],
                include_outer_lines=True,
                line_config={"stroke_color": BLACK, "stroke_width": 1},
            )
            .scale(0.2)
            .next_to(ORd_system, DOWN)
            .to_edge(0.8 * DOWN)
        )
        self.play(
            FadeOut(nome),
            Create(constrained),
            Create(rectangle_which_model),
            Create(table_FHN),
            Create(table_HH),
            Create(table_ORd),
        )
        self.wait(6)
        image_ending = ImageMobject("ending.jpg").scale_to_fit_width(config.frame_width)

        self.play(FadeIn(image_ending))
        self.wait(20)

    def createTensor(
        self, tensor_size, length, spacing, color=PURE_BLUE, opacity=1, angle=PI / 11
    ):
        tensor_group = VGroup()  # Create a group to hold all the cubes
        for i in range(tensor_size[0]):
            for j in range(tensor_size[1]):
                for k in range(tensor_size[2]):
                    # Create a small cube
                    cube = Square(
                        side_length=length,
                        fill_color=color,
                        fill_opacity=opacity,
                        stroke_color=WHITE,
                        stroke_width=1,
                        sheen_factor=0.5,  # Add sheen for a glossy effect
                        sheen_direction=UL,  # Direction of the sheen effect
                    )
                    # Position the cube based on its indices in the tensor
                    cube.move_to(np.array([i * spacing, j * spacing, k * spacing]))
                    # Add the cube to the group
                    tensor_group.add(cube)
        tensor_group.rotate(-angle, axis=UP)
        return tensor_group


if __name__ == "__main__":
    from manim import config

    # config.background_color = WHITE
    # config.pixel_height = 720
    # config.pixel_width = 1280
    # config.frame_height = 7.0
    # config.frame_width = 14.0
    scene = FNO_architecture_1d()
    scene.render()
