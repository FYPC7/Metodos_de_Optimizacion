from manim import *
import numpy as np

# Configuración global
config.frame_rate = 30
config.pixel_height = 720
config.pixel_width = 1280

class PaperIntroduction(Scene):
    def construct(self):
        # Título principal con animación especial
        title = Text("Optimización sin Derivadas para Funciones de Caja Negra", 
                    font_size=28, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        subtitle = Text("en el Ajuste de Hiperparámetros", 
                       font_size=24, color=WHITE, weight=BOLD)
        subtitle.next_to(title, DOWN, buff=0.3)
        
        # Autor y afiliación con mejor formato
        author = Text("Fiorella Yannet Paredes Coaguila", 
                     font_size=18, color=YELLOW, weight=BOLD)
        author.next_to(subtitle, DOWN, buff=0.8)
        
        affiliation = Text("Universidad Nacional del Altiplano - FINESI", 
                          font_size=14, color=GRAY)
        affiliation.next_to(author, DOWN, buff=0.3)
        
        # Línea decorativa
        line = Line(LEFT*6, RIGHT*6, color=BLUE, stroke_width=2)
        line.next_to(affiliation, DOWN, buff=0.5)
        
        # Animaciones mejoradas
        self.play(
            Write(title, run_time=2),
            FadeIn(subtitle, shift=UP*0.5, run_time=1.5)
        )
        self.play(
            Write(author, run_time=1.2),
            Write(affiliation, run_time=1)
        )
        self.play(Create(line))
        
        self.wait(2)
        self.clear()

class ProblemStatement(Scene):
    def construct(self):
        # Título de sección
        title = Text("El Problema Central", font_size=24, color=RED, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        
        # Contexto del problema
        context = VGroup(
            Text("En Machine Learning, el ajuste de hiperparámetros es crucial", 
                 font_size=14, color=WHITE),
            Text("pero presenta desafíos únicos:", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.2).next_to(title, DOWN, buff=0.5)
        
        self.play(Write(context))
        
        # Caja negra visual mejorada
        black_box = RoundedRectangle(width=4, height=2.5, color=BLACK, 
                                   fill_opacity=0.9, stroke_color=WHITE, 
                                   stroke_width=3, corner_radius=0.2)
        black_box.move_to(ORIGIN)
        
        # Etiqueta de la caja negra
        box_label = Text("Función de Caja Negra", font_size=16, color=WHITE, weight=BOLD)
        box_label.move_to(black_box.get_center() + UP*0.3)
        
        # Símbolo de función
        function_symbol = MathTex(r"f(\lambda) = ?", font_size=18, color=YELLOW)
        function_symbol.move_to(black_box.get_center() + DOWN*0.3)
        
        # Inputs detallados
        input_title = Text("Hiperparámetros", font_size=16, color=GREEN, weight=BOLD)
        input_examples = VGroup(
            MathTex(r"n\_estimators \in \{50, 100, 200, ...\}", font_size=10),
            MathTex(r"max\_depth \in \{3, 5, 7, ...\}", font_size=10),
            MathTex(r"learning\_rate \in [0.01, 0.3]", font_size=10),
            MathTex(r"subsample \in [0.5, 1.0]", font_size=10)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        
        inputs = VGroup(input_title, input_examples).arrange(DOWN, buff=0.3)
        inputs.next_to(black_box, LEFT, buff=1.5)
        
        # Outputs detallados
        output_title = Text("Métrica de Rendimiento", font_size=16, color=RED, weight=BOLD)
        output_formula = MathTex(
            r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}", 
            font_size=12
        )
        output_desc = Text("(Validación Cruzada)", font_size=10, color=GRAY)
        
        outputs = VGroup(output_title, output_formula, output_desc).arrange(DOWN, buff=0.2)
        outputs.next_to(black_box, RIGHT, buff=1.5)
        
        # Flechas con etiquetas
        arrow_in = Arrow(inputs.get_right(), black_box.get_left(), 
                        color=GREEN, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        arrow_out = Arrow(black_box.get_right(), outputs.get_left(), 
                         color=RED, stroke_width=4, max_tip_length_to_length_ratio=0.1)
        
        arrow_in_label = Text("Entrada", font_size=10, color=GREEN)
        arrow_in_label.next_to(arrow_in, UP, buff=0.1)
        
        arrow_out_label = Text("Salida", font_size=10, color=RED)
        arrow_out_label.next_to(arrow_out, UP, buff=0.1)
        
        # Animaciones secuenciales
        self.play(Create(black_box), Write(box_label))
        self.play(Write(function_symbol))
        self.wait(1)
        
        self.play(Write(inputs), Create(arrow_in), Write(arrow_in_label))
        self.wait(1)
        
        self.play(Write(outputs), Create(arrow_out), Write(arrow_out_label))
        self.wait(2)
        
        self.clear()

class ProblemCharacteristics(Scene):
    def construct(self):
        title = Text("Características del Problema", font_size=24, color=ORANGE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Crear iconos y descripciones para cada característica
        characteristics = [
            ("🚫", "No Diferenciable", "Sin gradientes disponibles\nOptimización tradicional no aplicable"),
            ("💰", "Evaluación Costosa", "Entrenamiento + Validación\nTiempo computacional significativo"),
            ("🎲", "Ruido Estocástico", "Variabilidad en resultados\nDebido a particiones aleatorias"),
            ("🔀", "Espacio Mixto", "Variables continuas y discretas\nRestricción de dominio compleja")
        ]
        
        char_groups = VGroup()
        
        for i, (icon, title_text, description) in enumerate(characteristics):
            # Crear caja para cada característica
            box = RoundedRectangle(width=2.8, height=2.2, color=BLUE, 
                                 stroke_width=2, corner_radius=0.1)
            
            # Icono
            icon_text = Text(icon, font_size=30)
            icon_text.move_to(box.get_center() + UP*0.6)
            
            # Título
            char_title = Text(title_text, font_size=12, color=YELLOW, weight=BOLD)
            char_title.move_to(box.get_center() + UP*0.1)
            
            # Descripción
            char_desc = Text(description, font_size=9, color=WHITE)
            char_desc.move_to(box.get_center() + DOWN*0.5)
            
            char_group = VGroup(box, icon_text, char_title, char_desc)
            char_groups.add(char_group)
        
        # Organizar en cuadrícula 2x2
        char_groups.arrange_in_grid(rows=2, cols=2, buff=0.5)
        char_groups.move_to(ORIGIN)
        
        # Animación de aparición
        for i, char_group in enumerate(char_groups):
            self.play(FadeIn(char_group, shift=UP*0.3), run_time=0.8)
            self.wait(0.5)
        
        # Conclusión
        conclusion = Text("Estas características justifican el uso de", font_size=16, color=WHITE)
        conclusion_highlight = Text("Optimización sin Derivadas", font_size=16, color=YELLOW, weight=BOLD)
        
        conclusion_group = VGroup(conclusion, conclusion_highlight).arrange(DOWN, buff=0.2)
        conclusion_group.to_edge(DOWN, buff=0.5)
        
        self.play(Write(conclusion_group))
        self.wait(3)

class WhyDerivativeFree(Scene):
    def construct(self):
        title = Text("¿Por qué Optimización sin Derivadas?", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Comparación visual
        # Lado izquierdo: Optimización tradicional
        traditional_title = Text("Optimización Tradicional", font_size=16, color=RED)
        traditional_title.move_to(LEFT*3 + UP*2)
        
        # Representación de función suave
        axes_trad = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1],
            x_length=3, y_length=2,
            axis_config={"include_numbers": False, "stroke_width": 2}
        ).move_to(LEFT*3 + UP*0.5)
        
        smooth_func = axes_trad.plot(lambda x: x**2, color=GREEN, stroke_width=3)
        gradient_arrow = Arrow(
            axes_trad.c2p(1, 1), axes_trad.c2p(0.5, 0.25),
            color=BLUE, stroke_width=3
        )
        
        trad_props = VGroup(
            Text("✓ Función suave", font_size=10, color=GREEN),
            Text("✓ Gradiente disponible", font_size=10, color=GREEN),
            Text("✓ Convergencia rápida", font_size=10, color=GREEN),
            Text("✗ No aplicable aquí", font_size=10, color=RED)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        trad_props.next_to(axes_trad, DOWN, buff=0.3)
        
        # Lado derecho: Optimización sin derivadas
        df_title = Text("Optimización sin Derivadas", font_size=16, color=GREEN)
        df_title.move_to(RIGHT*3 + UP*2)
        
        # Representación de función con ruido
        axes_df = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1],
            x_length=3, y_length=2,
            axis_config={"include_numbers": False, "stroke_width": 2}
        ).move_to(RIGHT*3 + UP*0.5)
        
        # Función con ruido
        x_vals = np.linspace(-2, 2, 50)
        y_vals = x_vals**2 + 0.2 * np.random.randn(50)
        noisy_points = VGroup(*[
            Dot(axes_df.c2p(x, y), radius=0.02, color=YELLOW)
            for x, y in zip(x_vals, y_vals)
        ])
        
        df_props = VGroup(
            Text("✓ Maneja ruido", font_size=10, color=GREEN),
            Text("✓ No requiere gradientes", font_size=10, color=GREEN),
            Text("✓ Espacio mixto", font_size=10, color=GREEN),
            Text("✓ Robusto", font_size=10, color=GREEN)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        df_props.next_to(axes_df, DOWN, buff=0.3)
        
        # Animaciones
        self.play(
            Write(traditional_title),
            Write(df_title)
        )
        
        self.play(
            Create(axes_trad),
            Create(axes_df)
        )
        
        self.play(
            Create(smooth_func),
            Create(gradient_arrow),
            Create(noisy_points)
        )
        
        self.play(
            Write(trad_props),
            Write(df_props)
        )
        
        # Línea divisoria
        divider = Line(UP*3, DOWN*3, color=WHITE, stroke_width=1)
        divider.move_to(ORIGIN)
        
        self.play(Create(divider))
        self.wait(3)

class MethodsOverview(Scene):
    def construct(self):
        title = Text("Métodos de Optimización sin Derivadas", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        
        self.play(Write(title))
        
        # Clasificación de métodos
        classification = VGroup(
            Text("Clasificación de Métodos:", font_size=16, color=YELLOW),
            Text("• Bayesianos: Usan modelos probabilísticos", font_size=12, color=WHITE),
            Text("• Evolutivos: Inspirados en evolución biológica", font_size=12, color=WHITE),
            Text("• Estocásticos: Basados en muestreo aleatorio", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        classification.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(classification))
        self.wait(2)
        self.clear()

class MethodCards(Scene):
    def construct(self):
        title = Text("Métodos Evaluados", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        
        self.play(Write(title))
        
        # Información detallada de cada método
        methods_info = [
            {
                "name": "TPE",
                "full_name": "Tree-structured\nParzen Estimator",
                "category": "Bayesiano",
                "color": BLUE,
                "description": "• Modela P(x|y) en lugar de P(y|x)\n• Usa árboles de decisión\n• Eficiente para espacios mixtos",
                "advantages": "✓ Rápida convergencia\n✓ Maneja categorías\n✓ Escalable"
            },
            {
                "name": "Random\nSearch",
                "full_name": "Búsqueda\nAleatoria",
                "category": "Estocástico",
                "color": GREEN,
                "description": "• Muestreo uniforme aleatorio\n• Baseline fundamental\n• Paralelizable",
                "advantages": "✓ Simple\n✓ Robusto\n✓ Sin parámetros"
            },
            {
                "name": "CMA-ES",
                "full_name": "Covariance Matrix\nAdaptation",
                "category": "Evolutivo",
                "color": RED,
                "description": "• Adaptación de covarianza\n• Estrategia evolutiva\n• Optimización continua",
                "advantages": "✓ Invariante rotacional\n✓ Autoadaptativo\n✓ Robusto"
            },
            {
                "name": "QMC",
                "full_name": "Quasi-Monte\nCarlo",
                "category": "Determinístico",
                "color": PURPLE,
                "description": "• Secuencias de baja discrepancia\n• Cobertura uniforme\n• Determinístico",
                "advantages": "✓ Reproducible\n✓ Cobertura uniforme\n✓ Eficiente"
            }
        ]
        
        # Crear tarjetas para cada método
        cards = VGroup()
        
        for i, method_info in enumerate(methods_info):
            # Tarjeta principal
            card = RoundedRectangle(
                width=2.8, height=3.5, 
                color=method_info["color"],
                stroke_width=2, 
                corner_radius=0.1
            )
            
            # Nombre del método
            name_text = Text(method_info["name"], font_size=14, color=method_info["color"], weight=BOLD)
            name_text.move_to(card.get_center() + UP*1.4)
            
            # Nombre completo
            full_name = Text(method_info["full_name"], font_size=9, color=WHITE)
            full_name.move_to(card.get_center() + UP*1.0)
            
            # Categoría
            category_badge = RoundedRectangle(
                width=1.8, height=0.3,
                color=method_info["color"],
                fill_opacity=0.3,
                stroke_width=1
            )
            category_text = Text(method_info["category"], font_size=8, color=method_info["color"])
            category_group = VGroup(category_badge, category_text)
            category_group.move_to(card.get_center() + UP*0.6)
            
            # Descripción
            description = Text(method_info["description"], font_size=8, color=WHITE)
            description.move_to(card.get_center() + UP*0.1)
            
            # Ventajas
            advantages = Text(method_info["advantages"], font_size=8, color=GREEN)
            advantages.move_to(card.get_center() + DOWN*0.8)
            
            # Separador
            separator = Line(
                card.get_center() + LEFT*1.2 + DOWN*0.4,
                card.get_center() + RIGHT*1.2 + DOWN*0.4,
                color=GRAY, stroke_width=1
            )
            
            card_group = VGroup(card, name_text, full_name, category_group, 
                              description, separator, advantages)
            cards.add(card_group)
        
        # Organizar en fila
        cards.arrange(RIGHT, buff=0.3)
        cards.next_to(title, DOWN, buff=0.5)
        
        # Animación de aparición
        for i, card in enumerate(cards):
            self.play(FadeIn(card, shift=UP*0.5), run_time=0.8)
            self.wait(0.3)
        
        self.wait(2)

class MethodDetailsTPE(Scene):
    def construct(self):
        title = Text("TPE: Tree-structured Parzen Estimator", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Concepto clave
        key_concept = Text("Concepto Clave: Inversión del Modelo Tradicional", 
                          font_size=16, color=YELLOW, weight=BOLD)
        key_concept.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(key_concept))
        
        # Comparación tradicional vs TPE
        # Lado izquierdo: Método tradicional
        trad_title = Text("Método Tradicional", font_size=14, color=RED)
        trad_title.move_to(LEFT*3 + UP*1.5)
        
        trad_formula = MathTex(r"P(y|x)", font_size=16, color=RED)
        trad_formula.next_to(trad_title, DOWN, buff=0.3)
        
        trad_desc = Text("Modela rendimiento\ngiven configuración", font_size=10, color=WHITE)
        trad_desc.next_to(trad_formula, DOWN, buff=0.3)
        
        # Lado derecho: TPE
        tpe_title = Text("TPE", font_size=14, color=BLUE)
        tpe_title.move_to(RIGHT*3 + UP*1.5)
        
        tpe_formula = MathTex(r"P(x|y)", font_size=16, color=BLUE)
        tpe_formula.next_to(tpe_title, DOWN, buff=0.3)
        
        tpe_desc = Text("Modela configuración\ngiven rendimiento", font_size=10, color=WHITE)
        tpe_desc.next_to(tpe_formula, DOWN, buff=0.3)
        
        # Flecha de conversión
        arrow = Arrow(LEFT*1, RIGHT*1, color=YELLOW, stroke_width=3)
        arrow.move_to(ORIGIN + UP*0.5)
        
        conversion_text = Text("Teorema de Bayes", font_size=12, color=YELLOW)
        conversion_text.next_to(arrow, DOWN, buff=0.2)
        
        self.play(
            Write(trad_title), Write(trad_formula), Write(trad_desc),
            Write(tpe_title), Write(tpe_formula), Write(tpe_desc)
        )
        
        self.play(Create(arrow), Write(conversion_text))
        
        # Algoritmo TPE
        algorithm_title = Text("Algoritmo TPE", font_size=16, color=YELLOW, weight=BOLD)
        algorithm_title.move_to(DOWN*1.5)
        
        algorithm_steps = VGroup(
            Text("1. Separar observaciones en 'buenas' y 'malas'", font_size=12, color=WHITE),
            Text("2. Construir densidades l(x) y g(x)", font_size=12, color=WHITE),
            Text("3. Maximizar l(x)/g(x) para encontrar candidatos", font_size=12, color=WHITE),
            Text("4. Evaluar candidatos y actualizar", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        algorithm_steps.next_to(algorithm_title, DOWN, buff=0.3)
        
        self.play(Write(algorithm_title))
        
        for step in algorithm_steps:
            self.play(Write(step), run_time=0.8)
            self.wait(0.3)
        
        self.wait(2)

class MethodDetailsCMAES(Scene):
    def construct(self):
        title = Text("CMA-ES: Covariance Matrix Adaptation", font_size=24, color=RED, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Concepto principal
        concept = Text("Estrategia Evolutiva con Adaptación de Covarianza", 
                      font_size=16, color=YELLOW)
        concept.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(concept))
        
        # Visualización conceptual
        # Crear elipse que representa la distribución
        ellipse = Ellipse(width=3, height=1.5, color=RED, stroke_width=2)
        ellipse.move_to(ORIGIN)
        
        # Puntos de muestra
        sample_points = VGroup(*[
            Dot(ellipse.point_from_proportion(i/10), radius=0.05, color=YELLOW)
            for i in range(10)
        ])
        
        # Flechas que muestran adaptación
        adaptation_arrows = VGroup(*[
            Arrow(ellipse.get_center(), point.get_center(), 
                 color=BLUE, stroke_width=2, max_tip_length_to_length_ratio=0.2)
            for point in sample_points[:5]
        ])
        
        self.play(Create(ellipse))
        self.play(Create(sample_points))
        self.play(Create(adaptation_arrows))
        
        # Características clave
        features = VGroup(
            Text("Características Clave:", font_size=14, color=YELLOW, weight=BOLD),
            Text("• Adapta matriz de covarianza automáticamente", font_size=11, color=WHITE),
            Text("• Invariante a rotaciones del problema", font_size=11, color=WHITE),
            Text("• Controla tamaño de paso σ", font_size=11, color=WHITE),
            Text("• Usa evolución (μ, λ)", font_size=11, color=WHITE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        features.to_edge(DOWN, buff=0.5)
        
        self.play(Write(features))
        
        # Fórmula principal
        formula = MathTex(
            r"x_{k+1} = x_k + \sigma \cdot \mathcal{N}(0, C)",
            font_size=14
        )
        formula.next_to(concept, DOWN, buff=0.8)
        
        self.play(Write(formula))
        self.wait(3)

class MethodDetailsQMC(Scene):
    def construct(self):
        title = Text("QMC: Quasi-Monte Carlo", font_size=24, color=PURPLE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Comparación Monte Carlo vs Quasi-Monte Carlo
        comparison_title = Text("Monte Carlo vs Quasi-Monte Carlo", 
                              font_size=16, color=YELLOW)
        comparison_title.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(comparison_title))
        
        # Lado izquierdo: Monte Carlo
        mc_title = Text("Monte Carlo", font_size=14, color=RED)
        mc_title.move_to(LEFT*3 + UP*1)
        
        # Crear puntos aleatorios
        np.random.seed(42)
        mc_points = VGroup(*[
            Dot([np.random.uniform(-2, 2), np.random.uniform(-1, 1), 0], 
                radius=0.03, color=RED)
            for _ in range(50)
        ])
        
        mc_desc = Text("Puntos aleatorios\nDistribución irregular", 
                      font_size=10, color=WHITE)
        mc_desc.next_to(mc_points, DOWN, buff=0.3)
        
        # Lado derecho: Quasi-Monte Carlo
        qmc_title = Text("Quasi-Monte Carlo", font_size=14, color=PURPLE)
        qmc_title.move_to(RIGHT*3 + UP*1)
        
        # Crear secuencia de baja discrepancia (simulada)
        qmc_points = VGroup()
        for i in range(50):
            x = -2 + 4 * ((i * 0.618033988749895) % 1)  # Golden ratio
            y = -1 + 2 * ((i * 0.414213562373095) % 1)  # sqrt(2) - 1
            qmc_points.add(Dot([x, y, 0], radius=0.03, color=PURPLE))
        
        qmc_desc = Text("Secuencia determinística\nCobertura uniforme", 
                       font_size=10, color=WHITE)
        qmc_desc.next_to(qmc_points, DOWN, buff=0.3)
        
        # Animaciones
        self.play(Write(mc_title), Write(qmc_title))
        self.play(Create(mc_points), Create(qmc_points))
        self.play(Write(mc_desc), Write(qmc_desc))
        
        # Ventajas del QMC
        advantages = VGroup(
            Text("Ventajas del QMC:", font_size=14, color=YELLOW, weight=BOLD),
            Text("• Mejor convergencia teórica: O(1/N) vs O(1/√N)", font_size=11, color=GREEN),
            Text("• Cobertura uniforme del espacio", font_size=11, color=GREEN),
            Text("• Determinístico y reproducible", font_size=11, color=GREEN),
            Text("• Eficiente para espacios de alta dimensión", font_size=11, color=GREEN)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        advantages.to_edge(DOWN, buff=0.5)
        
        self.play(Write(advantages))
        self.wait(3)

class MethodsComparison(Scene):
    def construct(self):
        title = Text("Comparación de Métodos", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Crear tabla comparativa
        comparison_data = [
            ["Criterio", "TPE", "Random Search", "CMA-ES", "QMC"],
            ["Tipo", "Bayesiano", "Estocástico", "Evolutivo", "Determinístico"],
            ["Espacio Mixto", "✓", "✓", "✗", "✓"],
            ["Velocidad", "Rápida", "Media", "Lenta", "Rápida"],
            ["Memoria", "Baja", "Muy Baja", "Alta", "Baja"],
            ["Paralelizable", "Parcial", "✓", "✓", "✓"],
            ["Reproducible", "✗", "✗", "✗", "✓"]
        ]
        
        # Crear tabla visual
        table = Table(
            comparison_data,
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": WHITE}
        ).scale(0.6)
        
        # Colorear elementos
        table.get_rows()[0].set_color(YELLOW)  # Headers
        table.get_columns()[1].set_color(BLUE)  # TPE column
        
        # Posicionar tabla
        table.move_to(ORIGIN)
        
        self.play(Create(table))
        
        # Destacar fortalezas de cada método
        strengths = VGroup(
            Text("Fortalezas Clave:", font_size=14, color=YELLOW, weight=BOLD),
            Text("• TPE: Mejor para espacios mixtos y convergencia rápida", font_size=11, color=BLUE),
            Text("• Random Search: Baseline simple y confiable", font_size=11, color=GREEN),
            Text("• CMA-ES: Excelente para espacios continuos", font_size=11, color=RED),
            Text("• QMC: Cobertura uniforme y reproducibilidad", font_size=11, color=PURPLE)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        strengths.to_edge(DOWN, buff=0.3)
        
        self.play(Write(strengths))
        self.wait(3)

class TPEExplanation(Scene):
    def construct(self):
        title = Text("Tree-structured Parzen Estimator (TPE)", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Concepto principal
        concept = Text("Optimización Bayesiana Secuencial", font_size=20, color=YELLOW).next_to(title, DOWN)
        self.play(Write(concept))
        
        # Proceso TPE
        step1 = Text("1. Evaluar configuraciones iniciales", font_size=14, color=WHITE)
        step2 = Text("2. Separar observaciones: buenas vs malas", font_size=14, color=WHITE)
        step3 = Text("3. Construir modelos probabilísticos", font_size=14, color=WHITE)
        step4 = Text("4. Maximizar Expected Improvement", font_size=14, color=WHITE)
        step5 = Text("5. Seleccionar siguiente configuración", font_size=14, color=WHITE)
        
        steps = VGroup(step1, step2, step3, step4, step5).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        steps.next_to(concept, DOWN, buff=0.8)
        
        # Animación de pasos
        for step in steps:
            self.play(Write(step), run_time=1)
            self.wait(0.5)
        
        # Fórmula matemática
        formula_title = Text("Expected Improvement:", font_size=16, color=YELLOW).to_edge(DOWN, buff=2)
        formula = MathTex(
            r"EI(\lambda) = \int_{-\infty}^{f^*} (f^* - f) \cdot p(\lambda | f) \, df",
            font_size=14
        ).next_to(formula_title, DOWN)
        
        self.play(Write(formula_title), Write(formula))
        self.wait(3)

class DatasetIntroduction(Scene):
    def construct(self):
        title = Text("Conjunto de Datos: House Prices", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Logo/Icono de casa
        house_icon = Text("🏠", font_size=60)
        house_icon.move_to(UP*1.5)
        
        # Información básica del dataset
        dataset_info = VGroup(
            Text("Competencia Kaggle: House Prices - Advanced Regression Techniques", 
                 font_size=14, color=YELLOW),
            Text("Predicción de precios de viviendas en Ames, Iowa", 
                 font_size=12, color=WHITE),
            Text("Problema de regresión con características mixtas", 
                 font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.3)
        dataset_info.next_to(house_icon, DOWN, buff=0.5)
        
        self.play(Write(house_icon))
        self.play(Write(dataset_info))
        
        # Características del problema
        problem_characteristics = VGroup(
            Text("¿Por qué es ideal para optimización sin derivadas?", 
                 font_size=14, color=RED, weight=BOLD),
            Text("• Función objetivo ruidosa (validación cruzada)", font_size=12, color=WHITE),
            Text("• Evaluación costosa (entrenamiento de modelos)", font_size=12, color=WHITE),
            Text("• Espacio de hiperparámetros mixto", font_size=12, color=WHITE),
            Text("• No hay gradientes disponibles", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        problem_characteristics.to_edge(DOWN, buff=0.5)
        
        self.play(Write(problem_characteristics))
        self.wait(3)

class DatasetStatistics(Scene):
    def construct(self):
        title = Text("Estadísticas del Dataset", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Caja principal de estadísticas
        stats_box = RoundedRectangle(width=10, height=5, color=BLUE, 
                                   stroke_width=2, corner_radius=0.2)
        stats_box.move_to(ORIGIN)
        
        # Estadísticas organizadas
        basic_stats = VGroup(
            Text("📊 Estadísticas Básicas", font_size=16, color=YELLOW, weight=BOLD),
            Text("• Observaciones: 1,460", font_size=14, color=WHITE),
            Text("• Características: 81 (37 numéricas + 43 categóricas + 1 ID)", font_size=14, color=WHITE),
            Text("• Variable objetivo: SalePrice (precio de venta)", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        basic_stats.move_to(stats_box.get_center() + UP*1.5)
        
        price_stats = VGroup(
            Text("💰 Estadísticas de Precios", font_size=16, color=GREEN, weight=BOLD),
            Text("• Media: $180,921", font_size=14, color=WHITE),
            Text("• Mediana: $163,000", font_size=14, color=WHITE),
            Text("• Desviación estándar: $79,443", font_size=14, color=WHITE),
            Text("• Rango: $34,900 - $755,000", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        price_stats.move_to(stats_box.get_center() + DOWN*0.5)
        
        self.play(Create(stats_box))
        self.play(Write(basic_stats))
        self.play(Write(price_stats))
        
        # Distribución visual simulada
        dist_title = Text("Distribución de Precios", font_size=12, color=YELLOW)
        dist_title.next_to(stats_box, DOWN, buff=0.3)
        
        # Crear histograma simulado
        bars = VGroup()
        bar_heights = [0.5, 1.2, 2.0, 2.5, 2.8, 2.3, 1.8, 1.2, 0.8, 0.4]
        
        for i, height in enumerate(bar_heights):
            bar = Rectangle(width=0.4, height=height, color=BLUE, fill_opacity=0.7)
            bar.next_to(dist_title, DOWN, buff=0.2)
            bar.shift(RIGHT * (i - 4.5) * 0.5)
            bars.add(bar)
        
        self.play(Write(dist_title))
        self.play(Create(bars))
        self.wait(2)

class FeatureAnalysis(Scene):
    def construct(self):
        title = Text("Análisis de Características", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Categorías de características
        categories = [
            {
                "name": "Físicas de la Casa",
                "color": GREEN,
                "icon": "🏗️",
                "features": ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"],
                "description": "Dimensiones y espacios"
            },
            {
                "name": "Calidad y Condición",
                "color": YELLOW,
                "icon": "⭐",
                "features": ["OverallQual", "OverallCond", "ExterQual", "KitchenQual"],
                "description": "Evaluaciones cualitativas"
            },
            {
                "name": "Ubicación y Zona",
                "color": RED,
                "icon": "📍",
                "features": ["Neighborhood", "MSZoning", "LotConfig", "LandContour"],
                "description": "Características geográficas"
            },
            {
                "name": "Antigüedad y Remodelación",
                "color": PURPLE,
                "icon": "🕐",
                "features": ["YearBuilt", "YearRemodAdd", "YrSold", "MoSold"],
                "description": "Aspectos temporales"
            }
        ]
        
        # Crear cajas para cada categoría
        category_boxes = VGroup()
        
        for i, cat in enumerate(categories):
            box = RoundedRectangle(width=2.5, height=2.8, color=cat["color"], 
                                 stroke_width=2, corner_radius=0.1)
            
            # Icono
            icon = Text(cat["icon"], font_size=24)
            icon.move_to(box.get_center() + UP*1.0)
            
            # Nombre de categoría
            name = Text(cat["name"], font_size=11, color=cat["color"], weight=BOLD)
            name.move_to(box.get_center() + UP*0.6)
            
            # Ejemplos de características
            features_text = Text("\n".join(cat["features"][:3]), font_size=8, color=WHITE)
            features_text.move_to(box.get_center() + UP*0.1)
            
            # Descripción
            desc = Text(cat["description"], font_size=8, color=GRAY)
            desc.move_to(box.get_center() + DOWN*0.8)
            
            category_box = VGroup(box, icon, name, features_text, desc)
            category_boxes.add(category_box)
        
        # Organizar en cuadrícula 2x2
        category_boxes.arrange_in_grid(rows=2, cols=2, buff=0.5)
        category_boxes.move_to(ORIGIN)
        
        # Animación de aparición
        for box in category_boxes:
            self.play(FadeIn(box, shift=UP*0.3), run_time=0.8)
        
        # Información adicional
        additional_info = VGroup(
            Text("Desafíos del Dataset:", font_size=14, color=RED, weight=BOLD),
            Text("• Valores faltantes en múltiples características", font_size=11, color=WHITE),
            Text("• Variables categóricas con muchas categorías", font_size=11, color=WHITE),
            Text("• Correlaciones complejas entre características", font_size=11, color=WHITE),
            Text("• Outliers en precios y características físicas", font_size=11, color=WHITE)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        additional_info.to_edge(DOWN, buff=0.3)
        
        self.play(Write(additional_info))
        self.wait(3)

class ModelSelection(Scene):
    def construct(self):
        title = Text("Selección de Modelos", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Justificación de modelos elegidos
        justification = VGroup(
            Text("¿Por qué Random Forest y XGBoost?", font_size=16, color=YELLOW, weight=BOLD),
            Text("• Robustos a outliers y valores faltantes", font_size=12, color=WHITE),
            Text("• Manejan características mixtas naturalmente", font_size=12, color=WHITE),
            Text("• Múltiples hiperparámetros importantes", font_size=12, color=WHITE),
            Text("• Ampliamente usados en problemas reales", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        justification.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(justification))
        
        # Comparación de modelos
        model_comparison = VGroup()
        
        # Random Forest
        rf_box = RoundedRectangle(width=5, height=3, color=GREEN, 
                                stroke_width=2, corner_radius=0.2)
        rf_box.move_to(LEFT*3)
        
        rf_content = VGroup(
            Text("🌲 Random Forest", font_size=16, color=GREEN, weight=BOLD),
            Text("Hiperparámetros clave:", font_size=12, color=YELLOW),
            Text("• n_estimators: [50, 500]", font_size=10, color=WHITE),
            Text("• max_depth: [3, 20]", font_size=10, color=WHITE),
            Text("• min_samples_split: [2, 20]", font_size=10, color=WHITE),
            Text("• min_samples_leaf: [1, 10]", font_size=10, color=WHITE),
            Text("• max_features: ['auto', 'sqrt', 'log2']", font_size=10, color=WHITE)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        rf_content.move_to(rf_box.get_center())
        
        # XGBoost
        xgb_box = RoundedRectangle(width=5, height=3, color=ORANGE, 
                                 stroke_width=2, corner_radius=0.2)
        xgb_box.move_to(RIGHT*3)
        
        xgb_content = VGroup(
            Text("🚀 XGBoost", font_size=16, color=ORANGE, weight=BOLD),
            Text("Hiperparámetros clave:", font_size=12, color=YELLOW),
            Text("• n_estimators: [50, 500]", font_size=10, color=WHITE),
            Text("• max_depth: [3, 10]", font_size=10, color=WHITE),
            Text("• learning_rate: [0.01, 0.3]", font_size=10, color=WHITE),
            Text("• subsample: [0.5, 1.0]", font_size=10, color=WHITE),
            Text("• colsample_bytree: [0.5, 1.0]", font_size=10, color=WHITE)
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        xgb_content.move_to(xgb_box.get_center())
        
        self.play(Create(rf_box), Create(xgb_box))
        self.play(Write(rf_content), Write(xgb_content))
        
        # Características del espacio de búsqueda
        search_space = VGroup(
            Text("Características del Espacio de Búsqueda:", font_size=14, color=RED, weight=BOLD),
            Text("• Variables continuas (learning_rate, subsample)", font_size=11, color=WHITE),
            Text("• Variables discretas (n_estimators, max_depth)", font_size=11, color=WHITE),
            Text("• Variables categóricas (max_features)", font_size=11, color=WHITE),
            Text("• ~10⁶ configuraciones posibles por modelo", font_size=11, color=YELLOW)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        search_space.to_edge(DOWN, buff=0.3)
        
        self.play(Write(search_space))
        self.wait(3)

class EvaluationMetric(Scene):
    def construct(self):
        title = Text("Métrica de Evaluación: RMSE", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Definición de RMSE
        rmse_definition = VGroup(
            Text("Root Mean Square Error (RMSE)", font_size=16, color=YELLOW, weight=BOLD),
            Text("Mide la diferencia promedio entre valores reales y predichos", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.3)
        rmse_definition.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(rmse_definition))
        
        # Fórmula principal
        formula_box = RoundedRectangle(width=8, height=2, color=BLUE, 
                                     stroke_width=2, corner_radius=0.2)
        formula_box.move_to(ORIGIN)
        
        rmse_formula = MathTex(
            r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}",
            font_size=20, color=WHITE
        )
        rmse_formula.move_to(formula_box.get_center())
        
        self.play(Create(formula_box))
        self.play(Write(rmse_formula))
        
        # Explicación de componentes
        components = VGroup(
            Text("Componentes:", font_size=14, color=YELLOW, weight=BOLD),
            MathTex(r"y_i", font_size=12, color=GREEN).next_to(Text("= Precio real de la casa i", font_size=12, color=WHITE), LEFT),
            MathTex(r"\hat{y}_i", font_size=12, color=RED).next_to(Text("= Precio predicho de la casa i", font_size=12, color=WHITE), LEFT),
            MathTex(r"n", font_size=12, color=BLUE).next_to(Text("= Número total de observaciones", font_size=12, color=WHITE), LEFT)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        components.next_to(formula_box, DOWN, buff=0.5)
        
        # Crear las explicaciones completas
        explanations = VGroup(
            Text("Componentes:", font_size=14, color=YELLOW, weight=BOLD),
            VGroup(MathTex(r"y_i", font_size=12, color=GREEN), Text("= Precio real de la casa i", font_size=12, color=WHITE)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"\hat{y}_i", font_size=12, color=RED), Text("= Precio predicho de la casa i", font_size=12, color=WHITE)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"n", font_size=12, color=BLUE), Text("= Número total de observaciones", font_size=12, color=WHITE)).arrange(RIGHT, buff=0.2)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        explanations.next_to(formula_box, DOWN, buff=0.5)
        
        self.play(Write(explanations))
        
        # Propiedades del RMSE
        properties = VGroup(
            Text("Propiedades del RMSE:", font_size=14, color=RED, weight=BOLD),
            Text("• Unidades: Mismas que la variable objetivo ($)", font_size=12, color=WHITE),
            Text("• Sensible a outliers (penaliza errores grandes)", font_size=12, color=WHITE),
            Text("• Siempre no negativo (0 = predicción perfecta)", font_size=12, color=WHITE),
            Text("• Métrica estándar en regresión", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        properties.to_edge(DOWN, buff=0.3)
        
        self.play(Write(properties))
        self.wait(3)

class CrossValidation(Scene):
    def construct(self):
        title = Text("Validación Cruzada de 5 Pliegues", font_size=24, color=BLUE, weight=BOLD)
        title.to_edge(UP)
        
        self.play(Write(title))
        
        # Explicación conceptual
        concept = VGroup(
            Text("Técnica para estimar rendimiento del modelo", font_size=14, color=YELLOW),
            Text("Divide datos en 5 partes, entrena en 4 y evalúa en 1", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.3)
        concept.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(concept))
        
        # Visualización de CV
        cv_title = Text("Proceso de Validación Cruzada", font_size=16, color=GREEN, weight=BOLD)
        cv_title.move_to(UP*1)
        
        # Crear 5 rectángulos para representar los pliegues
        folds = VGroup()
        colors = [BLUE, GREEN, RED, YELLOW, PURPLE]
        
        for i in range(5):
            fold = Rectangle(width=1.5, height=0.5, color=colors[i], 
                           fill_opacity=0.7, stroke_width=2)
            fold_label = Text(f"Fold {i+1}", font_size=10, color=WHITE)
            fold_label.move_to(fold.get_center())
            
            fold_group = VGroup(fold, fold_label)
            folds.add(fold_group)
        
        folds.arrange(RIGHT, buff=0.1)
        folds.move_to(ORIGIN)
        
        self.play(Write(cv_title))
        self.play(Create(folds))
        
        # Mostrar proceso iterativo
        iterations = VGroup()
        
        for i in range(5):
            iteration_text = Text(f"Iteración {i+1}:", font_size=12, color=WHITE)
            train_text = Text("Entrenamiento:", font_size=10, color=GREEN)
            test_text = Text("Prueba:", font_size=10, color=RED)
            
            # Crear mini-representación
            mini_folds = VGroup()
            for j in range(5):
                mini_fold = Rectangle(width=0.3, height=0.2, 
                                    color=GREEN if j != i else RED,
                                    fill_opacity=0.7)
                mini_folds.add(mini_fold)
            
            mini_folds.arrange(RIGHT, buff=0.05)
            
            iteration_group = VGroup(iteration_text, train_text, test_text, mini_folds)
            iteration_group.arrange(DOWN, buff=0.1)
            iterations.add(iteration_group)
        
        iterations.arrange(DOWN, buff=0.3)
        iterations.next_to(folds, DOWN, buff=0.5)
        
        # Mostrar iteraciones una por una
        for iteration in iterations:
            self.play(FadeIn(iteration), run_time=0.8)
            self.wait(0.3)
        
        # Fórmula final
        cv_formula = MathTex(
            r"RMSE_{CV} = \frac{1}{5}\sum_{k=1}^{5} RMSE_k",
            font_size=16
        )
        cv_formula.to_edge(DOWN, buff=0.5)
        
        self.play(Write(cv_formula))
        self.wait(3)
        
class ExperimentalDesign(Scene):
    def construct(self):
        title = Text("Diseño Experimental", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Protocolo experimental
        protocol_box = Rectangle(width=12, height=6, color=GREEN, stroke_width=2).move_to(ORIGIN)
        
        protocol_title = Text("Protocolo Riguroso", font_size=18, color=YELLOW).next_to(protocol_box.get_top(), DOWN, buff=0.3)
        
        protocol_items = VGroup(
            Text("🔬 50 evaluaciones por método", font_size=14, color=WHITE),
            Text("🔄 3 ejecuciones independientes", font_size=14, color=WHITE),
            Text("📊 Validación cruzada de 5 pliegues", font_size=14, color=WHITE),
            Text("📈 Métrica: RMSE", font_size=14, color=WHITE),
            Text("🎯 Modelos: Random Forest y XGBoost", font_size=14, color=WHITE),
            Text("🧪 Análisis estadístico: Kruskal-Wallis", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).next_to(protocol_title, DOWN, buff=0.5)
        
        self.play(Create(protocol_box))
        self.play(Write(protocol_title))
        
        for item in protocol_items:
            self.play(Write(item), run_time=0.8)
        
        # Fórmula RMSE
        rmse_formula = MathTex(
            r"RMSE = \sqrt{\frac{1}{K}\sum_{k=1}^{K} \frac{1}{|V_k|} \sum_{i \in V_k} (y_i - \hat{y}_i(\lambda))^2}",
            font_size=12
        ).to_edge(DOWN)
        
        self.play(Write(rmse_formula))
        self.wait(3)

class ResultsVisualization(Scene):
    def construct(self):
        title = Text("Resultados - Random Forest", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Crear tabla de resultados
        table_data = [
            ["Método", "RMSE Promedio", "Desv. Estándar", "Mejor RMSE"],
            ["TPE", "29,803.67", "316.86", "29,355.86"],
            ["QMC", "29,835.69", "0.00", "29,835.69"],
            ["Random Search", "30,027.87", "206.73", "29,737.75"],
            ["CMA-ES", "30,207.06", "161.50", "29,991.96"]
        ]
        
        # Crear tabla visual
        table = Table(
            table_data,
            row_labels=[Text("", font_size=1)] * 5,
            col_labels=[Text("", font_size=1)] * 4,
            include_outer_lines=True
        ).scale(0.6).move_to(ORIGIN)
        
        # Colorear la tabla
        table.get_rows()[0].set_color(YELLOW)  # Header
        table.get_rows()[1].set_color(GREEN)   # TPE (ganador)
        
        self.play(Create(table))
        
        # Destacar TPE como ganador
        winner_arrow = Arrow(start=LEFT*2, end=table.get_rows()[1].get_left(), color=GOLD)
        winner_text = Text("GANADOR", font_size=16, color=GOLD).next_to(winner_arrow, LEFT)
        
        self.play(Create(winner_arrow), Write(winner_text))
        
        # Estadísticas clave
        key_stats = VGroup(
            Text("📊 Resultados Clave:", font_size=16, color=YELLOW),
            Text("• TPE: Mejor rendimiento promedio", font_size=12, color=GREEN),
            Text("• QMC: Variabilidad nula (determinístico)", font_size=12, color=WHITE),
            Text("• Random Search: Rendimiento intermedio", font_size=12, color=WHITE),
            Text("• CMA-ES: Peor rendimiento", font_size=12, color=RED)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).to_edge(DOWN, buff=1)
        
        self.play(Write(key_stats))
        self.wait(3)

class ConvergenceAnalysis(Scene):
    def construct(self):
        title = Text("Análisis de Convergencia", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Crear gráfico de convergencia simulado
        axes = Axes(
            x_range=[0, 50, 10],
            y_range=[29000, 35000, 1000],
            x_length=8,
            y_length=5,
            axis_config={"color": WHITE}
        ).shift(DOWN*0.5)
        
        # Etiquetas
        x_label = axes.get_x_axis_label("Evaluaciones")
        y_label = axes.get_y_axis_label("RMSE", direction=UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Curvas de convergencia simuladas
        def tpe_curve(x):
            return 34000 - 4000 * (1 - np.exp(-x/15))
        
        def random_curve(x):
            return 34000 - 3500 * (1 - np.exp(-x/20)) + 200 * np.sin(x/3)
        
        def qmc_curve(x):
            return 34000 - 3800 * (1 - np.exp(-x/25))
        
        def cma_curve(x):
            return 34000 - 3200 * (1 - np.exp(-x/30))
        
        # Crear curvas
        tpe_line = axes.plot(tpe_curve, color=BLUE, x_range=[0, 50])
        random_line = axes.plot(random_curve, color=GREEN, x_range=[0, 50])
        qmc_line = axes.plot(qmc_curve, color=PURPLE, x_range=[0, 50])
        cma_line = axes.plot(cma_curve, color=RED, x_range=[0, 50])
        
        # Leyenda
        legend = VGroup(
            Line(start=ORIGIN, end=RIGHT*0.5, color=BLUE),
            Text("TPE", font_size=12, color=BLUE),
            Line(start=ORIGIN, end=RIGHT*0.5, color=GREEN),
            Text("Random Search", font_size=12, color=GREEN),
            Line(start=ORIGIN, end=RIGHT*0.5, color=PURPLE),
            Text("QMC", font_size=12, color=PURPLE),
            Line(start=ORIGIN, end=RIGHT*0.5, color=RED),
            Text("CMA-ES", font_size=12, color=RED)
        ).arrange_in_grid(rows=4, cols=2, buff=0.2).to_edge(RIGHT)
        
        # Animación de curvas
        self.play(Create(tpe_line), run_time=2)
        self.play(Create(random_line), run_time=2)
        self.play(Create(qmc_line), run_time=2)
        self.play(Create(cma_line), run_time=2)
        self.play(Create(legend))
        
        # Análisis
        analysis = VGroup(
            Text("🔍 Análisis de Convergencia:", font_size=14, color=YELLOW),
            Text("• TPE: Convergencia rápida inicial", font_size=10, color=BLUE),
            Text("• QMC: Convergencia suave y predecible", font_size=10, color=PURPLE),
            Text("• Random: Variabilidad típica", font_size=10, color=GREEN),
            Text("• CMA-ES: Convergencia inicial lenta", font_size=10, color=RED)
        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT).to_edge(DOWN)
        
        self.play(Write(analysis))
        self.wait(3)

class StatisticalAnalysis(Scene):
    def construct(self):
        title = Text("Análisis Estadístico", font_size=24, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Test de Kruskal-Wallis
        kruskal_title = Text("Test de Kruskal-Wallis", font_size=20, color=YELLOW).next_to(title, DOWN, buff=0.5)
        self.play(Write(kruskal_title))
        
        # Hipótesis
        hypotheses = VGroup(
            MathTex(r"H_0: \text{Las medianas son iguales}", font_size=14, color=WHITE),
            MathTex(r"H_1: \text{Al menos una mediana es diferente}", font_size=14, color=WHITE)
        ).arrange(DOWN, buff=0.3).next_to(kruskal_title, DOWN, buff=0.5)
        
        self.play(Write(hypotheses))
        
        # Fórmula
        formula = MathTex(
            r"H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)",
            font_size=16
        ).next_to(hypotheses, DOWN, buff=0.5)
        
        self.play(Write(formula))
        
        # Resultado
        result_box = Rectangle(width=8, height=2, color=GREEN, stroke_width=2).next_to(formula, DOWN, buff=0.5)
        result_text = VGroup(
            Text("Resultado: p < 0.05", font_size=16, color=GREEN, weight=BOLD),
            Text("Rechazamos H₀", font_size=14, color=WHITE),
            Text("Existen diferencias significativas entre métodos", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.2).move_to(result_box)
        
        self.play(Create(result_box))
        self.play(Write(result_text))
        
        # Post-hoc
        posthoc = VGroup(
            Text("Análisis Post-hoc:", font_size=14, color=YELLOW),
            Text("• TPE > Random Search (significativo)", font_size=12, color=GREEN),
            Text("• TPE > CMA-ES (significativo)", font_size=12, color=GREEN),
            Text("• TPE vs QMC (no significativo)", font_size=12, color=BLUE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).to_edge(DOWN)
        
        self.play(Write(posthoc))
        self.wait(3)

class Conclusions(Scene):
    def construct(self):
        title = Text("Conclusiones", font_size=28, color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Conclusiones principales
        conclusions = VGroup(
            Text("🏆 Principales Hallazgos:", font_size=18, color=YELLOW),
            Text("1. TPE logró el mejor rendimiento promedio", font_size=14, color=GREEN),
            Text("2. La optimización bayesiana es superior para este problema", font_size=14, color=GREEN),
            Text("3. QMC ofrece reproducibilidad pero menor rendimiento", font_size=14, color=WHITE),
            Text("4. XGBoost mostró alta sensibilidad a hiperparámetros", font_size=14, color=RED),
            Text("5. El análisis estadístico confirma diferencias significativas", font_size=14, color=BLUE)
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(ORIGIN)
        
        for conclusion in conclusions:
            self.play(Write(conclusion), run_time=1)
        
        # Implicaciones prácticas
        implications = VGroup(
            Text("💡 Implicaciones Prácticas:", font_size=16, color=PURPLE),
            Text("• TPE es recomendado para ajuste automático de hiperparámetros", font_size=12, color=WHITE),
            Text("• Importante considerar el diseño cuidadoso del espacio de búsqueda", font_size=12, color=WHITE),
            Text("• La validación experimental rigurosa es crucial", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).to_edge(DOWN)
        
        self.play(Write(implications))
        self.wait(3)

class FinalSlide(Scene):
    def construct(self):
        # Título final
        title = Text("¡Gracias por su atención!", font_size=36, color=BLUE).move_to(UP*2)
        
        # Información del trabajo
        info = VGroup(
            Text("Optimización sin Derivadas para Funciones de Caja Negra", font_size=20, color=WHITE),
            Text("en el Ajuste de Hiperparámetros", font_size=20, color=WHITE),
            Text("Fiorella Yannet Paredes Coaguila", font_size=16, color=YELLOW),
            Text("Universidad Nacional del Altiplano - FINESI", font_size=14, color=GRAY)
        ).arrange(DOWN, buff=0.3).move_to(ORIGIN)
        
        # Enlaces
        links = VGroup(
            Text("📊 Código disponible en:", font_size=14, color=GREEN),
            Text("GitHub: FYPC7/Paper_Derivative_Free_Optimization", font_size=12, color=BLUE),
            Text("📧 Contacto: 75959821@est.unap.edu.pe", font_size=12, color=WHITE)
        ).arrange(DOWN, buff=0.2).move_to(DOWN*2)
        
        self.play(Write(title), run_time=2)
        self.play(Write(info), run_time=2)
        self.play(Write(links), run_time=2)
        
        # Efecto final
        stars = VGroup()
        for _ in range(20):
            star = Text("⭐", font_size=random.randint(10, 30))
            star.move_to([random.uniform(-6, 6), random.uniform(-3, 3), 0])
            stars.add(star)
        
        self.play(FadeIn(stars), run_time=2)
        self.wait(3)

# Clase principal para renderizar todas las escenas
class CompletePresentation(Scene):
    def construct(self):
        scenes = [
            PaperIntroduction,
            ProblemStatement,
            ProblemCharacteristics,
            WhyDerivativeFree,
            MethodsOverview,
            MethodCards,
            MethodDetailsTPE,
            MethodDetailsCMAES,
            MethodDetailsQMC,
            MethodsComparison,
            TPEExplanation,
            DatasetIntroduction,
            DatasetStatistics,
            FeatureAnalysis,
            ModelSelection,
            EvaluationMetric,
            CrossValidation,
            ExperimentalDesign,
            ResultsVisualization,
            ConvergenceAnalysis,
            StatisticalAnalysis,
            Conclusions,
            FinalSlide
        ]
        
        for scene_class in scenes:
            scene = scene_class()
            scene.construct()
            self.wait(1)
            self.clear()