import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import random
from collections import Counter
import threading

class JuegoSupervivenciaGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🍬 Juego de Supervivencia con Dulces")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables del juego
        self.tipos_dulces = ['limón', 'pera', 'huevo']
        self.jugadores = []
        self.dulces_totales = Counter()
        self.supervivientes = []
        self.comodines = 0
        self.grupos_formados = []
        
        # Colores para los dulces
        self.colores_dulces = {
            'limón': '#FFE135',
            'pera': '#90EE90',
            'huevo': '#F5DEB3'
        }
        
        self.crear_interfaz()
        
    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = tk.Label(main_frame, text="🍬 JUEGO DE SUPERVIVENCIA CON DULCES 🍬", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Entrada para número de jugadores
        ttk.Label(config_frame, text="Número de jugadores:").grid(row=0, column=0, padx=(0, 10))
        self.num_jugadores_var = tk.StringVar(value="6")
        num_jugadores_entry = ttk.Entry(config_frame, textvariable=self.num_jugadores_var, width=10)
        num_jugadores_entry.grid(row=0, column=1, padx=(0, 20))
        
        # Botón para iniciar juego
        self.btn_jugar = ttk.Button(config_frame, text="🎮 Nuevo Juego", command=self.iniciar_juego)
        self.btn_jugar.grid(row=0, column=2, padx=(0, 20))
        
        # Botón para mostrar formación de grupos
        self.btn_formar_grupos = ttk.Button(config_frame, text="🔄 Formar Grupos", 
                                           command=self.mostrar_formacion_grupos, state='disabled')
        self.btn_formar_grupos.grid(row=0, column=3)
        
        # Notebook para las pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Pestaña: Estado del juego
        self.crear_pestaña_estado()
        
        # Pestaña: Jugadores
        self.crear_pestaña_jugadores()
        
        # Pestaña: Formación de grupos
        self.crear_pestaña_formacion()
        
        # Pestaña: Resultados
        self.crear_pestaña_resultados()
        
        # Configurar redimensionamiento
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
    def crear_pestaña_estado(self):
        self.frame_estado = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_estado, text="📊 Estado del Juego")
        
        # Frame para estadísticas generales
        stats_frame = ttk.LabelFrame(self.frame_estado, text="Estadísticas Generales", padding="10")
        stats_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.lbl_num_jugadores = ttk.Label(stats_frame, text="Número de jugadores: -")
        self.lbl_num_jugadores.grid(row=0, column=0, sticky=tk.W)
        
        self.lbl_total_dulces = ttk.Label(stats_frame, text="Total de dulces: -")
        self.lbl_total_dulces.grid(row=1, column=0, sticky=tk.W)
        
        self.lbl_supervivientes_teoricos = ttk.Label(stats_frame, text="Supervivientes teóricos: -")
        self.lbl_supervivientes_teoricos.grid(row=2, column=0, sticky=tk.W)
        
        self.lbl_comodines = ttk.Label(stats_frame, text="Comodines disponibles: -")
        self.lbl_comodines.grid(row=3, column=0, sticky=tk.W)
        
        # Frame para distribución de dulces
        dist_frame = ttk.LabelFrame(self.frame_estado, text="Distribución de Dulces", padding="10")
        dist_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.canvas_dulces = tk.Canvas(dist_frame, height=200, bg='white')
        self.canvas_dulces.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Frame para cálculos matemáticos
        calc_frame = ttk.LabelFrame(self.frame_estado, text="Cálculos Matemáticos", padding="10")
        calc_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.text_calculos = scrolledtext.ScrolledText(calc_frame, height=10, width=40)
        self.text_calculos.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar redimensionamiento
        self.frame_estado.columnconfigure(0, weight=1)
        self.frame_estado.columnconfigure(1, weight=1)
        self.frame_estado.rowconfigure(1, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        dist_frame.columnconfigure(0, weight=1)
        calc_frame.columnconfigure(0, weight=1)
        calc_frame.rowconfigure(0, weight=1)
        
    def crear_pestaña_jugadores(self):
        self.frame_jugadores = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_jugadores, text="👥 Jugadores")
        
        # Frame para la lista de jugadores
        self.tree_jugadores = ttk.Treeview(self.frame_jugadores, columns=('Dulces', 'Tipos', 'Estado'), show='tree headings')
        self.tree_jugadores.heading('#0', text='Jugador')
        self.tree_jugadores.heading('Dulces', text='Dulces')
        self.tree_jugadores.heading('Tipos', text='Tipos Diferentes')
        self.tree_jugadores.heading('Estado', text='Estado')
        
        self.tree_jugadores.column('#0', width=100)
        self.tree_jugadores.column('Dulces', width=150)
        self.tree_jugadores.column('Tipos', width=120)
        self.tree_jugadores.column('Estado', width=120)
        
        scrollbar_jugadores = ttk.Scrollbar(self.frame_jugadores, orient=tk.VERTICAL, command=self.tree_jugadores.yview)
        self.tree_jugadores.configure(yscrollcommand=scrollbar_jugadores.set)
        
        self.tree_jugadores.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_jugadores.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configurar redimensionamiento
        self.frame_jugadores.columnconfigure(0, weight=1)
        self.frame_jugadores.rowconfigure(0, weight=1)
        
    def crear_pestaña_formacion(self):
        self.frame_formacion = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_formacion, text="🔄 Formación de Grupos")
        
        # Área de texto para mostrar la formación paso a paso
        self.text_formacion = scrolledtext.ScrolledText(self.frame_formacion, height=20, width=80)
        self.text_formacion.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar redimensionamiento
        self.frame_formacion.columnconfigure(0, weight=1)
        self.frame_formacion.rowconfigure(0, weight=1)
        
    def crear_pestaña_resultados(self):
        self.frame_resultados = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_resultados, text="🏆 Resultados")
        
        # Frame para supervivientes
        super_frame = ttk.LabelFrame(self.frame_resultados, text="✅ Supervivientes", padding="10")
        super_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.tree_supervivientes = ttk.Treeview(super_frame, columns=('Dulces', 'Grupo'), show='tree headings')
        self.tree_supervivientes.heading('#0', text='Jugador')
        self.tree_supervivientes.heading('Dulces', text='Dulces Originales')
        self.tree_supervivientes.heading('Grupo', text='Grupo Formado')
        
        scroll_super = ttk.Scrollbar(super_frame, orient=tk.VERTICAL, command=self.tree_supervivientes.yview)
        self.tree_supervivientes.configure(yscrollcommand=scroll_super.set)
        
        self.tree_supervivientes.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scroll_super.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Frame para eliminados
        elim_frame = ttk.LabelFrame(self.frame_resultados, text="❌ Eliminados", padding="10")
        elim_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.tree_eliminados = ttk.Treeview(elim_frame, columns=('Dulces', 'Razon'), show='tree headings')
        self.tree_eliminados.heading('#0', text='Jugador')
        self.tree_eliminados.heading('Dulces', text='Dulces')
        self.tree_eliminados.heading('Razon', text='Razón')
        
        scroll_elim = ttk.Scrollbar(elim_frame, orient=tk.VERTICAL, command=self.tree_eliminados.yview)
        self.tree_eliminados.configure(yscrollcommand=scroll_elim.set)
        
        self.tree_eliminados.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scroll_elim.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Frame para resumen final
        resumen_frame = ttk.LabelFrame(self.frame_resultados, text="📊 Resumen Final", padding="10")
        resumen_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.lbl_resumen = ttk.Label(resumen_frame, text="", justify=tk.LEFT)
        self.lbl_resumen.grid(row=0, column=0, sticky=tk.W)
        
        # Configurar redimensionamiento
        self.frame_resultados.columnconfigure(0, weight=1)
        self.frame_resultados.columnconfigure(1, weight=1)
        self.frame_resultados.rowconfigure(0, weight=1)
        super_frame.columnconfigure(0, weight=1)
        super_frame.rowconfigure(0, weight=1)
        elim_frame.columnconfigure(0, weight=1)
        elim_frame.rowconfigure(0, weight=1)
        resumen_frame.columnconfigure(0, weight=1)
        
    def iniciar_juego(self):
        try:
            num_jugadores = int(self.num_jugadores_var.get())
            if num_jugadores < 1:
                messagebox.showerror("Error", "El número de jugadores debe ser mayor a 0")
                return
        except ValueError:
            messagebox.showerror("Error", "Por favor, ingresa un número válido")
            return
        
        # Limpiar datos anteriores
        self.jugadores = []
        self.dulces_totales = Counter()
        self.supervivientes = []
        self.grupos_formados = []
        
        # Crear jugadores
        self.crear_jugadores(num_jugadores)
        
        # Actualizar interfaz
        self.actualizar_estado_juego()
        self.actualizar_lista_jugadores()
        self.limpiar_pestañas()
        
        # Habilitar botón de formar grupos
        self.btn_formar_grupos.config(state='normal')
        
        messagebox.showinfo("Juego Iniciado", f"Se han creado {num_jugadores} jugadores con dulces aleatorios")
        
    def crear_jugadores(self, num_jugadores):
        """Crea jugadores y les asigna 2 dulces aleatorios a cada uno"""
        self.jugadores = []
        self.dulces_totales = Counter()
        
        for i in range(num_jugadores):
            dulces_jugador = [random.choice(self.tipos_dulces) for _ in range(2)]
            self.jugadores.append({
                'id': i + 1,
                'dulces': dulces_jugador,
                'tipos_diferentes': len(set(dulces_jugador)),
                'superviviente': False
            })
            
            # Contar dulces totales
            for dulce in dulces_jugador:
                self.dulces_totales[dulce] += 1
                
    def actualizar_estado_juego(self):
        """Actualiza la pestaña de estado del juego"""
        # Calcular estadísticas
        supervivientes_primera_ronda, dulces_restantes = self.calcular_supervivencia_primera_ronda()
        comodines = self.calcular_comodines(dulces_restantes)
        self.comodines = comodines
        
        # Actualizar etiquetas
        self.lbl_num_jugadores.config(text=f"Número de jugadores: {len(self.jugadores)}")
        self.lbl_total_dulces.config(text=f"Total de dulces: {sum(self.dulces_totales.values())}")
        self.lbl_supervivientes_teoricos.config(text=f"Supervivientes teóricos: {min(supervivientes_primera_ronda + comodines, len(self.jugadores))}")
        self.lbl_comodines.config(text=f"Comodines disponibles: {comodines}")
        
        # Actualizar gráfico de distribución
        self.dibujar_distribucion_dulces()
        
        # Actualizar cálculos matemáticos
        self.mostrar_calculos_matematicos(supervivientes_primera_ronda, dulces_restantes, comodines)
        
    def dibujar_distribucion_dulces(self):
        """Dibuja un gráfico de barras simple para la distribución de dulces"""
        self.canvas_dulces.delete("all")
        
        if not self.dulces_totales:
            return
            
        canvas_width = self.canvas_dulces.winfo_width()
        canvas_height = self.canvas_dulces.winfo_height()
        
        if canvas_width <= 1:  # Canvas no inicializado
            self.root.after(100, self.dibujar_distribucion_dulces)
            return
            
        margin = 40
        bar_width = (canvas_width - 2 * margin) / len(self.tipos_dulces)
        max_count = max(self.dulces_totales.values()) if self.dulces_totales else 1
        
        for i, tipo in enumerate(self.tipos_dulces):
            count = self.dulces_totales[tipo]
            x1 = margin + i * bar_width
            x2 = x1 + bar_width - 10
            y2 = canvas_height - margin
            y1 = y2 - (count / max_count) * (canvas_height - 2 * margin)
            
            # Dibujar barra
            self.canvas_dulces.create_rectangle(x1, y1, x2, y2, 
                                              fill=self.colores_dulces[tipo], 
                                              outline='black')
            
            # Etiqueta del tipo
            self.canvas_dulces.create_text(x1 + bar_width/2 - 5, y2 + 15, 
                                         text=tipo, font=('Arial', 10))
            
            # Valor
            self.canvas_dulces.create_text(x1 + bar_width/2 - 5, y1 - 10, 
                                         text=str(count), font=('Arial', 10, 'bold'))
                                         
    def mostrar_calculos_matematicos(self, supervivientes_primera_ronda, dulces_restantes, comodines):
        """Muestra los cálculos matemáticos en el área de texto"""
        self.text_calculos.delete(1.0, tk.END)
        
        texto = "🧮 CÁLCULOS MATEMÁTICOS:\n"
        texto += "-" * 30 + "\n\n"
        
        texto += f"📊 Dulces disponibles:\n"
        for tipo, cantidad in self.dulces_totales.items():
            texto += f"  • {tipo}: {cantidad}\n"
        
        texto += f"\n🔢 Máximo de combinaciones completas: {supervivientes_primera_ronda}\n"
        texto += f"   (Mínimo entre: {', '.join([str(v) for v in self.dulces_totales.values()])})\n\n"
        
        texto += f"📦 Dulces restantes después de primera ronda:\n"
        for tipo, cantidad in dulces_restantes.items():
            texto += f"  • {tipo}: {cantidad}\n"
        
        total_restantes = sum(dulces_restantes.values())
        texto += f"\n🎪 Total dulces restantes: {total_restantes}\n"
        texto += f"🎯 Comodines posibles: {comodines}\n"
        texto += f"   (Cada comodín necesita 2 dulces)\n\n"
        
        texto += f"🏆 Supervivientes teóricos máximos:\n"
        texto += f"   {supervivientes_primera_ronda} (primera ronda) + {comodines} (comodines) = {min(supervivientes_primera_ronda + comodines, len(self.jugadores))}\n"
        
        self.text_calculos.insert(tk.END, texto)
        
    def calcular_supervivencia_primera_ronda(self):
        """Calcula cuántos jugadores pueden sobrevivir en la primera ronda"""
        dulces_disponibles = self.dulces_totales.copy()
        
        # Calcular máximo de combinaciones completas posibles
        max_combinaciones = min(dulces_disponibles.values()) if dulces_disponibles else 0
        
        # Actualizar dulces disponibles después de formar combinaciones
        for tipo in self.tipos_dulces:
            dulces_disponibles[tipo] -= max_combinaciones
            
        return max_combinaciones, dulces_disponibles
        
    def calcular_comodines(self, dulces_restantes):
        """Calcula cuántos comodines se pueden generar con dulces restantes"""
        total_dulces_restantes = sum(dulces_restantes.values())
        comodines_posibles = total_dulces_restantes // 2
        return comodines_posibles
        
    def actualizar_lista_jugadores(self):
        """Actualiza la lista de jugadores en la interfaz"""
        # Limpiar lista actual
        for item in self.tree_jugadores.get_children():
            self.tree_jugadores.delete(item)
            
        # Agregar jugadores
        for jugador in self.jugadores:
            dulces_str = ", ".join(jugador['dulces'])
            estado = "✅ Superviviente" if jugador['superviviente'] else "⏳ Pendiente"
            
            self.tree_jugadores.insert('', 'end', 
                                     text=f"Jugador {jugador['id']}", 
                                     values=(dulces_str, jugador['tipos_diferentes'], estado))
                                     
    def mostrar_formacion_grupos(self):
        """Ejecuta la formación de grupos en un hilo separado"""
        def formar_grupos():
            try:
                # Deshabilitar botón temporalmente
                self.btn_formar_grupos.config(state='disabled')
                
                # Calcular supervivientes y comodines
                supervivientes_primera_ronda, dulces_restantes = self.calcular_supervivencia_primera_ronda()
                comodines = self.calcular_comodines(dulces_restantes)
                
                # Formar grupos
                self.grupos_formados = self.asignar_supervivientes(supervivientes_primera_ronda, comodines)
                
                # Actualizar interfaz
                self.root.after(0, self.actualizar_interfaz_post_formacion)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al formar grupos: {str(e)}")
                self.btn_formar_grupos.config(state='normal')
                
        # Ejecutar en hilo separado
        thread = threading.Thread(target=formar_grupos)
        thread.daemon = True
        thread.start()
        
    def actualizar_interfaz_post_formacion(self):
        """Actualiza la interfaz después de formar grupos"""
        self.actualizar_lista_jugadores()
        self.actualizar_resultados()
        self.btn_formar_grupos.config(state='normal')
        
        # Cambiar a pestaña de resultados
        self.notebook.select(3)
        
        messagebox.showinfo("Grupos Formados", f"Se formaron {len(self.grupos_formados)} grupos exitosamente")
        
    def asignar_supervivientes(self, supervivientes_primera_ronda, comodines):
        """Asigna supervivientes y actualiza el texto de formación"""
        dulces_disponibles = self.dulces_totales.copy()
        grupos_formados = []
        
        # Limpiar y preparar texto de formación
        self.text_formacion.delete(1.0, tk.END)
        
        def agregar_texto(texto):
            self.text_formacion.insert(tk.END, texto)
            self.text_formacion.see(tk.END)
            self.root.update()
            
        agregar_texto("🔄 FORMACIÓN DE GRUPOS PASO A PASO:\n")
        agregar_texto("=" * 50 + "\n\n")
        
        # FASE 1: Formación directa
        agregar_texto("📋 FASE 1: FORMACIÓN DIRECTA\n")
        agregar_texto("-" * 30 + "\n")
        
        jugadores_2_tipos = [j for j in self.jugadores if j['tipos_diferentes'] == 2]
        jugadores_2_tipos.sort(key=lambda x: x['id'])
        
        grupos_fase1 = 0
        for i in range(min(supervivientes_primera_ronda, len(jugadores_2_tipos))):
            jugador = jugadores_2_tipos[i]
            
            # Determinar dulce faltante
            dulces_jugador = set(jugador['dulces'])
            dulce_faltante = None
            for tipo in self.tipos_dulces:
                if tipo not in dulces_jugador:
                    dulce_faltante = tipo
                    break
                    
            if dulces_disponibles[dulce_faltante] > 0:
                # Formar grupo
                grupo = {
                    'numero': grupos_fase1 + 1,
                    'jugador': jugador,
                    'dulces_jugador': jugador['dulces'].copy(),
                    'dulce_faltante': dulce_faltante,
                    'combinacion_final': jugador['dulces'] + [dulce_faltante]
                }
                grupos_formados.append(grupo)
                
                # Actualizar dulces disponibles
                for dulce in jugador['dulces']:
                    dulces_disponibles[dulce] -= 1
                dulces_disponibles[dulce_faltante] -= 1
                
                jugador['superviviente'] = True
                self.supervivientes.append(jugador)
                grupos_fase1 += 1
                
                agregar_texto(f"  Grupo {grupos_fase1}:\n")
                agregar_texto(f"    👤 Jugador {jugador['id']}: {jugador['dulces']} + {dulce_faltante} (del pool)\n")
                agregar_texto(f"    🍬 Combinación final: {sorted(grupo['combinacion_final'])}\n")
                agregar_texto(f"    📦 Dulces restantes: {dict(dulces_disponibles)}\n\n")
                
        agregar_texto(f"✅ Grupos formados en Fase 1: {grupos_fase1}\n")
        agregar_texto(f"📊 Dulces restantes: {dict(dulces_disponibles)}\n\n")
        
        # FASE 2: Usar comodines
        if comodines > 0:
            agregar_texto("📋 FASE 2: FORMACIÓN CON COMODINES\n")
            agregar_texto("-" * 30 + "\n")
            
            jugadores_restantes = [j for j in self.jugadores if not j['superviviente']]
            jugadores_restantes.sort(key=lambda x: (-x['tipos_diferentes'], x['id']))
            
            grupos_fase2 = 0
            comodines_usados = 0
            
            for jugador in jugadores_restantes:
                if comodines_usados >= comodines:
                    break
                    
                # Determinar dulces necesarios
                dulces_jugador = jugador['dulces']
                tipos_jugador = set(dulces_jugador)
                
                if jugador['tipos_diferentes'] == 2:
                    dulce_faltante = None
                    for tipo in self.tipos_dulces:
                        if tipo not in tipos_jugador:
                            dulce_faltante = tipo
                            break
                    dulces_necesarios = [dulce_faltante]
                elif jugador['tipos_diferentes'] == 1:
                    dulces_necesarios = []
                    for tipo in self.tipos_dulces:
                        if tipo not in tipos_jugador:
                            dulces_necesarios.append(tipo)
                            
                # Verificar si podemos formar el grupo
                dulces_del_pool = 0
                comodines_necesarios = 0
                
                for dulce_necesario in dulces_necesarios:
                    if dulces_disponibles[dulce_necesario] > 0:
                        dulces_del_pool += 1
                        dulces_disponibles[dulce_necesario] -= 1
                    else:
                        comodines_necesarios += 1
                        
                if comodines_necesarios <= (comodines - comodines_usados):
                    # Formar grupo
                    grupo = {
                        'numero': grupos_fase1 + grupos_fase2 + 1,
                        'jugador': jugador,
                        'dulces_jugador': jugador['dulces'].copy(),
                        'dulces_del_pool': dulces_del_pool,
                        'comodines_usados': comodines_necesarios,
                        'dulces_necesarios': dulces_necesarios.copy()
                    }
                    grupos_formados.append(grupo)
                    
                    # Actualizar contadores
                    for dulce in jugador['dulces']:
                        dulces_disponibles[dulce] -= 1
                        
                    comodines_usados += comodines_necesarios
                    grupos_fase2 += 1
                    
                    jugador['superviviente'] = True
                    self.supervivientes.append(jugador)
                    
                    agregar_texto(f"  Grupo {grupos_fase1 + grupos_fase2}:\n")
                    agregar_texto(f"    👤 Jugador {jugador['id']}: {jugador['dulces']}\n")
                    agregar_texto(f"    🍬 Necesita: {dulces_necesarios}\n")
                    agregar_texto(f"    🎯 Del pool: {dulces_del_pool}, Comodines: {comodines_necesarios}\n")
                    agregar_texto(f"    📦 Dulces restantes: {dict(dulces_disponibles)}\n")
                    agregar_texto(f"    🎪 Comodines usados: {comodines_usados}/{comodines}\n\n")
                    
            agregar_texto(f"✅ Grupos formados en Fase 2: {grupos_fase2}\n")
            agregar_texto(f"🎪 Total comodines usados: {comodines_usados}/{comodines}\n\n")
            
        agregar_texto(f"🏆 TOTAL DE GRUPOS FORMADOS: {len(grupos_formados)}\n")
        agregar_texto(f"📊 Dulces finales restantes: {dict(dulces_disponibles)}\n")
        
        return grupos_formados
        
    def actualizar_resultados(self):
        """Actualiza la pestaña de resultados"""
        # Limpiar árboles
        for item in self.tree_supervivientes.get_children():
            self.tree_supervivientes.delete(item)
        for item in self.tree_eliminados.get_children():
            self.tree_eliminados.delete(item)
            
        # Llenar supervivientes
        supervivientes = [j for j in self.jugadores if j['superviviente']]
        for i, jugador in enumerate(supervivientes):
            dulces_str = ", ".join(jugador['dulces'])
            grupo_info = f"Grupo {i+1}"
            
            self.tree_supervivientes.insert('', 'end',
                                          text=f"Jugador {jugador['id']}",
                                          values=(dulces_str, grupo_info))
                                          
        # Llenar eliminados
        eliminados = [j for j in self.jugadores if not j['superviviente']]
        for jugador in eliminados:
            dulces_str = ", ".join(jugador['dulces'])
            razon = "No pudo completar grupo de 3 dulces diferentes"
            
            self.tree_eliminados.insert('', 'end',
                                      text=f"Jugador {jugador['id']}",
                                      values=(dulces_str, razon))
                                      
        # Actualizar resumen
        self.actualizar_resumen_final()
        
    def actualizar_resumen_final(self):
        """Actualiza el resumen final con estadísticas"""
        total_jugadores = len(self.jugadores)
        total_supervivientes = len(self.supervivientes)
        total_dulces = sum(self.dulces_totales.values())
        dulces_usados = total_supervivientes * 3
        
        if total_jugadores > 0 and total_dulces > 0:
            eficiencia = (dulces_usados / total_dulces) * 100
            tasa_supervivencia = (total_supervivientes / total_jugadores) * 100
            
            resumen = f"""📊 ESTADÍSTICAS FINALES:
            
• Total de jugadores: {total_jugadores}
• Jugadores supervivientes: {total_supervivientes}
• Jugadores eliminados: {total_jugadores - total_supervivientes}
• Tasa de supervivencia: {tasa_supervivencia:.1f}%

• Total de dulces disponibles: {total_dulces}
• Dulces utilizados: {dulces_usados}
• Dulces desperdiciados: {total_dulces - dulces_usados}
• Eficiencia en uso de dulces: {eficiencia:.1f}%"""
        else:
            resumen = "No hay datos para mostrar"
            
        self.lbl_resumen.config(text=resumen)
        
    def limpiar_pestañas(self):
        """Limpia las pestañas de formación y resultados"""
        self.text_formacion.delete(1.0, tk.END)
        
        for item in self.tree_supervivientes.get_children():
            self.tree_supervivientes.delete(item)
        for item in self.tree_eliminados.get_children():
            self.tree_eliminados.delete(item)
            
        self.lbl_resumen.config(text="")
        
    def ejecutar(self):
        """Inicia la aplicación"""
        self.root.mainloop()

def main():
    """Función principal"""
    try:
        app = JuegoSupervivenciaGUI()
        app.ejecutar()
    except Exception as e:
        messagebox.showerror("Error", f"Error al iniciar la aplicación: {str(e)}")

if __name__ == "__main__":
    main()