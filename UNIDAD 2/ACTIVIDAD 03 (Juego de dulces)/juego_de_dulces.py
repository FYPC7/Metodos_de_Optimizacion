import random
from collections import Counter
import math

class JuegoSupervivencia:
    def __init__(self):
        self.tipos_dulces = ['limon', 'pera', 'huevo']
        self.jugadores = []
        self.dulces_totales = Counter()
        self.supervivientes = []
        self.comodines = 0
        
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
    
    def mostrar_estado_inicial(self):
        """Muestra el estado inicial del juego"""
        print("=" * 60)
        print("🍬 JUEGO DE SUPERVIVENCIA CON DULCES 🍬")
        print("=" * 60)
        print(f"Número de jugadores: {len(self.jugadores)}")
        print(f"Total de dulces: {sum(self.dulces_totales.values())}")
        print(f"Distribución de dulces:")
        for tipo, cantidad in self.dulces_totales.items():
            print(f"  - {tipo}: {cantidad}")
        print()
        
        print("👥 ASIGNACIÓN DE DULCES A JUGADORES:")
        print("-" * 40)
        for jugador in self.jugadores:
            dulces_str = ", ".join(jugador['dulces'])
            print(f"Jugador {jugador['id']}: [{dulces_str}] - {jugador['tipos_diferentes']} tipos diferentes")
        print()
    
    def calcular_supervivencia_primera_ronda(self):
        """Calcula cuántos jugadores pueden sobrevivir en la primera ronda"""
        dulces_disponibles = self.dulces_totales.copy()
        
        print("🧮 CÁLCULO MATEMÁTICO DE SUPERVIVENCIA:")
        print("-" * 40)
        print(f"Dulces disponibles: {dict(dulces_disponibles)}")
        
        # Calcular máximo de combinaciones completas posibles
        max_combinaciones = min(dulces_disponibles.values())
        print(f"Máximo de combinaciones completas posibles: {max_combinaciones}")
        
        # Actualizar dulces disponibles después de formar combinaciones
        for tipo in self.tipos_dulces:
            dulces_disponibles[tipo] -= max_combinaciones
        
        supervivientes_primera_ronda = max_combinaciones
        
        print(f"Dulces restantes después de primera ronda: {dict(dulces_disponibles)}")
        print(f"Jugadores que sobreviven en primera ronda: {supervivientes_primera_ronda}")
        
        return supervivientes_primera_ronda, dulces_disponibles
    
    def calcular_comodines(self, dulces_restantes):
        """Calcula cuántos comodines se pueden generar con dulces restantes"""
        total_dulces_restantes = sum(dulces_restantes.values())
        comodines_posibles = total_dulces_restantes // 2  # Cada comodín necesita 2 dulces
        
        print(f"Total de dulces restantes: {total_dulces_restantes}")
        print(f"Comodines que se pueden generar: {comodines_posibles}")
        
        return comodines_posibles
    
    def asignar_supervivientes(self, supervivientes_primera_ronda, comodines):
        """Asigna qué jugadores sobreviven con formación paso a paso"""
        dulces_disponibles = self.dulces_totales.copy()
        grupos_formados = []
        
        print("🔄 FORMACIÓN DE GRUPOS PASO A PASO:")
        print("=" * 50)
        
        # FASE 1: Formación directa con jugadores que tienen 2 tipos diferentes
        print("📋 FASE 1: FORMACIÓN DIRECTA")
        print("-" * 30)
        
        # Encontrar jugadores con 2 tipos diferentes
        jugadores_2_tipos = [j for j in self.jugadores if j['tipos_diferentes'] == 2]
        jugadores_2_tipos.sort(key=lambda x: x['id'])
        
        grupos_fase1 = 0
        for i in range(min(supervivientes_primera_ronda, len(jugadores_2_tipos))):
            jugador = jugadores_2_tipos[i]
            
            # Determinar qué dulce le falta
            dulces_jugador = set(jugador['dulces'])
            dulce_faltante = None
            for tipo in self.tipos_dulces:
                if tipo not in dulces_jugador:
                    dulce_faltante = tipo
                    break
            
            if dulces_disponibles[dulce_faltante] > 0:
                # Formar grupo completo
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
                
                print(f"  Grupo {grupos_fase1}:")
                print(f"    👤 Jugador {jugador['id']}: {jugador['dulces']} + {dulce_faltante} (del pool)")
                print(f"    🍬 Combinación final: {sorted(grupo['combinacion_final'])}")
                print(f"    📦 Dulces restantes: {dict(dulces_disponibles)}")
                print()
        
        print(f"✅ Grupos formados en Fase 1: {grupos_fase1}")
        print(f"📊 Dulces restantes: {dict(dulces_disponibles)}")
        
        # FASE 2: Usar comodines para formar grupos adicionales
        if comodines > 0:
            print("\n📋 FASE 2: FORMACIÓN CON COMODINES")
            print("-" * 30)
            
            # Encontrar jugadores restantes ordenados por prioridad
            jugadores_restantes = [j for j in self.jugadores if not j['superviviente']]
            
            # Priorizar jugadores con 2 tipos diferentes primero
            jugadores_restantes.sort(key=lambda x: (-x['tipos_diferentes'], x['id']))
            
            grupos_fase2 = 0
            comodines_usados = 0
            
            for jugador in jugadores_restantes:
                if comodines_usados >= comodines:
                    break
                
                # Determinar qué dulces necesita
                dulces_jugador = jugador['dulces']
                tipos_jugador = set(dulces_jugador)
                
                if jugador['tipos_diferentes'] == 2:
                    # Necesita 1 dulce específico
                    dulce_faltante = None
                    for tipo in self.tipos_dulces:
                        if tipo not in tipos_jugador:
                            dulce_faltante = tipo
                            break
                    
                    dulces_necesarios = [dulce_faltante]
                    
                elif jugador['tipos_diferentes'] == 1:
                    # Necesita 2 dulces específicos
                    dulces_necesarios = []
                    for tipo in self.tipos_dulces:
                        if tipo not in tipos_jugador:
                            dulces_necesarios.append(tipo)
                
                # Verificar si podemos formar el grupo con comodines
                dulces_del_pool = 0
                comodines_necesarios = 0
                
                for dulce_necesario in dulces_necesarios:
                    if dulces_disponibles[dulce_necesario] > 0:
                        dulces_del_pool += 1
                        dulces_disponibles[dulce_necesario] -= 1
                    else:
                        comodines_necesarios += 1
                
                if comodines_necesarios <= (comodines - comodines_usados):
                    # Podemos formar el grupo
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
                    
                    print(f"  Grupo {grupos_fase1 + grupos_fase2}:")
                    print(f"    👤 Jugador {jugador['id']}: {jugador['dulces']}")
                    print(f"    🍬 Necesita: {dulces_necesarios}")
                    print(f"    🎯 Del pool: {dulces_del_pool}, Comodines: {comodines_necesarios}")
                    print(f"    📦 Dulces restantes: {dict(dulces_disponibles)}")
                    print(f"    🎪 Comodines usados: {comodines_usados}/{comodines}")
                    print()
            
            print(f"✅ Grupos formados en Fase 2: {grupos_fase2}")
            print(f"🎪 Total comodines usados: {comodines_usados}/{comodines}")
        
        print(f"\n🏆 TOTAL DE GRUPOS FORMADOS: {len(grupos_formados)}")
        print(f"📊 Dulces finales restantes: {dict(dulces_disponibles)}")
        
        return grupos_formados
    
    def mostrar_formacion_grupos(self, supervivientes_primera_ronda, comodines):
        """Muestra resumen de las fases de formación"""
        print("📋 RESUMEN DE FASES:")
        print("-" * 25)
        
        print(f"🎯 Fase 1 - Formación directa:")
        print(f"   • {supervivientes_primera_ronda} grupos posibles con dulces del pool")
        print(f"   • Prioridad: Jugadores con 2 tipos diferentes")
        
        if comodines > 0:
            print(f"🎪 Fase 2 - Formación con comodines:")
            print(f"   • {comodines} comodines disponibles")
            print(f"   • Cada comodín = 1 dulce faltante")
            print(f"   • Se pueden formar hasta {comodines} grupos adicionales")
        
        print(f"📊 Máximo teórico de supervivientes: {min(supervivientes_primera_ronda + comodines, len(self.jugadores))}")
        print()
    
    def mostrar_resultado_final(self):
        """Muestra el resultado final del juego"""
        print("🏆 RESULTADO FINAL:")
        print("=" * 40)
        
        supervivientes = [j for j in self.jugadores if j['superviviente']]
        eliminados = [j for j in self.jugadores if not j['superviviente']]
        
        print("✅ JUGADORES SUPERVIVIENTES:")
        for i, jugador in enumerate(supervivientes, 1):
            dulces_str = ", ".join(jugador['dulces'])
            print(f"  {i}. Jugador {jugador['id']}: [{dulces_str}] → Formó grupo completo")
        
        if eliminados:
            print("\n❌ JUGADORES ELIMINADOS:")
            for i, jugador in enumerate(eliminados, 1):
                dulces_str = ", ".join(jugador['dulces'])
                razon = "No pudo completar grupo de 3 dulces diferentes"
                print(f"  {i}. Jugador {jugador['id']}: [{dulces_str}] → {razon}")
        
        print()
    
    def mostrar_resumen(self):
        """Muestra el resumen final con estadísticas"""
        total_jugadores = len(self.jugadores)
        total_supervivientes = len(self.supervivientes)
        total_dulces = sum(self.dulces_totales.values())
        dulces_usados = total_supervivientes * 3  # Cada superviviente usa 3 dulces
        
        eficiencia = (dulces_usados / total_dulces) * 100
        tasa_supervivencia = (total_supervivientes / total_jugadores) * 100
        
        print("📊 RESUMEN FINAL:")
        print("=" * 40)
        print(f"Total de jugadores: {total_jugadores}")
        print(f"Jugadores supervivientes: {total_supervivientes}")
        print(f"Jugadores eliminados: {total_jugadores - total_supervivientes}")
        print(f"Tasa de supervivencia: {tasa_supervivencia:.1f}%")
        print(f"Total de dulces disponibles: {total_dulces}")
        print(f"Dulces utilizados: {dulces_usados}")
        print(f"Dulces desperdiciados: {total_dulces - dulces_usados}")
        print(f"Eficiencia en uso de dulces: {eficiencia:.1f}%")
        print("=" * 40)
    
    def jugar(self, num_jugadores):
        """Ejecuta el juego completo"""
        # Crear jugadores
        self.crear_jugadores(num_jugadores)
        
        # Mostrar estado inicial
        self.mostrar_estado_inicial()
        
        # Calcular supervivencia primera ronda
        supervivientes_primera_ronda, dulces_restantes = self.calcular_supervivencia_primera_ronda()
        
        # Calcular comodines
        comodines = self.calcular_comodines(dulces_restantes)
        self.comodines = comodines
        
        print()
        
        # Mostrar formación de grupos (resumen)
        self.mostrar_formacion_grupos(supervivientes_primera_ronda, comodines)
        
        # Asignar supervivientes con formación paso a paso
        grupos_formados = self.asignar_supervivientes(supervivientes_primera_ronda, comodines)
        
        # Mostrar resultado final
        self.mostrar_resultado_final()
        
        # Mostrar resumen
        self.mostrar_resumen()

def main():
    """Función principal para ejecutar el juego"""
    print("🎮 Bienvenido al Juego de Supervivencia con Dulces! 🎮")
    print()
    
    while True:
        try:
            num_jugadores = int(input("Ingresa el número de jugadores (o 0 para salir): "))
            
            if num_jugadores == 0:
                print("¡Gracias por jugar! 👋")
                break
            
            if num_jugadores < 1:
                print("❌ El número de jugadores debe ser mayor a 0")
                continue
            
            # Crear y ejecutar el juego
            juego = JuegoSupervivencia()
            juego.jugar(num_jugadores)
            
            print("\n" + "="*60 + "\n")
            
            # Preguntar si quiere jugar otra vez
            otra_vez = input("¿Quieres jugar otra vez? (s/n): ").lower()
            if otra_vez != 's':
                print("¡Gracias por jugar! 👋")
                break
                
        except ValueError:
            print("❌ Por favor, ingresa un número válido")
        except KeyboardInterrupt:
            print("\n¡Juego terminado! 👋")
            break

if __name__ == "__main__":
    main()