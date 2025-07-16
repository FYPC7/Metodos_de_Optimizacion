import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimaBattlePortfolio:
    def __init__(self, dataset_path="Ronda1.xlsx"):
        """
        Inicializa el sistema de optimización de portafolio para OptimaBattle Arena
        
        Args:
            dataset_path (str): Ruta al archivo Excel con los datos de los activos
        """
        self.dataset_path = dataset_path
        self.data = None
        self.solution = None
        self.start_time = None
        self.end_time = None
        self.lambda_risk = 0.5  # Factor de aversión al riesgo
        self.budget = 1_000_000  # Presupuesto total en soles
        
        # Cargar datos
        self.load_data()
        
    def load_data(self):
        """Carga y prepara los datos del dataset"""
        try:
            # Intentar cargar el archivo Excel
            self.data = pd.read_excel(self.dataset_path)
            
            # Verificar columnas necesarias
            required_columns = ['activo_id', 'retorno_esperado', 'volatilidad', 'beta', 
                              'liquidez_score', 'sector', 'precio_accion', 'min_inversion']
            
            if not all(col in self.data.columns for col in required_columns):
                print("⚠️  Columnas faltantes en el dataset. Generando datos de ejemplo...")
                self.generate_sample_data()
            else:
                print("✅ Dataset cargado exitosamente")
                
        except FileNotFoundError:
            print("⚠️  Archivo no encontrado. Generando datos de ejemplo...")
            self.generate_sample_data()
            
        # Procesar datos
        self.process_data()
        
    def generate_sample_data(self):
        """Genera datos de ejemplo basados en el PDF"""
        np.random.seed(42)
        n_assets = 20
        
        sample_data = {
            'activo_id': [f'A{str(i+1).zfill(3)}' for i in range(n_assets)],
            'retorno_esperado': np.random.uniform(5, 18, n_assets),
            'volatilidad': np.random.uniform(7, 30, n_assets),
            'beta': np.random.uniform(0.5, 1.7, n_assets),
            'liquidez_score': np.random.randint(1, 11, n_assets),
            'sector': np.random.randint(1, 6, n_assets),
            'precio_accion': np.random.uniform(50, 350, n_assets),
            'min_inversion': np.random.uniform(2000, 10500, n_assets)
        }
        
        self.data = pd.DataFrame(sample_data)
        print("✅ Datos de ejemplo generados")
        
    def process_data(self):
        """Procesa y valida los datos"""
        # Convertir porcentajes a decimales
        self.data['retorno_esperado'] = self.data['retorno_esperado'] / 100
        self.data['volatilidad'] = self.data['volatilidad'] / 100
        
        # Agregar información de sectores
        sector_names = {1: 'Tech', 2: 'Salud', 3: 'Energía', 4: 'Financiero', 5: 'Consumo'}
        self.data['sector_nombre'] = self.data['sector'].map(sector_names)
        
        print(f"📊 Datos procesados: {len(self.data)} activos disponibles")
        
    def display_data_summary(self):
        """Muestra resumen de los datos"""
        print("\n" + "="*60)
        print("📈 RESUMEN DEL DATASET")
        print("="*60)
        
        print(f"Total de activos: {len(self.data)}")
        print(f"Retorno esperado promedio: {self.data['retorno_esperado'].mean():.2%}")
        print(f"Volatilidad promedio: {self.data['volatilidad'].mean():.2%}")
        print(f"Beta promedio: {self.data['beta'].mean():.2f}")
        
        print("\n📊 Distribución por sectores:")
        sector_dist = self.data['sector_nombre'].value_counts()
        for sector, count in sector_dist.items():
            print(f"  {sector}: {count} activos")
            
        print("\n💰 Rango de precios:")
        print(f"  Precio mínimo: S/. {self.data['precio_accion'].min():.2f}")
        print(f"  Precio máximo: S/. {self.data['precio_accion'].max():.2f}")
        
    def objective_function(self, weights):
        """
        Función objetivo: Maximizar utilidad del portafolio
        U = Σ(ri * wi) - λ * Σ(σi² * wi²)
        """
        returns = self.data['retorno_esperado'].values
        volatilities = self.data['volatilidad'].values
        
        portfolio_return = np.sum(returns * weights)
        portfolio_risk = np.sum((volatilities ** 2) * (weights ** 2))
        
        # Maximizar utilidad (minimizar su negativo)
        utility = portfolio_return - self.lambda_risk * portfolio_risk
        return -utility
    
    def constraint_budget(self, weights):
        """Restricción de presupuesto"""
        prices = self.data['precio_accion'].values
        min_investments = self.data['min_inversion'].values
        
        # Calcular número de acciones necesarias
        shares = np.floor(weights * self.budget / prices)
        total_investment = np.sum(shares * prices)
        
        return self.budget - total_investment
    
    def constraint_sector_diversification(self, weights):
        """Restricción de diversificación sectorial (máximo 30% por sector)"""
        constraints = []
        
        for sector in range(1, 6):
            sector_mask = self.data['sector'] == sector
            sector_weight = np.sum(weights[sector_mask])
            constraints.append(0.30 - sector_weight)
            
        return np.array(constraints)
    
    def constraint_min_assets(self, weights):
        """Restricción de mínimo 5 activos"""
        # Contar activos con peso > 0.001 (prácticamente > 0)
        active_assets = np.sum(weights > 0.001)
        return active_assets - 5
    
    def constraint_systematic_risk(self, weights):
        """Restricción de riesgo sistemático (beta promedio ≤ 1.2)"""
        betas = self.data['beta'].values
        portfolio_beta = np.sum(betas * weights)
        return 1.2 - portfolio_beta
    
    def constraint_minimum_investment(self, weights):
        """Restricción de inversión mínima por activo"""
        prices = self.data['precio_accion'].values
        min_investments = self.data['min_inversion'].values
        
        constraints = []
        for i in range(len(weights)):
            if weights[i] > 0.001:  # Si invertimos en el activo
                shares = np.floor(weights[i] * self.budget / prices[i])
                actual_investment = shares * prices[i]
                constraints.append(actual_investment - min_investments[i])
            else:
                constraints.append(0)  # No hay restricción si no invertimos
                
        return np.array(constraints)
    
    def optimize_portfolio(self):
        """Optimiza el portafolio usando programación no lineal"""
        print("\n🚀 Iniciando optimización del portafolio...")
        self.start_time = datetime.now()
        
        n_assets = len(self.data)
        
        # Variables iniciales (pesos iguales)
        x0 = np.ones(n_assets) / n_assets
        
        # Restricciones
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Suma de pesos = 1
            {'type': 'ineq', 'fun': self.constraint_budget},
            {'type': 'ineq', 'fun': lambda x: self.constraint_sector_diversification(x)},
            {'type': 'ineq', 'fun': self.constraint_min_assets},
            {'type': 'ineq', 'fun': self.constraint_systematic_risk},
            {'type': 'ineq', 'fun': lambda x: self.constraint_minimum_investment(x)}
        ]
        
        # Límites (pesos entre 0 y 1)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Optimización
        result = minimize(
            self.objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        self.end_time = datetime.now()
        
        if result.success:
            self.solution = result
            print("✅ Optimización exitosa!")
            self.analyze_solution()
        else:
            print("❌ Error en la optimización:")
            print(result.message)
            
    def analyze_solution(self):
        """Analiza la solución obtenida"""
        if self.solution is None:
            print("No hay solución para analizar")
            return
            
        weights = self.solution.x
        
        # Filtrar activos con peso significativo
        significant_assets = weights > 0.001
        selected_data = self.data[significant_assets].copy()
        selected_weights = weights[significant_assets]
        
        # Calcular métricas del portafolio
        portfolio_return = np.sum(selected_data['retorno_esperado'] * selected_weights)
        portfolio_volatility = np.sqrt(np.sum((selected_data['volatilidad'] ** 2) * (selected_weights ** 2)))
        portfolio_beta = np.sum(selected_data['beta'] * selected_weights)
        
        # Calcular inversión por activo
        prices = selected_data['precio_accion'].values
        shares = np.floor(selected_weights * self.budget / prices)
        actual_investments = shares * prices
        
        # Crear DataFrame de resultados
        self.portfolio_results = pd.DataFrame({
            'Activo': selected_data['activo_id'].values,
            'Sector': selected_data['sector_nombre'].values,
            'Peso': selected_weights,
            'Acciones': shares.astype(int),
            'Inversión': actual_investments,
            'Retorno': selected_data['retorno_esperado'].values,
            'Volatilidad': selected_data['volatilidad'].values,
            'Beta': selected_data['beta'].values
        })
        
        # Calcular puntaje del torneo
        execution_time = (self.end_time - self.start_time).total_seconds() / 60
        
        # Factor de tiempo
        if execution_time < 15:
            time_factor = 1.5
        elif execution_time < 20:
            time_factor = 1.2
        else:
            time_factor = 1.0
            
        # Factor de restricciones (asumiendo que cumple todas)
        restrictions_factor = 1.0
        
        # Puntaje final
        score = 1000 * (portfolio_return - 0.5 * portfolio_volatility) * restrictions_factor * time_factor
        
        # Mostrar resultados
        print("\n" + "="*60)
        print("🏆 RESULTADOS DEL PORTAFOLIO OPTIMIZADO")
        print("="*60)
        
        print(f"📊 Métricas del Portafolio:")
        print(f"  Retorno esperado: {portfolio_return:.2%}")
        print(f"  Volatilidad: {portfolio_volatility:.2%}")
        print(f"  Beta del portafolio: {portfolio_beta:.2f}")
        print(f"  Ratio Sharpe aproximado: {(portfolio_return/portfolio_volatility):.2f}")
        
        print(f"\n⏱️  Tiempo de ejecución: {execution_time:.2f} minutos")
        print(f"🎯 Puntaje del torneo: {score:.2f}")
        
        print(f"\n💰 Inversión total: S/. {actual_investments.sum():,.2f}")
        print(f"💵 Presupuesto restante: S/. {self.budget - actual_investments.sum():,.2f}")
        
        print(f"\n📈 Activos seleccionados: {len(self.portfolio_results)}")
        
        # Verificar restricciones
        self.verify_constraints()
        
    def verify_constraints(self):
        """Verifica que se cumplan todas las restricciones"""
        print("\n🔍 VERIFICACIÓN DE RESTRICCIONES:")
        print("-" * 40)
        
        weights = self.solution.x
        significant_assets = weights > 0.001
        
        # 1. Presupuesto
        total_investment = self.portfolio_results['Inversión'].sum()
        print(f"1. Presupuesto: S/. {total_investment:,.2f} / S/. {self.budget:,.2f} {'✅' if total_investment <= self.budget else '❌'}")
        
        # 2. Diversificación sectorial
        print("2. Diversificación sectorial (máx 30% por sector):")
        for sector in range(1, 6):
            sector_mask = self.data['sector'] == sector
            sector_weight = np.sum(weights[sector_mask])
            sector_name = {1: 'Tech', 2: 'Salud', 3: 'Energía', 4: 'Financiero', 5: 'Consumo'}[sector]
            status = '✅' if sector_weight <= 0.30 else '❌'
            print(f"   {sector_name}: {sector_weight:.1%} {status}")
        
        # 3. Mínimo de activos
        active_assets = np.sum(significant_assets)
        print(f"3. Mínimo de activos: {active_assets} {'✅' if active_assets >= 5 else '❌'}")
        
        # 4. Riesgo sistemático
        portfolio_beta = np.sum(self.data['beta'] * weights)
        print(f"4. Beta del portafolio: {portfolio_beta:.2f} {'✅' if portfolio_beta <= 1.2 else '❌'}")
        
        # 5. Inversión mínima
        min_investment_violations = 0
        for _, row in self.portfolio_results.iterrows():
            if row['Inversión'] < self.data[self.data['activo_id'] == row['Activo']]['min_inversion'].iloc[0]:
                min_investment_violations += 1
        
        print(f"5. Inversión mínima: {min_investment_violations} violaciones {'✅' if min_investment_violations == 0 else '❌'}")
        
    def generate_report(self):
        """Genera un reporte detallado del portafolio"""
        if self.solution is None:
            print("No hay solución para generar reporte")
            return
            
        print("\n" + "="*80)
        print("📋 REPORTE DETALLADO DEL PORTAFOLIO")
        print("="*80)
        
        # Mostrar portafolio
        print("\n💼 COMPOSICIÓN DEL PORTAFOLIO:")
        print("-" * 60)
        
        portfolio_display = self.portfolio_results.copy()
        portfolio_display['Peso'] = portfolio_display['Peso'].apply(lambda x: f"{x:.1%}")
        portfolio_display['Inversión'] = portfolio_display['Inversión'].apply(lambda x: f"S/. {x:,.0f}")
        portfolio_display['Retorno'] = portfolio_display['Retorno'].apply(lambda x: f"{x:.1%}")
        portfolio_display['Volatilidad'] = portfolio_display['Volatilidad'].apply(lambda x: f"{x:.1%}")
        
        print(portfolio_display.to_string(index=False))
        
        # Análisis por sectores
        print("\n🏢 ANÁLISIS POR SECTORES:")
        print("-" * 40)
        
        sector_analysis = self.portfolio_results.groupby('Sector').agg({
            'Peso': 'sum',
            'Inversión': 'sum',
            'Retorno': 'mean',
            'Volatilidad': 'mean'
        }).round(4)
        
        print(sector_analysis)
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        print("-" * 30)
        
        avg_return = self.portfolio_results['Retorno'].mean()
        avg_volatility = self.portfolio_results['Volatilidad'].mean()
        
        if avg_return > 0.12:
            print("• Portafolio con alto potencial de retorno")
        if avg_volatility < 0.15:
            print("• Portafolio con riesgo controlado")
        if len(self.portfolio_results) >= 8:
            print("• Buena diversificación en número de activos")
            
    def create_visualizations(self):
        """Crea visualizaciones del portafolio"""
        if self.solution is None:
            print("No hay solución para visualizar")
            return
            
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución de pesos
        ax1.pie(self.portfolio_results['Peso'], labels=self.portfolio_results['Activo'], 
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribución de Pesos por Activo')
        
        # 2. Inversión por sector
        sector_investment = self.portfolio_results.groupby('Sector')['Inversión'].sum()
        ax2.bar(sector_investment.index, sector_investment.values)
        ax2.set_title('Inversión por Sector')
        ax2.set_ylabel('Inversión (S/.)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Retorno vs Volatilidad
        ax3.scatter(self.portfolio_results['Volatilidad'], self.portfolio_results['Retorno'],
                   s=self.portfolio_results['Peso']*1000, alpha=0.6)
        ax3.set_xlabel('Volatilidad')
        ax3.set_ylabel('Retorno Esperado')
        ax3.set_title('Retorno vs Volatilidad (tamaño = peso)')
        
        # 4. Beta vs Peso
        ax4.scatter(self.portfolio_results['Beta'], self.portfolio_results['Peso'])
        ax4.set_xlabel('Beta')
        ax4.set_ylabel('Peso en Portafolio')
        ax4.set_title('Beta vs Peso en Portafolio')
        
        plt.tight_layout()
        plt.show()
        
    def run_optimization(self):
        """Ejecuta el proceso completo de optimización"""
        print("🎯 OPTIMABATTLE ARENA - OPTIMIZACIÓN DE PORTAFOLIO")
        print("="*60)
        
        # Mostrar datos
        self.display_data_summary()
        
        # Optimizar
        self.optimize_portfolio()
        
        # Generar reporte
        if self.solution is not None:
            self.generate_report()
            
        print("\n🏁 Proceso completado. ¡Buena suerte en el torneo!")

# Función principal para ejecutar el sistema
def main():
    """Función principal para ejecutar el sistema OptimaBattle"""
    try:
        # Crear instancia del sistema
        optimizer = OptimaBattlePortfolio("Ronda1.xlsx")
        
        # Ejecutar optimización completa
        optimizer.run_optimization()
        
        # Opción para visualizaciones
        show_plots = input("\n¿Deseas ver las visualizaciones? (s/n): ").lower()
        if show_plots == 's':
            optimizer.create_visualizations()
            
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        print("Verifica que tengas todas las librerías instaladas:")
        print("pip install pandas numpy scipy matplotlib seaborn openpyxl")

if __name__ == "__main__":
    main()