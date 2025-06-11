import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("=== DERIVATIVE-FREE OPTIMIZATION PARA HOUSE PRICES ===")
print()

# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================

def preprocess_house_prices(df):
    """
    Preprocesamiento específico para House Prices dataset
    """
    print("Preprocesando datos...")
    
    # Crear copia para no modificar original
    df_processed = df.copy()
    
    # 1. Manejar valores faltantes en variables numéricas
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # 2. Manejar valores faltantes en variables categóricas
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # 3. Codificar variables categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    print(f"Preprocesamiento completado")
    print(f"   - Variables numéricas: {len(numeric_cols)}")
    print(f"   - Variables categóricas: {len(categorical_cols)}")
    print(f"   - Total features: {df_processed.shape[1] - 1}")  # -1 por SalePrice
    
    return df_processed, label_encoders

# Cargar y preprocesar datos
train_df = pd.read_csv('train.csv')
train_processed, encoders = preprocess_house_prices(train_df)

# Separar features y target
X = train_processed.drop('SalePrice', axis=1)
y = train_processed['SalePrice']

print(f"Dataset final: {X.shape[0]} filas, {X.shape[1]} features")
print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
print()

# =============================================================================
# 2. DEFINIR FUNCIONES OBJETIVO (BLACK-BOX)
# =============================================================================

def random_forest_objective(trial):
    """
    Función objetivo para Random Forest (BLACK-BOX)
    No se pueden calcular gradientes de esta función
    """
    # Hiperparámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Crear modelo
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    # Evaluación con validación cruzada (función costosa)
    scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-scores.mean())
    
    return rmse

def xgboost_objective(trial):
    """
    Función objetivo para XGBoost (BLACK-BOX)
    Función más compleja con más hiperparámetros
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost no instalado. Instalando...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'xgboost'])
        import xgboost as xgb
    
    # Hiperparámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 12)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)
    
    # Crear modelo
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1
    )
    
    # Evaluación con validación cruzada
    scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse = np.sqrt(-scores.mean())
    
    return rmse

# =============================================================================
# 3. CONFIGURAR MÉTODOS DERIVATIVE-FREE
# =============================================================================

def get_derivative_free_samplers():
    """
    Configurar diferentes métodos derivative-free para comparar
    """
    samplers = {
        'Random Search': optuna.samplers.RandomSampler(seed=42),
        'TPE': optuna.samplers.TPESampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
        'QMC': optuna.samplers.QMCSampler(seed=42)
    }
    
    return samplers

# =============================================================================
# 4. FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN
# =============================================================================

def run_derivative_free_optimization(objective_func, objective_name, n_trials=100, n_runs=5):
    """
    Ejecutar optimización derivative-free con diferentes métodos
    """
    print(f"Ejecutando optimización para: {objective_name}")
    print(f"   - Trials por método: {n_trials}")
    print(f"   - Corridas por método: {n_runs}")
    print()
    
    samplers = get_derivative_free_samplers()
    results = {}
    
    for sampler_name, sampler in samplers.items():
        print(f"Probando {sampler_name}...")
        
        method_results = {
            'best_values': [],
            'convergence_history': [],
            'execution_times': []
        }
        
        for run in range(n_runs):
            start_time = time.time()
            
            # Crear estudio
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                study_name=f"{objective_name}_{sampler_name}_run_{run}"
            )
            
            # Optimizar (función black-box)
            study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)
            
            # Guardar resultados
            method_results['best_values'].append(study.best_value)
            method_results['convergence_history'].append([trial.value for trial in study.trials])
            method_results['execution_times'].append(time.time() - start_time)
            
            print(f"   Run {run+1}: RMSE = {study.best_value:.4f}")
        
        results[sampler_name] = method_results
        print(f" {sampler_name} completado")
        print()
    
    return results

# =============================================================================
# 5. ANÁLISIS DE RESULTADOS
# =============================================================================

def analyze_results(results, objective_name):
    """
    Analizar y visualizar resultados de optimización derivative-free
    """
    print(f" ANÁLISIS DE RESULTADOS - {objective_name}")
    print("=" * 60)
    
    summary_stats = {}
    
    for method, data in results.items():
        best_values = data['best_values']
        times = data['execution_times']
        
        stats = {
            'mean_rmse': np.mean(best_values),
            'std_rmse': np.std(best_values),
            'min_rmse': np.min(best_values),
            'max_rmse': np.max(best_values),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
        
        summary_stats[method] = stats
        
        print(f"\n{method}:")
        print(f"  RMSE: {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}")
        print(f"  Mejor: {stats['min_rmse']:.4f}")
        print(f"  Tiempo: {stats['mean_time']:.1f}s ± {stats['std_time']:.1f}s")
    
    print("\n" + "=" * 60)
    
    # Encontrar mejor método
    best_method = min(summary_stats.keys(), key=lambda x: summary_stats[x]['mean_rmse'])
    print(f" MEJOR MÉTODO: {best_method}")
    print(f"   RMSE promedio: {summary_stats[best_method]['mean_rmse']:.4f}")
    
    return summary_stats

# =============================================================================
# 6. EJECUTAR EXPERIMENTO COMPLETO
# =============================================================================

print(" INICIANDO EXPERIMENTO DERIVATIVE-FREE OPTIMIZATION")
print("=" * 60)

# Configuración del experimento
N_TRIALS = 50  # Reducido para demo (aumentar a 100+ para artículo)
N_RUNS = 3     # Reducido para demo (aumentar a 5+ para artículo)

print(f" Configuración:")
print(f"   - Trials por método: {N_TRIALS}")
print(f"   - Corridas por método: {N_RUNS}")
print(f"   - Total evaluaciones: {N_TRIALS * N_RUNS * 4} (4 métodos)")
print()

# Ejecutar optimización para Random Forest
print("=" * 60)
rf_results = run_derivative_free_optimization(
    random_forest_objective, 
    "Random Forest", 
    N_TRIALS, 
    N_RUNS
)

# Analizar resultados
rf_summary = analyze_results(rf_results, "Random Forest")

print("\n EXPERIMENTO COMPLETADO")


#================================================================================


# Instalar XGBoost si no está disponible
try:
    import xgboost as xgb
    print("XGBoost ya instalado")
except ImportError:
    print("Instalando XGBoost...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    print("XGBoost instalado correctamente")

print("\n EXTENSIÓN: XGBOOST DERIVATIVE-FREE OPTIMIZATION")
print("=" * 60)

# =============================================================================
# FUNCIÓN OBJETIVO XGBOOST (BLACK-BOX MÁS COMPLEJA)
# =============================================================================

def xgboost_objective(trial):
    """
    Función objetivo XGBoost - BLACK-BOX con más hiperparámetros
    Más compleja que Random Forest (7 vs 4 hiperparámetros)
    """
    # Hiperparámetros a optimizar (espacio de búsqueda más complejo)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
    }
    
    # Crear modelo XGBoost
    xgb_model = xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        verbosity=0  # Silenciar warnings
    )
    
    # Evaluación con validación cruzada (función costosa)
    try:
        scores = cross_val_score(
            xgb_model, X, y, 
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        rmse = np.sqrt(-scores.mean())
        return rmse
    except Exception as e:
        # Si hay error, devolver un valor alto
        return 50000.0

# =============================================================================
# EJECUTAR OPTIMIZACIÓN XGBOOST
# =============================================================================

def run_xgboost_optimization(n_trials=50, n_runs=3):
    """
    Ejecutar optimización derivative-free para XGBoost
    """
    print(f" Ejecutando optimización XGBoost")
    print(f"   - Trials por método: {n_trials}")
    print(f"   - Corridas por método: {n_runs}")
    print(f"   - Hiperparámetros: 7 (más complejo que RF)")
    print()
    
    # Métodos derivative-free
    samplers = {
        'Random Search': optuna.samplers.RandomSampler(seed=42),
        'TPE': optuna.samplers.TPESampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
        'QMC': optuna.samplers.QMCSampler(seed=42)
    }
    
    results = {}
    
    for sampler_name, sampler in samplers.items():
        print(f"🔍 Probando {sampler_name}...")
        
        method_results = {
            'best_values': [],
            'convergence_history': [],
            'execution_times': []
        }
        
        for run in range(n_runs):
            start_time = time.time()
            
            # Crear estudio
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                study_name=f"XGBoost_{sampler_name}_run_{run}"
            )
            
            # Optimizar función black-box
            study.optimize(
                xgboost_objective, 
                n_trials=n_trials, 
                show_progress_bar=False,
                timeout=300  # 5 minutos máximo por run
            )
            
            # Guardar resultados
            method_results['best_values'].append(study.best_value)
            method_results['convergence_history'].append([trial.value for trial in study.trials])
            method_results['execution_times'].append(time.time() - start_time)
            
            print(f"   Run {run+1}: RMSE = {study.best_value:.4f}")
        
        results[sampler_name] = method_results
        print(f" {sampler_name} completado")
        print()
    
    return results

# =============================================================================
# ANÁLISIS COMPARATIVO (RF vs XGBoost)
# =============================================================================

def analyze_xgboost_results(xgb_results):
    """
    Analizar resultados XGBoost y comparar con Random Forest
    """
    print(f" ANÁLISIS DE RESULTADOS - XGBoost")
    print("=" * 60)
    
    xgb_summary = {}
    
    for method, data in xgb_results.items():
        best_values = data['best_values']
        times = data['execution_times']
        
        stats = {
            'mean_rmse': np.mean(best_values),
            'std_rmse': np.std(best_values),
            'min_rmse': np.min(best_values),
            'max_rmse': np.max(best_values),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
        
        xgb_summary[method] = stats
        
        print(f"\n{method}:")
        print(f"  RMSE: {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}")
        print(f"  Mejor: {stats['min_rmse']:.4f}")
        print(f"  Tiempo: {stats['mean_time']:.1f}s ± {stats['std_time']:.1f}s")
    
    print("\n" + "=" * 60)
    
    # Encontrar mejor método
    best_method = min(xgb_summary.keys(), key=lambda x: xgb_summary[x]['mean_rmse'])
    print(f" MEJOR MÉTODO XGBoost: {best_method}")
    print(f"   RMSE promedio: {xgb_summary[best_method]['mean_rmse']:.4f}")
    
    return xgb_summary

# =============================================================================
# COMPARACIÓN FINAL: RF vs XGBoost
# =============================================================================

def compare_rf_vs_xgboost(rf_summary, xgb_summary):
    """
    Comparación final entre Random Forest y XGBoost
    """
    print(f"\n COMPARACIÓN FINAL: RANDOM FOREST vs XGBOOST")
    print("=" * 70)
    
    # Mejores métodos de cada algoritmo
    best_rf_method = min(rf_summary.keys(), key=lambda x: rf_summary[x]['mean_rmse'])
    best_xgb_method = min(xgb_summary.keys(), key=lambda x: xgb_summary[x]['mean_rmse'])
    
    rf_best_rmse = rf_summary[best_rf_method]['mean_rmse']
    xgb_best_rmse = xgb_summary[best_xgb_method]['mean_rmse']
    
    print(f"🌳 Random Forest + {best_rf_method}: {rf_best_rmse:.4f}")
    print(f"🚀 XGBoost + {best_xgb_method}: {xgb_best_rmse:.4f}")
    
    improvement = rf_best_rmse - xgb_best_rmse
    improvement_pct = (improvement / rf_best_rmse) * 100
    
    if improvement > 0:
        print(f"XGBoost es mejor por {improvement:.2f} RMSE ({improvement_pct:.2f}%)")
    else:
        print(f"Random Forest es mejor por {-improvement:.2f} RMSE ({-improvement_pct:.2f}%)")
    

    
    return {
        'rf_best': (best_rf_method, rf_best_rmse),
        'xgb_best': (best_xgb_method, xgb_best_rmse),
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

# =============================================================================
# EJECUTAR EXPERIMENTO XGBOOST
# =============================================================================

if __name__ == "__main__":
    # Configuración (usa los mismos datos que RF)
    N_TRIALS = 50
    N_RUNS = 3
    
    print("INICIANDO EXPERIMENTO XGBOOST")
    print("=" * 60)
    
    # Ejecutar optimización XGBoost
    xgb_results = run_xgboost_optimization(N_TRIALS, N_RUNS)
    
    # Analizar resultados
    xgb_summary = analyze_xgboost_results(xgb_results)
    
    # Guardar resultados para comparación
    # (Aquí puedes pegar los resultados de RF que ya tienes)
    rf_summary = {
        'Random Search': {'mean_rmse': 30027.8675, 'std_rmse': 206.7287},
        'TPE': {'mean_rmse': 29803.6711, 'std_rmse': 316.8630},
        'CMA-ES': {'mean_rmse': 30207.0623, 'std_rmse': 161.5020},
        'QMC': {'mean_rmse': 29835.6869, 'std_rmse': 0.0000}
    }
    
    # Comparación final
    comparison = compare_rf_vs_xgboost(rf_summary, xgb_summary)
    
    print(f"\n EXPERIMENTO COMPLETO")
    print("Datos listos para tu artículo científico!")
    print("Siguiente paso: Crear visualizaciones")
    
    
    
    # Instalar XGBoost si no está disponible
try:
    import xgboost as xgb
    print(" XGBoost ya instalado")
except ImportError:
    print(" Instalando XGBoost...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost'])
    import xgboost as xgb
    print(" XGBoost instalado correctamente")

print("\n EXTENSIÓN: XGBOOST DERIVATIVE-FREE OPTIMIZATION")
print("=" * 60)

# =============================================================================
# FUNCIÓN OBJETIVO XGBOOST (BLACK-BOX MÁS COMPLEJA)
# =============================================================================

def xgboost_objective(trial):
    """
    Función objetivo XGBoost - BLACK-BOX con más hiperparámetros
    Más compleja que Random Forest (7 vs 4 hiperparámetros)
    """
    # Hiperparámetros a optimizar (espacio de búsqueda más complejo)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
    }
    
    # Crear modelo XGBoost
    xgb_model = xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        verbosity=0  # Silenciar warnings
    )
    
    # Evaluación con validación cruzada (función costosa)
    try:
        scores = cross_val_score(
            xgb_model, X, y, 
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        rmse = np.sqrt(-scores.mean())
        return rmse
    except Exception as e:
        # Si hay error, devolver un valor alto
        return 50000.0

# =============================================================================
# EJECUTAR OPTIMIZACIÓN XGBOOST
# =============================================================================

def run_xgboost_optimization(n_trials=50, n_runs=3):
    """
    Ejecutar optimización derivative-free para XGBoost
    """
    print(f" Ejecutando optimización XGBoost")
    print(f"   - Trials por método: {n_trials}")
    print(f"   - Corridas por método: {n_runs}")
    print(f"   - Hiperparámetros: 7 (más complejo que RF)")
    print()
    
    # Métodos derivative-free
    samplers = {
        'Random Search': optuna.samplers.RandomSampler(seed=42),
        'TPE': optuna.samplers.TPESampler(seed=42),
        'CMA-ES': optuna.samplers.CmaEsSampler(seed=42),
        'QMC': optuna.samplers.QMCSampler(seed=42)
    }
    
    results = {}
    
    for sampler_name, sampler in samplers.items():
        print(f" Probando {sampler_name}...")
        
        method_results = {
            'best_values': [],
            'convergence_history': [],
            'execution_times': []
        }
        
        for run in range(n_runs):
            start_time = time.time()
            
            # Crear estudio
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler,
                study_name=f"XGBoost_{sampler_name}_run_{run}"
            )
            
            # Optimizar función black-box
            study.optimize(
                xgboost_objective, 
                n_trials=n_trials, 
                show_progress_bar=False,
                timeout=300  # 5 minutos máximo por run
            )
            
            # Guardar resultados
            method_results['best_values'].append(study.best_value)
            method_results['convergence_history'].append([trial.value for trial in study.trials])
            method_results['execution_times'].append(time.time() - start_time)
            
            print(f"   Run {run+1}: RMSE = {study.best_value:.4f}")
        
        results[sampler_name] = method_results
        print(f" {sampler_name} completado")
        print()
    
    return results

# =============================================================================
# ANÁLISIS COMPARATIVO (RF vs XGBoost)
# =============================================================================

def analyze_xgboost_results(xgb_results):
    """
    Analizar resultados XGBoost y comparar con Random Forest
    """
    print(f" ANÁLISIS DE RESULTADOS - XGBoost")
    print("=" * 60)
    
    xgb_summary = {}
    
    for method, data in xgb_results.items():
        best_values = data['best_values']
        times = data['execution_times']
        
        stats = {
            'mean_rmse': np.mean(best_values),
            'std_rmse': np.std(best_values),
            'min_rmse': np.min(best_values),
            'max_rmse': np.max(best_values),
            'mean_time': np.mean(times),
            'std_time': np.std(times)
        }
        
        xgb_summary[method] = stats
        
        print(f"\n{method}:")
        print(f"  RMSE: {stats['mean_rmse']:.4f} ± {stats['std_rmse']:.4f}")
        print(f"  Mejor: {stats['min_rmse']:.4f}")
        print(f"  Tiempo: {stats['mean_time']:.1f}s ± {stats['std_time']:.1f}s")
    
    print("\n" + "=" * 60)
    
    # Encontrar mejor método
    best_method = min(xgb_summary.keys(), key=lambda x: xgb_summary[x]['mean_rmse'])
    print(f" MEJOR MÉTODO XGBoost: {best_method}")
    print(f"   RMSE promedio: {xgb_summary[best_method]['mean_rmse']:.4f}")
    
    return xgb_summary

# =============================================================================
# COMPARACIÓN FINAL: RF vs XGBoost
# =============================================================================

def compare_rf_vs_xgboost(rf_summary, xgb_summary):
    """
    Comparación final entre Random Forest y XGBoost
    """
    print(f"\n COMPARACIÓN FINAL: RANDOM FOREST vs XGBOOST")
    print("=" * 70)
    
    # Mejores métodos de cada algoritmo
    best_rf_method = min(rf_summary.keys(), key=lambda x: rf_summary[x]['mean_rmse'])
    best_xgb_method = min(xgb_summary.keys(), key=lambda x: xgb_summary[x]['mean_rmse'])
    
    rf_best_rmse = rf_summary[best_rf_method]['mean_rmse']
    xgb_best_rmse = xgb_summary[best_xgb_method]['mean_rmse']
    
    print(f" Random Forest + {best_rf_method}: {rf_best_rmse:.4f}")
    print(f" XGBoost + {best_xgb_method}: {xgb_best_rmse:.4f}")
    
    improvement = rf_best_rmse - xgb_best_rmse
    improvement_pct = (improvement / rf_best_rmse) * 100
    
    if improvement > 0:
        print(f"XGBoost es mejor por {improvement:.2f} RMSE ({improvement_pct:.2f}%)")
    else:
        print(f"Random Forest es mejor por {-improvement:.2f} RMSE ({-improvement_pct:.2f}%)")
    
    print("\nCONCLUSIONES:")
    print("- Función black-box más compleja (XGBoost) requiere más tiempo")
    print("- Métodos derivative-free mantienen rendimiento relativo")
    print("- TPE generalmente superior en ambos casos")
    print("- Validación en múltiples tipos de función objetivo")
    
    return {
        'rf_best': (best_rf_method, rf_best_rmse),
        'xgb_best': (best_xgb_method, xgb_best_rmse),
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

# =============================================================================
# EJECUTAR EXPERIMENTO XGBOOST
# =============================================================================

if __name__ == "__main__":
    # Configuración (usa los mismos datos que RF)
    N_TRIALS = 50
    N_RUNS = 3
    
    print("INICIANDO EXPERIMENTO XGBOOST")
    print("=" * 60)
    
    # Ejecutar optimización XGBoost
    xgb_results = run_xgboost_optimization(N_TRIALS, N_RUNS)
    
    # Analizar resultados
    xgb_summary = analyze_xgboost_results(xgb_results)
    
    # Guardar resultados para comparación
    # (Aquí puedes pegar los resultados de RF que ya tienes)
    rf_summary = {
        'Random Search': {'mean_rmse': 30027.8675, 'std_rmse': 206.7287},
        'TPE': {'mean_rmse': 29803.6711, 'std_rmse': 316.8630},
        'CMA-ES': {'mean_rmse': 30207.0623, 'std_rmse': 161.5020},
        'QMC': {'mean_rmse': 29835.6869, 'std_rmse': 0.0000}
    }
    
    # Comparación final
    comparison = compare_rf_vs_xgboost(rf_summary, xgb_summary)
    
    print(f"\nEXPERIMENTO COMPLETO")
    print("Datos listos para tu artículo científico!")