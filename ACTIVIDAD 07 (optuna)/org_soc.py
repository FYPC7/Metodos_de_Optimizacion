import pandas as pd
from datetime import datetime
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carga de datos
df = pd.read_excel("Organizaciones_Sociales - CERCADO DE LIMA_Mayo.xlsx")  # cambia por el nombre real del archivo

# Convierte fecha de solicitud a datetime
df['FECHA_SOLICITUD'] = pd.to_datetime(df['FECHA_SOLICITUD'], format='%d%m%Y', errors='coerce')

# Elimina filas con datos faltantes
df = df.dropna(subset=['FECHA_SOLICITUD', 'CANTIDAD_MIEMBROS'])

# Transforma la fecha en días desde 01/01/2024
fecha_referencia = datetime(2024, 1, 1)
df['DIAS_SOLICITUD'] = (df['FECHA_SOLICITUD'] - fecha_referencia).dt.days

# Prepara X e y
X = df[['DIAS_SOLICITUD']]
y = df['CANTIDAD_MIEMBROS']



# Divide datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Evaluación
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))





def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Mejores parámetros:", study.best_params)


best_params = study.best_params
final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
y_pred_opt = final_model.predict(X_test)

print("MSE (Optuna):", mean_squared_error(y_test, y_pred_opt))
print("R² (Optuna):", r2_score(y_test, y_pred_opt))



plt.scatter(y_test, y_pred_opt, color='blue')
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.title("Predicción de Cantidad de Miembros (RandomForest + Optuna)")
plt.grid(True)
plt.show()
