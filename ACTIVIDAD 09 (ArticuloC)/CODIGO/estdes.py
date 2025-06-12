import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('train.csv')

print("=== ESTADÍSTICOS DESCRIPTIVOS SIMPLES ===")
print()
# 1. INFORMACIÓN GENERAL DEL DATASET
print(" INFORMACIÓN GENERAL:")
print(f"• Total de casas: {df.shape[0]:,}")
print(f"• Total de características: {df.shape[1]-1}")  # -1 por SalePrice
print(f"• Variables numéricas: {df.select_dtypes(include=[np.number]).shape[1]-1}")
print(f"• Variables categóricas: {df.select_dtypes(include=['object']).shape[1]}")
print()

# 2. VARIABLE OBJETIVO (SalePrice)
print(" PRECIOS DE CASAS (SalePrice):")
print(f"• Precio mínimo: ${df['SalePrice'].min():,}")
print(f"• Precio máximo: ${df['SalePrice'].max():,}")
print(f"• Precio promedio: ${df['SalePrice'].mean():,.0f}")
print(f"• Precio mediano: ${df['SalePrice'].median():,.0f}")
print(f"• Desviación estándar: ${df['SalePrice'].std():,.0f}")
print()

# 3. VARIABLES NUMÉRICAS MÁS IMPORTANTES
print(" VARIABLES NUMÉRICAS CLAVE:")
vars_importantes = ['GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual', 'GarageCars']
for var in vars_importantes:
    if var in df.columns:
        print(f"• {var}:")
        print(f"  - Promedio: {df[var].mean():.1f}")
        print(f"  - Rango: {df[var].min():.0f} - {df[var].max():.0f}")
        print(f"  - Valores faltantes: {df[var].isnull().sum()}")
print()

# 4. VARIABLES CATEGÓRICAS MÁS IMPORTANTES
print(" VARIABLES CATEGÓRICAS CLAVE:")
vars_categoricas = ['Neighborhood', 'HouseStyle', 'ExterQual']
for var in vars_categoricas:
    if var in df.columns:
        print(f"• {var}:")
        print(f"  - Categorías únicas: {df[var].nunique()}")
        print(f"  - Más común: {df[var].mode()[0]} ({df[var].value_counts().iloc[0]} casas)")
        print(f"  - Valores faltantes: {df[var].isnull().sum()}")
print()

# 5. RESUMEN DE VALORES FALTANTES
print(" RESUMEN DE VALORES FALTANTES:")
missing_total = df.isnull().sum().sum()
missing_percent = (missing_total / (df.shape[0] * df.shape[1])) * 100
print(f"• Total de valores faltantes: {missing_total:,}")
print(f"• Porcentaje del dataset: {missing_percent:.1f}%")


