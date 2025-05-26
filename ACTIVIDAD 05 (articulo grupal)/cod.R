# Cargar librerías necesarias
library(forecast)
library(tseries)
library(lpSolve)

# Cargar datos
datos <- read.csv("ventass.csv")

# Función para calcular métricas de error
calcular_metricas <- function(real, predicho) {
  mae <- mean(abs(real - predicho))
  rmse <- sqrt(mean((real - predicho)^2))
  mape <- mean(abs((real - predicho)/real)) * 100
  return(list(MAE = mae, RMSE = rmse, MAPE = mape))
}

# Función para predecir cada producto
predecir_producto <- function(producto, datos) {
  col_exogenas <- setdiff(names(datos)[-1], producto)
  
  entrenamiento <- datos[1:56, ]
  prueba <- datos[57:78, ]
  
  ts_objetivo <- ts(entrenamiento[[producto]])
  xreg_entrenamiento <- as.matrix(entrenamiento[, col_exogenas])
  xreg_prueba <- as.matrix(prueba[, col_exogenas])
  
  modelo <- auto.arima(ts_objetivo, xreg = xreg_entrenamiento)
  pred <- forecast(modelo, xreg = xreg_prueba, h = 22)
  
  # Graficar
  plot(pred, main = paste("Predicción de Ventas de", toupper(producto)), col = "blue")
  lines(57:78, prueba[[producto]], col = "red", type = "o", pch = 19)
  legend("topleft", legend = c("Predicción", "Real"), col = c("blue", "red"), lty = 1, pch = 19)
  
  # Calcular y mostrar métricas
  metricas <- calcular_metricas(prueba[[producto]], pred$mean)
  cat("\n--- Métricas para", toupper(producto), "---\n")
  print(metricas)
  
  return(pred$mean)
}

# Lista de productos
productos <- c("pan", "pollo", "arroz", "huevo", "detergente", "shampoo")

# Almacenar predicciones
predicciones_lista <- list()

for (prod in productos) {
  predicciones_lista[[prod]] <- predecir_producto(prod, datos)
}

# ---------- ILP: Optimización de compras ----------

# Supuestos de precios de compra por unidad
costos <- c(pan = 0.23, pollo = 10, arroz = 4, huevo = 0.43, detergente = 1.0, shampoo = 0.8)

# Supuestos de precio de venta por unidad
ventas <- c(pan = 0.35, pollo = 12, arroz = 4.8, huevo = 0.66, detergente = 1.5, shampoo = 1.20)

# Presupuesto semanal máximo (puedes ajustarlo)
presupuesto_semanal <- 500

# Resultados por semana
for (semana in 1:22) {
  demanda <- sapply(predicciones_lista, function(x) ceiling(x[semana]))
  costo_unitario <- costos[productos]
  ingreso_unitario <- ventas[productos]
  beneficio_unitario <- ingreso_unitario - costo_unitario
  
  # Coeficientes para maximizar utilidad
  f.obj <- beneficio_unitario
  
  # Restricción: no exceder el presupuesto
  f.con <- matrix(costo_unitario, nrow = 1)
  f.dir <- "<="
  f.rhs <- presupuesto_semanal
  
  # Restricción adicional: no comprar más de lo que se predice (demanda semanal)
  for (i in 1:length(productos)) {
    restriccion <- rep(0, length(productos))
    restriccion[i] <- 1
    f.con <- rbind(f.con, restriccion)
    f.dir <- c(f.dir, "<=")
    f.rhs <- c(f.rhs, demanda[i])
  }
  
  # Resolver ILP
  solucion <- lp("max", f.obj, f.con, f.dir, f.rhs, all.int = TRUE)
  
  cat("\nSemana", semana + 56, ": Compra óptima\n")
  print(setNames(solucion$solution, productos))
  cat("Total invertido: ", sum(solucion$solution * costo_unitario), "\n")
  cat("Ganancia estimada: ", solucion$objval, "\n")
}
# ------------------graficas --------------------------
# Guardar resultados
ganancias <- c()
inversiones <- c()

for (semana in 1:22) {
  demanda <- sapply(predicciones_lista, function(x) ceiling(x[semana]))
  costo_unitario <- costos[productos]
  ingreso_unitario <- ventas[productos]
  beneficio_unitario <- ingreso_unitario - costo_unitario
  
  f.obj <- beneficio_unitario
  f.con <- matrix(costo_unitario, nrow = 1)
  f.dir <- "<="
  f.rhs <- presupuesto_semanal
  
  for (i in 1:length(productos)) {
    restriccion <- rep(0, length(productos))
    restriccion[i] <- 1
    f.con <- rbind(f.con, restriccion)
    f.dir <- c(f.dir, "<=")
    f.rhs <- c(f.rhs, demanda[i])
  }
  
  solucion <- lp("max", f.obj, f.con, f.dir, f.rhs, all.int = TRUE)
  
  inversiones[semana] <- sum(solucion$solution * costo_unitario)
  ganancias[semana] <- solucion$objval
}

# Graficar
df_opt <- data.frame(
  Semana = 57:78,
  Inversión = inversiones,
  Ganancia = ganancias
)

library(ggplot2)
ggplot(df_opt, aes(x = Semana)) +
  geom_line(aes(y = Inversión, color = "Inversión"), size = 1.2) +
  geom_line(aes(y = Ganancia, color = "Ganancia"), size = 1.2) +
  labs(title = "📊 Evolución de inversión y ganancia (semanas 57–78)",
       y = "Soles", color = "") +
  theme_minimal()
#----------------------------------------------------------------------------------------------------------------------
compras <- matrix(0, nrow = 22, ncol = length(productos))
colnames(compras) <- productos

for (semana in 1:22) {
  demanda <- sapply(predicciones_lista, function(x) ceiling(x[semana]))
  costo_unitario <- costos[productos]
  ingreso_unitario <- ventas[productos]
  beneficio_unitario <- ingreso_unitario - costo_unitario
  
  f.obj <- beneficio_unitario
  f.con <- matrix(costo_unitario, nrow = 1)
  f.dir <- "<="
  f.rhs <- presupuesto_semanal
  
  for (i in 1:length(productos)) {
    restriccion <- rep(0, length(productos))
    restriccion[i] <- 1
    f.con <- rbind(f.con, restriccion)
    f.dir <- c(f.dir, "<=")
    f.rhs <- c(f.rhs, demanda[i])
  }
  
  solucion <- lp("max", f.obj, f.con, f.dir, f.rhs, all.int = TRUE)
  compras[semana, ] <- solucion$solution
}

# Convertir a long format
df_compras <- as.data.frame(compras)
df_compras$Semana <- 57:78
df_compras_long <- reshape2::melt(df_compras, id.vars = "Semana")

# Graficar
ggplot(df_compras_long, aes(x = Semana, y = value, color = variable)) +
  geom_line(size = 1.2) +
  geom_point() +
  labs(title = "📦 Unidades compradas por producto (57–78)",
       y = "Unidades", color = "Producto") +
  theme_minimal()