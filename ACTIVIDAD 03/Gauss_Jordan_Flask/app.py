import re
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

def obtener_numeros(ecuacion):
    izquierda, derecha = ecuacion.split("=")

    if izquierda[0] not in "+-":
        izquierda = "+" + izquierda

    terminos = re.findall(r'[+-](?:\d*\.?\d*)?[a-zA-Z]', izquierda)
    coeficientes = []

    for termino in terminos:
        match = re.match(r'([+-]?\d*\.?\d*)[a-zA-Z]', termino)
        numero = match.group(1)

        if numero in ["+", "-"]:
            numero += "1"
        coeficientes.append(float(numero))
    coeficientes.append(float(derecha))
    return coeficientes

def gauss_jordan_pasos(matriz, n):
    pasos = []
    matriz = matriz.copy()

    for i in range(n):
        if matriz[i][i] == 0:
            raise ValueError("División por cero: hay una columna pivote con valor 0")

        matriz[i] = matriz[i] / matriz[i][i]
        pasos.append((f"Normalizando fila {i+1}", matriz.copy()))

        for j in range(n):
            if i != j:
                matriz[j] = matriz[j] - matriz[i] * matriz[j][i]
                pasos.append((f"Eliminando columna {i+1} de fila {j+1}", matriz.copy()))

    return matriz[:, -1], pasos

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            num_ecuaciones = int(request.form.get("num_ecuaciones"))
            ecuaciones = [request.form.get(f"eq_{i+1}") for i in range(num_ecuaciones)]

            variables = sorted(set(re.findall(r'[a-zA-Z]', ecuaciones[0])))
            A = [obtener_numeros(eq) for eq in ecuaciones]
            A = np.array(A, dtype=float)

            soluciones, pasos = gauss_jordan_pasos(A, len(variables))

            emparejados = list(zip(variables, soluciones))
            return render_template("resultado.html", emparejados=emparejados, pasos=pasos)


        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
