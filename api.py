from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# Cargar datos
df = pd.read_csv("diabetes.csv")
X = df[["HighChol", "BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "DiffWalk", "Sex", "Age"]]
y = df["Diabetes_012"]

modelo_mejor = None
nombre_mejor_modelo = ""

@app.route("/train", methods=["POST"])
def train():
    global modelo_mejor, nombre_mejor_modelo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1️⃣ Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    precision_rf = accuracy_score(y_test, y_pred_rf)
    
    # 2️⃣ Regresión Logística
    modelo_lr = LogisticRegression(max_iter=1000)
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    precision_lr = accuracy_score(y_test, y_pred_lr)

    # 3️⃣ K-Means (No supervisado, no tiene precisión directa)
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    modelo_kmeans.fit(X)
    df["Cluster"] = modelo_kmeans.labels_

    # Selección del mejor modelo
    modelos = {
        "Random Forest": (modelo_rf, precision_rf),
        "Regresión Logística": (modelo_lr, precision_lr),
    }

    nombre_mejor_modelo, (modelo_mejor, mejor_precision) = max(modelos.items(), key=lambda x: x[1][1])

    # Guardar el mejor modelo
    joblib.dump(modelo_mejor, "modelo_mejor.pkl")

    return jsonify({
        "modelo_seleccionado": nombre_mejor_modelo,
        "precision_random_forest": precision_rf,
        "precision_logistic_regression": precision_lr,
        "precision_mejor_modelo": mejor_precision
    })

@app.route("/predict", methods=["POST"])
def predict():
    global modelo_mejor, nombre_mejor_modelo

    if modelo_mejor is None:
        try:
            modelo_mejor = joblib.load("modelo_mejor.pkl")
        except:
            return jsonify({"error": "Modelo no encontrado. Por favor, entrena primero."})

    data = request.get_json()
    valores = np.array([[data["HighChol"], data["BMI"], data["Smoker"], data["PhysActivity"],
                         data["Fruits"], data["Veggies"], data["DiffWalk"], data["Sex"], data["Age"]]]).reshape(1, -1)

    resultado = modelo_mejor.predict(valores)[0]

    return jsonify({
        "modelo_utilizado": nombre_mejor_modelo,
        "prediccion": int(resultado)
    })

if __name__ == "__main__":
    app.run(debug=True)
