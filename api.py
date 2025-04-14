from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode

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

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1️⃣ Modelo: Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)
    y_pred_rf = modelo_rf.predict(X_test)
    precision_rf = accuracy_score(y_test, y_pred_rf)
    
    # 2️⃣ Modelo: Regresión Logística
    modelo_lr = LogisticRegression(max_iter=1000)
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    precision_lr = accuracy_score(y_test, y_pred_lr)

    # 3️⃣ Modelo: K-Means (No supervisado)
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    modelo_kmeans.fit(X)
    # Guardar la asignación de clusters en el DataFrame
    df["Cluster"] = modelo_kmeans.labels_

    # Calcular la “precisión ajustada” para K-Means:
    cluster_map = {}
    for cluster in np.unique(modelo_kmeans.labels_):
        indices = df[df["Cluster"] == cluster].index
        # Utilizamos keepdims=True para asegurar que mode devuelva un arreglo
        most_common_class = mode(df.loc[indices, "Diabetes_012"], keepdims=True).mode[0]
        cluster_map[cluster] = most_common_class
    # Mapeamos los clusters a las clases estimadas
    mapped_labels = df["Cluster"].map(cluster_map)
    precision_kmeans = accuracy_score(df["Diabetes_012"], mapped_labels)
    print("Precisión K-Means (ajustada):", precision_kmeans)

    # Selección del mejor modelo (se consideran los modelos supervisados)
    modelos = {
        "Random Forest": (modelo_rf, precision_rf),
        "Regresión Logística": (modelo_lr, precision_lr),
    }
    nombre_mejor_modelo, (modelo_mejor, mejor_precision) = max(modelos.items(), key=lambda x: x[1][1])

    # Guardar el mejor modelo supervisado
    joblib.dump(modelo_mejor, "modelo_mejor.pkl")

    return jsonify({
        "modelo_seleccionado": nombre_mejor_modelo,
        "precision_random_forest": precision_rf,
        "precision_logistic_regression": precision_lr,
        "precision_kmeans": precision_kmeans,
        "precision_mejor_modelo": mejor_precision
    })

@app.route("/predict", methods=["POST"])
def predict():
    global modelo_mejor, nombre_mejor_modelo

    if modelo_mejor is None:
        try:
            modelo_mejor = joblib.load("modelo_mejor.pkl")
        except Exception as e:
            return jsonify({"error": "Modelo no encontrado. Por favor, entrena primero."})

    data = request.get_json()

    # Lista de nombres de columnas que el modelo espera
    features = ["HighChol", "BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "DiffWalk", "Sex", "Age"]

    # Crear un diccionario, intentando convertir cada valor a float
    input_data = {}
    for feature in features:
        try:
            input_data[feature] = [float(data.get(feature))]
        except (TypeError, ValueError):
            return jsonify({"error": f"El valor de {feature} es inválido o está faltando."})

    # Convertir el diccionario en un DataFrame con las columnas apropiadas
    input_df = pd.DataFrame(input_data)

    # Verificar si existen valores NaN
    if input_df.isnull().values.any():
        return jsonify({"error": "Algunos valores ingresados son inválidos (NaN)."})

    try:
        # Realizar la predicción usando el DataFrame
        resultado = modelo_mejor.predict(input_df)[0]
    except Exception as e:
        return jsonify({"error": "Error al realizar la predicción: " + str(e)})

    return jsonify({
        "modelo_utilizado": nombre_mejor_modelo,
        "prediccion": int(resultado)
    })

@app.route("/graph")
def graph():
    # Calcular la distribución de la clase Diabetes_012
    distribucion = df['Diabetes_012'].value_counts().sort_index()
    
    # Crear un gráfico de barras con la distribución
    plt.figure(figsize=(8, 6))
    sns.barplot(x=distribucion.index, y=distribucion.values, palette="viridis")
    plt.title("Distribución de Diabetes_012 en el dataset")
    plt.xlabel("Clase (0, 1, 2)")
    plt.ylabel("Frecuencia")
    
    # Guardar el gráfico en un buffer en memoria como imagen PNG
    img = io.BytesIO()
    plt.savefig(img, format="png")
    plt.close()  # cerrar la figura para liberar memoria
    img.seek(0)
    
    return Response(img.getvalue(), mimetype="image/png")



if __name__ == "__main__":
    app.run(debug=True)
