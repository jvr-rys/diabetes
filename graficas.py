import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Cargar los datos
df = pd.read_csv("diabetes.csv")

# Dividir los datos en características (X) y etiqueta (y)
X = df[["HighChol", "BMI", "Smoker", "PhysActivity", "Fruits", "Veggies", "DiffWalk", "Sex", "Age"]]
y = df["Diabetes_012"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
precision_rf = accuracy_score(y_test, y_pred_rf)
print(f"Precisión Random Forest: {precision_rf}")
joblib.dump(rf_model, "modelo_random_forest.pkl")

# Modelo 2: Regresión Logística
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
precision_lr = accuracy_score(y_test, y_pred_lr)
print(f"Precisión Regresión Logística: {precision_lr}")
joblib.dump(lr_model, "modelo_logistic_regression.pkl")

# Modelo 3: K-Means (No Supervisado)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
df["Cluster"] = kmeans_labels

# Visualización de clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Age'], y=df['BMI'], hue=df['Cluster'], palette='coolwarm', s=100)
plt.title("Clusters de K-Means")
plt.xlabel("Edad")
plt.ylabel("Índice de Masa Corporal")
plt.show()

# Comparación de precisiones
modelos = ["Random Forest", "Regresión Logística"]
precisiones = [precision_rf, precision_lr]

plt.figure(figsize=(8, 6))
plt.bar(modelos, precisiones, color=['blue', 'green'])
plt.title("Comparación de Precisión de Modelos")
plt.ylabel("Precisión")
plt.ylim(0, 1)
plt.show()

#Comparar clusterización con el dataset real