from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

url = "https://storage.googleapis.com/breathecode/project-files/bank-marketing-campaign-data.csv"
df = pd.read_csv(url, sep=';')

print(df.head())
print(df.info())

df = df.drop_duplicates()

df = df.drop(columns=['duration'])

df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_final.drop('y', axis=1)
y = df_final['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

from sklearn.model_selection import GridSearchCV

model_opt = LogisticRegression(class_weight='balanced', max_iter=1000)

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2']
}

grid_search = GridSearchCV(model_opt, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
y_pred_opt = best_model.predict(X_test_scaled)

print("Mejores parámetros:", grid_search.best_params_)
print("\nReporte de Clasificación Optimizado:\n", classification_report(y_test, y_pred_opt))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='y', palette='viridis')
plt.title('Distribución de suscripciones al depósito (Target)')
plt.xlabel('¿Se suscribió?')
plt.ylabel('Cantidad de clientes')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Distribución de la Edad de los Clientes')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='job', hue='y', palette='magma')
plt.title('Suscripciones por Tipo de Trabajo')
plt.xticks(rotation=45)
plt.xlabel('Profesión')
plt.ylabel('Cantidad')
plt.legend(title='Suscrito', labels=['No', 'Sí'])
plt.show()


subset_cols = ['age', 'campaign', 'emp.var.rate', 'cons.conf.idx', 'y']

sns.pairplot(df[subset_cols], hue='y', palette='husl', diag_kind='kde')
plt.suptitle('Relaciones entre variables clave segmentadas por éxito (y)', y=1.02)
plt.show()



