import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utiles import save_object

# Cargar datos
df = pd.read_csv(
    'DS_Prediccion_EnfermedadCardiaca_SinDatosPerdidos.csv', sep=';')

# Separar variables independientes y dependiente
X = df.drop('Resultado', axis=1).values
y = df['Resultado'].values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar el scaler
save_object('scaler.pkl', scaler)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=30)

# Crear y entrenar el modelo ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=500, random_state=50)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
print("\nMatriz de confusión:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=[
      "No Infarto Cardiaco", "Infarto Cardaico"], columns=["No Infarto Cardaico", "Infarto Cardaico"]))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred,
      target_names=["Infarto Cardaico", "Infarto Cardaico"]))

# Guardar el modelo
save_object('modelo_entrenado.pkl', model)
