from flask import Flask, render_template, request, jsonify
import pandas as pd
from utiles import load_object
import joblib

app = Flask(__name__)

# Cargar modelo y escalador
modelo = joblib.load('modelo_entrenado.pkl')
escalador = load_object('scaler.pkl')


@app.route('/')
def index():
    return render_template('index.html', prediccion=None)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        datos = {
            'Genero': float(request.form['Genero']),
            'Edad': float(request.form['Edad']),
            'Educacion': float(request.form['Educacion']),
            'Fumador_actual': float(request.form['Fumador_actual']),
            'Cigarros_por_dia': float(request.form['Cigarros_por_dia']),
            'PAM': float(request.form['PAM']),
            'ACV_prevalente': float(request.form['ACV_prevalente']),
            'HTA_prevalente': float(request.form['HTA_prevalente']),
            'Diabetes': float(request.form['Diabetes']),
            'Colesterol': float(request.form['Colesterol']),
            'PAS': float(request.form['PAS']),
            'PAD': float(request.form['PAD']),
            'IMC': float(request.form['IMC']),
            'FC': float(request.form['FC']),
            'Glucosa': float(request.form['Glucosa'])
        }

        df = pd.DataFrame([datos])
        X_scaled = escalador.transform(df)
        prediccion = modelo.predict(X_scaled)[0]

        probabilidades = modelo.predict_proba(X_scaled)[0]
        prob_cardio = round(probabilidades[1] * 100, 2)

        return render_template('index.html', prediccion=prediccion, prob_cardio=prob_cardio)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8000, debug=True)
