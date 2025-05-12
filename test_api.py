import json
import requests

# Dirección del servidor Flask (ajusta el puerto si es necesario)
url = 'http://127.0.0.1:8000/predict'

# Datos de prueba con columnas actualizadas
datos = {
    "Genero": 1,
    "Edad": 58,
    "Educacion": 3,
    "Fumador_actual": 1,
    "Cigarros_por_dia": 10,
    "PAM": 95,
    "ACV_prevalente": 0,
    "HTA_prevalente": 1,
    "Diabetes": 0,
    "Colesterol": 230,
    "PAS": 140,
    "PAD": 90,
    "IMC": 27.5,
    "FC": 80,
    "Glucosa": 100
}

# ✅ CAMBIO: Asegurarse que el `Content-Type` es JSON
headers = {'Content-Type': 'application/json'}

# ✅ CAMBIO: Usar json.dumps para asegurar formato correcto
respuesta = requests.post(url, data=json.dumps(datos[0]), headers=headers)

# Mostrar la respuesta
print('Código de estado:', respuesta.status_code)
print('Respuesta del servidor:', respuesta.json())
