from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
from utiles import load_object
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from twilio.rest import Client

import os

app = Flask(__name__)
app.secret_key = 'clave_secreta_para_flash'

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

        # Redondear valores para las recomendaciones
        datos_recomendaciones = {
            'Fumador_actual': int(datos['Fumador_actual']),
            'Cigarros_por_dia': int(datos['Cigarros_por_dia']),
            'IMC': round(float(request.form['IMC']), 1),
            'PAS': int(datos['PAS']),
            'PAD': int(datos['PAD']),
            'Colesterol': int(datos['Colesterol']),
            'Glucosa': int(datos['Glucosa']),
            'Edad': int(datos['Edad']),
            'Diabetes': int(datos['Diabetes']),
            'HTA_prevalente': int(datos['HTA_prevalente'])
        }

        datos_originales = request.form.to_dict()

        return render_template('index.html', prediccion=prediccion, prob_cardio=prob_cardio, datos=datos_recomendaciones, form_data=datos_originales)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#Generar PDF

def generar_pdf(resultado, probabilidad, recomendaciones, nombre_archivo):
    pdf_path = f"static/{nombre_archivo}.pdf"
    
    # Definir el documento
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    Story = []

    # Estilos
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    normal_style = styles['Normal']
    
    # Añadir el título
    Story.append(Paragraph("Evaluación Predictiva de Enfermedad Cardiovascular", title_style))
    Story.append(Spacer(1, 0.5 * inch))

    # Predicción y probabilidad
    resultado_text = f"<b>Predicción:</b> {'Alto riesgo para ECV' if resultado == 1 else 'Bajo riesgo para ECV'}"
    probabilidad_text = f"<b>Probabilidad de riesgo:</b> {probabilidad}%"

    Story.append(Paragraph(resultado_text, normal_style))
    Story.append(Spacer(1, 0.25 * inch))
    Story.append(Paragraph(probabilidad_text, normal_style))
    Story.append(Spacer(1, 0.5 * inch))

    # Recomendaciones
    Story.append(Paragraph("<b>Recomendaciones para mejorar tu salud cardíaca:</b>", normal_style))
    Story.append(Spacer(1, 0.25 * inch))
    
    for recomendacion in recomendaciones:
        Story.append(Paragraph(f"• {recomendacion}", normal_style))
        Story.append(Spacer(1, 0.2 * inch))

    # Crear el PDF
    doc.build(Story)
    return pdf_path

@app.route('/enviar_correo', methods=['POST'])
def enviar_correo():
    email_destino = request.form['email']
    resultado = request.form['resultado']
    probabilidad = request.form['probabilidad']

    # Recomendaciones a incluir en el PDF
    recomendaciones = []
    if resultado == 'Positivo para Enfermedad Cardiaca':
        recomendaciones = [
            "Visita un cardiólogo lo más pronto posible.",
            "Disminuir tu peso.",
            "Haz ejercicio aeróbico 5 días a la semana.",
            "Deja el cigarrillo.",
            "Controla tu presión arterial, colesterol y glucosa."
        ]
    else:
        recomendaciones = [
            "Haz ejercicio aeróbico 5 días a la semana.",
            "Evita el cigarrillo.",
            "Controla tu presión arterial, colesterol y glucosa.",
            "No olvides tu chequeo médico anual."
        ]
    
    # Generar PDF
    nombre_archivo_pdf = f"recomendaciones_{email_destino}"
    pdf_path = generar_pdf(resultado, probabilidad, recomendaciones, nombre_archivo_pdf)

    # Correo de Gmail
    remitente = 'gustavorpd04@gmail.com'  # Cambia por tu correo
    password = 'rzev yyvg ajqy isom'  # Contraseña de aplicación de Gmail

    mensaje = MIMEMultipart()
    mensaje['From'] = remitente
    mensaje['To'] = email_destino
    mensaje['Subject'] = 'Resultado de Evaluación Cardíaca con Recomendaciones'

    # Cuerpo del mensaje
    cuerpo = f"""
    <html>
      <body>
        <p>Hola,</p>
        <p>Este es el resultado de tu evaluación predictiva:</p>
        <ul>
          <li><b>Resultado:</b> {resultado}</li>
          <li><b>Probabilidad:</b> {probabilidad}%</li>
        </ul>
        <p>En el archivo adjunto encontrarás las recomendaciones para mejorar tu salud cardíaca.</p>
        <p>Saludos.</p>
      </body>
    </html>
    """
    mensaje.attach(MIMEText(cuerpo, 'html'))

    # Adjuntar el PDF
    with open(pdf_path, "rb") as archivo_pdf:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(archivo_pdf.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={nombre_archivo_pdf}.pdf")
        mensaje.attach(part)

    # Enviar el correo
    try:
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        servidor.starttls()
        servidor.login(remitente, password)
        servidor.sendmail(remitente, email_destino, mensaje.as_string())
        servidor.quit()
        flash('Correo enviado correctamente con el archivo PDF adjunto.', 'success')
    except Exception as e:
        flash(f'Error al enviar correo: {e}', 'danger')

    # Eliminar archivo PDF después de enviarlo
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
