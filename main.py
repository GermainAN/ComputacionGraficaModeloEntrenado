from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import numpy as np
import io
import base64
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
modelo = load_model("modelo_kimetsu.h5")

# Diccionario para convertir índices a respiraciones
indice_a_respiracion = {
    0: 'respiracion de la bestia',
    1: 'respiracion de la neblina',
    2: 'respiracion de la roca',
    3: 'respiracion del agua',
    4: 'respiracion del amor',
    5: 'respiracion solar',
    6: 'respiracion de la flor'
}

ESCALA_MODELO = 28  # Ajusta si tu modelo usa otro tamaño

app = Flask(__name__)

# Ruta para servir el frontend
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def preprocesar_imagen(imagen_pil):
    imagen_reducida = imagen_pil.resize((ESCALA_MODELO, ESCALA_MODELO))
    imagen_invertida = ImageOps.invert(imagen_reducida)
    imagen_array = np.array(imagen_invertida) / 255.0
    imagen_array = imagen_array.reshape(1, ESCALA_MODELO, ESCALA_MODELO, 1)
    return imagen_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Decodificar imagen base64
    try:
        image_data = base64.b64decode(data['image'].split(',')[-1])
        imagen_pil = Image.open(io.BytesIO(image_data)).convert('L')
    except Exception as e:
        return jsonify({'error': 'Invalid image data'}), 400

    imagen_array = preprocesar_imagen(imagen_pil)
    salida = modelo.predict(imagen_array)[0]
    prediccion_idx = np.argmax(salida)
    caracter = indice_a_respiracion[prediccion_idx]
    probabilidad = float(salida[prediccion_idx])

    return jsonify({
        'prediccion': caracter,
        'confianza': probabilidad
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
