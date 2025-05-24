from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Gunakan nama folder yang benar
app = Flask(__name__, template_folder='Frontend', static_folder='style')

# Coba load model dan tokenizer
try:
    MODEL_PATH = 'model'
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    INDEX2LABEL = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

except Exception as e:
    logger.error(f"Failed to load model or tokenizer: {e}")
    tokenizer = None
    model = None
    INDEX2LABEL = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tech')
def tech():
    return render_template('tech.html')

@app.route('/analisis')
def analisis():
    return render_template('analisis.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text']

        # Jika model/tokenizer gagal dimuat
        if tokenizer is None or model is None:
            return render_template('analisis.html', input_text=text_input, result='Model loading error.')

        try:
            # Tokenisasi dan prediksi
            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs)

            prediction_index = torch.argmax(output.logits, dim=1).item()
            result = INDEX2LABEL[prediction_index]

            # Ubah format hasil sedikit untuk tampilan frontend
            if result == 'Positive':
                display_result = 'Positif'
            elif result == 'Negative':
                display_result = 'Negatif'
            else:
                display_result = 'Netral'

            return render_template('analisis.html', input_text=text_input, result=display_result)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render_template('analisis.html', input_text=text_input, result='Prediction error.')

if __name__ == '__main__':
    app.run(debug=True)