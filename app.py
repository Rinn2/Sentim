from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Inisialisasi Flask
app = Flask(__name__)

# Load model dan tokenizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cpu")
model.to(device)
model.eval()

INDEX2LABEL = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text_input = data.get('text', '')

        if not text_input:
            return jsonify({'error': 'Text is required'}), 400

        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        prediction_index = torch.argmax(output.logits, dim=1).item()
        label = INDEX2LABEL[prediction_index]

        return jsonify({'label': label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run di Railway
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
