from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Inisialisasi Flask
app = Flask(__name__, template_folder='FrondEnd', static_folder='style')

# Load tokenizer dan model dari folder "model"
tokenizer = BertTokenizer.from_pretrained('model')
model = BertForSequenceClassification.from_pretrained('model')

# Gunakan CPU
device = torch.device("cpu")
model.to(device)
model.eval()  # Atur model ke evaluasi

# Mapping index ke label
INDEX2LABEL = {0: 'positive', 1: 'neutral', 2: 'negative'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text']

        # Tokenisasi teks input
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}  #device CPU

        # Prediksi dengan model
        with torch.no_grad():
            output = model(**inputs)

        # menaambil hasil prediksi
        prediction_index = torch.argmax(output.logits, dim=1).item()
        result = INDEX2LABEL[prediction_index].capitalize()

        return render_template('index.html', input_text=text_input, result=result)

if __name__ == '__main__':
    app.run(debug=True)
