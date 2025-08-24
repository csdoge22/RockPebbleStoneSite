import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, DistilBertTokenizerFast


app = Flask(__name__)
CORS(app)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../rps-model-v1.0")

print(f"Loading model from {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=True)

@app.route('/api')
def index():
    return "Hello"

@app.route('/api/classify', methods=["GET","POST"])
def classify():
    """
    API endpoint to predict the RPS category of a task.
    
    Expects a JSON body: {"text": "your task description here"}
    Returns a JSON response: {"label": "rock|pebble|sand"}
    """
    try:
        # 1. Get the JSON data from the request
        data = request.get_json()
        
        # 2. Extract the 'text' field
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
            
        text = data['text']
        
        # 3. Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # 4. Run the model in inference mode
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # 5. Get the predicted class ID and map it to a label
        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        
        # 6. Return the result as JSON
        return jsonify({"label": predicted_label})
        
    except Exception as e:
        # Handle any errors (e.g., model failure, invalid input)
        return jsonify({"error": str(e)}), 500

if __name__=="__main__":
    app.run(debug=False)