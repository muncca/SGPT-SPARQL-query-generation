from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import torch
from scripts.model import GPT2LMHeadModel
import os

app = Flask(__name__)

# Load the pre-trained model and tokenizer
checkpoint_dir = "/scratch/capolcorsin/SGPT-SPARQL-query-generation/runs/sgpt_ep70_lr6e-4/qald9/"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
#model.to("cuda" if torch.cuda.is_available() else "cpu")
model.to("cpu")
model.eval()


def process_input(input_text):
    # Tokenize and preprocess the input text
    tokenized_input = tokenizer.encode(input_text, return_tensors="pt")
    return tokenized_input.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_sparql(input_text):
    # Process the input text
    input_ids = process_input(input_text)

    # Generate the output using the model
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the output to get the SPARQL query
    sparql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sparql_query


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get("question", "")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    sparql_query = generate_sparql(input_text)
    return jsonify({"sparql_query": sparql_query})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)