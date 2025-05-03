from flask import Flask, request, jsonify
from prompt_builder import build_prompt
from sd_inference import generate_image
import os

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = build_prompt(data)
    image_path = generate_image(prompt)
    return jsonify({"prompt": prompt, "image_path": image_path})

if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True)
