from flask import Flask, request, jsonify
from backup import generate  # from your existing app.py

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 150)
    temperature = data.get("temperature", 0.8)
    
    try:
        result = generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
