from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from app import generate, log_interaction

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "ui.html")

@app.route("/generate", methods=["POST"])
def generate_response():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        temperature = float(data.get("temperature", 0.8))
        max_tokens = int(data.get("max_tokens", 150))

        if not prompt.strip():
            return jsonify({"error": "Empty prompt"}), 400

        response = generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        log_interaction(prompt, response)
        return jsonify({"response": response})
    except ValueError as ve:
        return jsonify({"error": f"Invalid parameter value: {ve}"}), 400
    except KeyError as ke:
        return jsonify({"error": f"Missing parameter: {ke}"}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == "__main__":
    # Recommended change: Disable debug mode and use a production server for deployment.
    # The console output warns against using this for production.
    app.run(debug=True)