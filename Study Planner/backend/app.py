
import os
from flask import Flask, render_template, request, jsonify
from gemini_client import GeminiClient

app = Flask(__name__, template_folder='../templates')
client = GeminiClient()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = payload.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        response_text = client.generate_response(user_message)
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({'error': 'Error generating response'}), 500

if __name__ == '__main__':
    app.run(debug=True)

