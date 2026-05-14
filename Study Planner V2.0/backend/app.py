import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='../templates')

class GeminiClient:
    def __init__(self):
        try:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            # Added a System Instruction to ensure the AI uses proper Markdown structure
            self.model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                system_instruction="You are an expert AI Study Planner. Use clear Markdown formatting. "
                                   "Use ## for headings, **bold** for emphasis, and bullet points for lists. "
                                   "Keep your responses structured, professional, and easy to read."
            )
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            self.chat = None

    def generate_response(self, user_input: str) -> str:
        if not self.chat:
            return "⚠️ AI service is not configured correctly."
        try:
            response = self.chat.send_message(user_input)
            # RETURN RAW TEXT: Let the frontend (Marked.js) handle the HTML conversion
            return response.text 
        except Exception as e:
            print(f"Error generating response: {e}")
            return "⚠️ Error processing your request."

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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)