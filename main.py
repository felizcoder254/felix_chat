from flask import Flask, request, jsonify, send_from_directory
import requests
import os

app = Flask(__name__, static_folder='static')

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "togethercomputer/llama-3-70b-chat"

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    headers = {
        'Authorization': f"Bearer {TOGETHER_API_KEY}",
        'Content-Type': 'application/json'
    }
    payload = {
        'model': TOGETHER_MODEL,
        'messages': [
            {'role': 'system', 'content': (
                'You are a world-class medical expert and research assistant. ' 
                'Answer all medical questions accurately, cite sources, and when needed, perform web searches or reference up-to-date medical guidelines.'
            )},
            {'role': 'user', 'content': user_input}
        ]
    }
    resp = requests.post('https://api.together.xyz/v1/chat/completions', headers=headers, json=payload)
    reply = resp.json()['choices'][0]['message']['content']
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)