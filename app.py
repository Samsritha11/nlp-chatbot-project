from flask import Flask, render_template, request, jsonify
from chatbot_model import ChatBot

app = Flask(__name__)
chatbot = ChatBot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message']
    bot_response = chatbot.get_response(user_msg)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
