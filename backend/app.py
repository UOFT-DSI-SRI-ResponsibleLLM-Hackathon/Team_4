from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

url = "mongodb+srv://ainamerchant10:aSyCBGn0EtvfQUp3@llmcluster.kgbtd.mongodb.net/?retryWrites=true&w=majority&appName=LLMCluster"
client = MongoClient(url, server_api=ServerApi('1'))
db = client['LLMCluster']

# Create collections if they don't exist
if 'users' not in db.list_collection_names():
    db.create_collection('users')
if 'chats' not in db.list_collection_names():
    db.create_collection('chats')

# API endpoint to register a new user
@app.route('/signup', methods=['POST'])
def signup():
    user_data = request.json
    user = {
        "username": user_data['username'],
        "password": user_data['password'],
        "demographics": {
            "age": user_data['age'],
            "gender": user_data['gender'],
            "ethnicity": user_data['location'],
            "education": user_data['education'],
            "employment": user_data['employment']
        }
    }
    db.users.insert_one(user)
    return jsonify({"message": "User registered successfully!"}), 201

# API endpoint to log in a user
@app.route('/login', methods=['POST'])
def login_user():
    login_data = request.json
    user = db.users.find_one({"username": login_data['username']})
    if user:
        return jsonify({"message": "Login successful!", "user_id": str(user['_id'])}), 200
    return jsonify({"message": "Invalid credentials!"}), 401

# API endpoint to log chat data

# TODO: Change this according to implementation
@app.route('/chat', methods=['POST'])
def log_chat():
    chat_data = request.json
    chat_entry = {
        "user_id": chat_data['user_id'],  # Reference to the user's ID
        "prompt": chat_data['prompt'],
        "response": chat_data['response'],
        "feedback": chat_data['feedback'],
        "sentiment": chat_data['sentiment'],
        "topic": chat_data['topic'],
        "timestamp": chat_data.get('timestamp', None)  # Optional timestamp
    }
    db.chats.insert_one(chat_entry)
    return jsonify({"message": "Chat data logged successfully!"}), 201

# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

if __name__ == '__main__':
    app.run(debug=True)
