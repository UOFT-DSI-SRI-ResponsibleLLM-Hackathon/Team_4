from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS
from openai import OpenAI
import random

base_url = "https://api.aimlapi.com/v1"
api_key = "59324168641e4660879344cf8d23d6f4"

api = OpenAI(api_key=api_key, base_url=base_url)

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
            "ethnicity": user_data['ethnicity'],
            "education": user_data['education'],
            "employment": user_data['employment']
        }
    }
    result = db.users.insert_one(user)
    user_id = str(result.inserted_id)  # Convert ObjectId to string
    print(user_id)
    return jsonify({"message": "User registered successfully!", "user_id": user_id}), 201

# TODO: User_id to be used in login front-end
# API endpoint to log in a user
@app.route('/login', methods=['POST'])
def login_user():
    login_data = request.json
    user = db.users.find_one({"username": login_data['username']})
    if user:
        return jsonify({"message": "Login successful!", "user_id": str(user['_id'])}), 200
    return jsonify({"message": "Invalid credentials!"}), 401

# API endpoint to log chat data

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    system_prompts = ["Respond to the following scenario with purely factual information", "Respond to the following scenario with an even mix of emotion and logic", "Respond to the following scenario with primarily factual information while briefly acknowleding the emotional aspect of the situation", "Respond to the following scenario empathetically but provide brief logical reasoning", "Respond to the following scenario with empathy and emotional support, focusing solely on the feelings"]
    system_prompt = system_prompts[random.randint(0,4)]
    print(system_prompt)

    user_prompt = data.get('prompt')
    print(user_prompt)

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        completion = api.chat.completions.create(
            model="gpt-3.5-turbo-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=256,
        )

        response = completion.choices[0].message.content
        print(response)
        return jsonify({"output": response})

    except Exception as e:
        print("Error in OpenAI API call:", e)
        return jsonify({"error": "Error fetching AI response"}), 500
    
# # TODO: Change this according to implementation
# @app.route('/log_chat', methods=['POST'])
# def log_chat():
#     chat_data = request.json
#     chat_entry = {
#         "user_id": chat_data['user_id'],  # Reference to the user's ID
#         "feedback": chat_data['feedback'],
#         "sentiment": chat_data['sentiment'],
#         "topic": chat_data['topic'],
#     }
#     db.chats.insert_one(chat_entry)
#     return jsonify({"message": "Chat data logged successfully!"}), 201

# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

if __name__ == '__main__':
    app.run()
