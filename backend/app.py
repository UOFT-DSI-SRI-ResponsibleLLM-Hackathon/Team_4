from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS
from prompt_analysis import emotion, category
from neural_net import get_result
import joblib
import data_processing
from openai import OpenAI
from bson.objectid import ObjectId
import os


base_url = "https://api.aimlapi.com/v1"
api_key = "59324168641e4660879344cf8d23d6f4"

api = OpenAI(api_key=api_key, base_url=base_url)

app = Flask(__name__)
CORS(app)

url = "mongodb+srv://ainamerchant10:aSyCBGn0EtvfQUp3@llmcluster.kgbtd.mongodb.net/?retryWrites=true&w=majority&appName=LLMCluster"
client = MongoClient(url, server_api=ServerApi('1'))
db = client['LLMCluster']

global_categories = ["exams", "family", "friends", "internship", "sleep", "trauma", "relationships"]

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
    # system_prompt = system_prompts[random.randint(0,4)]
    # print(system_prompt)

    user_prompt = data.get('prompt')
    user_id = data.get('user_id')
    print(user_prompt)
    print(user_id)

    user = db["users"].find_one({"_id": ObjectId(user_id)})


    print('user')
    print(user)
    print(user_id)

    age, gender, ethnicity, education, employment = user['demographics']['age'], user['demographics']['gender'], user['demographics']['ethnicity'], user['demographics']['education'], user['demographics']['employment']
    
    demographics = {
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "education": education,
            "employment": employment
        }
    
    ratios = [0.01, 0.5, 0.25, 0.75, 1.0]

    # analyze the message to find other paramaters
    emotions = emotion.identify_emotions(user_prompt)
    categories = category.categorize(user_prompt, global_categories)
    
    # TODO: the best ratio, changing variable
    # Load the trained model and the scaler
    print('file path')
    print(os.path.dirname(__file__))
    model_path = os.path.join(os.path.dirname(__file__), 'neural_net', 'model', 'feedback_nn_model.pth')
    input_size = 59  # Set the input size to 58 based on the trained model's input

    model = get_result.load_model(model_path, input_size)

    scaler_path = os.path.join(os.path.dirname(__file__), 'neural_net', 'model', 'scaler.pkl')

    scaler = joblib.load(scaler_path)

    best_feedback, best_ratio_index = 0, 0

    for i in range(len(ratios)):
        data = data_processing.process_data(categories, demographics, emotions, ratios[i])
        predicted_feedback = get_result.predict(model, scaler, data)
        if predicted_feedback > best_feedback:
            best_feedback, best_ratio_index = predicted_feedback, i

    system_prompt = system_prompts[best_ratio_index]


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

# @app.route('/analyze_message', methods=['POST'])
# def analyze_message():
#     # assume request is a JSON dict with user_id and message
#     data = request.json
#     user_id = data['username']
#     message = data['message']


#     # query the database collection called users to find user demographics
#     user = db["users"].find_one({"username": user_id})

#     age, gender, ethnicity, education, employment = user['age'], user['gender'], user['ethnicity'], user['education'], user['employment']

#     demographics = {
#             "age": age,
#             "gender": gender,
#             "ethnicity": ethnicity,
#             "education": education,
#             "employment": employment
#         }


#     ratios = [0.01, 0.25, 0.5, 0.75, 1.0]

#     # analyze the message to find other paramaters
#     emotions = emotion.identify_emotions(message)
#     categories = category.categorize(message)
    
#     # TODO: the best ratio, changing variable
#     # Load the trained model and the scaler
#     model_path = 'backend/neural_net/model/feedback_nn_model.pth'
#     input_size = 59  # Set the input size to 58 based on the trained model's input

#     model = get_result.load_model(model_path, input_size)

#     scaler_path = 'backend/neural_net/model/scaler.pkl'  # Load the saved scaler

#     scaler = joblib.load(scaler_path)

#     best_feedback, best_ratio = 0, 0

#     for ratio in ratios:
#         data = data_processing.process_data(categories, demographics, emotions, ratio)
#         predicted_feedback = get_result(model, scaler, data)
#         if predicted_feedback > best_feedback:
#             best_feedback, best_ratio = predicted_feedback, ratio

#     # call the LLM with a prompt incorporating the best ratio
#     ...

#     # TODO: return the jsonified LLM response
    
    

# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

if __name__ == '__main__':
    app.run()
