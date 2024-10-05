import pandas as pd
import torch
import torch.nn as nn
import joblib

# Define the neural network model
class FeedbackNN(nn.Module):
    def __init__(self, input_size):
        super(FeedbackNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path, input_size):
    model = FeedbackNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, scaler, new_data): # predict feedback
    # Preprocess the new data
    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.FloatTensor(new_data_scaled)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(new_data_tensor)
    
    return predictions.numpy()




# Load the trained model and the scaler
# model_path = 'backend/neural_net/model/feedback_nn_model.pth'
# input_size = 59  # Set the input size to 58 based on the trained model's input

# model = load_model(model_path, input_size)

# Load the scaler used during training
scaler_path = 'backend/neural_net/model/scaler.pkl'  # Load the saved scaler
scaler = joblib.load(scaler_path)

# Example new data for predictions (this should be a DataFrame with the same columns as the training data, minus 'feedback')
new_data = pd.DataFrame({
    'exams': [0],
    'family': [1],
    'friends': [0],
    'internship': [1],
    'sleep': [0],
    'trauma': [0],
    'relationships': [0],
    'disappointment': [0],
    'sadness': [0],
    'neutral': [0],
    'disapproval': [0],
    'annoyance': [0],
    'realization': [0],
    'remorse': [0],
    'embarrassment': [0],
    'approval': [0],
    'optimism': [1],
    'nervousness': [0],
    'love': [0],
    'anger': [0],
    'disgust': [0],
    'joy': [0],
    'desire': [0],
    'grief': [0],
    'admiration': [0],
    'confusion': [0],
    'relief': [0],
    'surprise': [0],
    'fear': [0],
    'caring': [0],
    'excitement': [0],
    'curiosity': [0],
    'amusement': [0],
    'gratitude': [0],
    'pride': [0],
    'age': [25],
    'gender_male': [0],
    'gender_female': [1],
    'gender_non_binary': [0],
    'gender_prefer_not_to_say': [0],
    'ethnicity_black_or_african_american': [0],
    'ethnicity_hispanic_or_latino': [0],
    'ethnicity_white': [1],
    'ethnicity_native_american': [0],
    'ethnicity_pacific_islander': [0],
    'ethnicity_middle_eastern': [0],
    'ethnicity_south_asian': [0],
    'ethnicity_east_asian': [0],
    'ethnicity_southeast_asian': [0],
    'ethnicity_latino': [0],
    'education_high_school': [0],
    'education_bachelor': [1],
    'education_master': [0],
    'education_phd': [0],
    'employment_employed': [1],
    'employment_student': [0],
    'employment_unemployed': [0],
    'employment_self-employed': [0],
    'ratio': [0.2]
})

# Ensure the new data has the correct number of features
if new_data.shape[1] != input_size:
    raise ValueError(f"New data must have {input_size} features, but has {new_data.shape[1]}.")

# Get predictions
predictions = predict(model, scaler, new_data)
print("Predicted feedback:", predictions)
