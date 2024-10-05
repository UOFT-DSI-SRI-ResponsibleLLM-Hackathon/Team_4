import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the dataset
df = pd.read_csv('backend/neural_net/data/train.csv', index_col=False)

# Split into features and target
X = df.drop(columns=['feedback'])  # Features

print("SHAPEEEEEE")
print("Column names:", X.columns.tolist())
print(X.shape)


y = df['feedback']  # Target (feedback)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val.values)

# Define the Neural Network
class FeedbackNN(nn.Module):
    def __init__(self, input_size):
        super(FeedbackNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer for regression (feedback)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Should be 58
model = FeedbackNN(input_size)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Validate the model
model.eval()
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.4f}')

# Define the model and scaler save paths
model_dir = 'backend/neural_net/model'
os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
model_path = os.path.join(model_dir, 'feedback_nn_model.pth')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

# Save the model
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')

# Save the scaler for future use
joblib.dump(scaler, scaler_path)
print(f'Scaler saved to {scaler_path}')
