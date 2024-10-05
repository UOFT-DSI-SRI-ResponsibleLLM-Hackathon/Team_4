import pandas as pd
import numpy as np

# Define the schema for conversation categories (one-hot encoding)
conversation_categories = ["exams", "family", "friends", "internship", "sleep", "trauma", "relationships"]

# Define emotion categories (continuous probabilities between 0 and 1)
emotion_categories = [
    "disappointment", "sadness", "neutral", "disapproval", "annoyance", "realization", 
    "remorse", "embarrassment", "approval", "optimism", "nervousness", "love", "anger", 
    "disgust", "joy", "desire", "grief", "admiration", "confusion", "relief", 
    "surprise", "fear", "caring", "excitement", "curiosity", "amusement", 
    "gratitude", "pride"
]

# Define demographic categories
genders = ["male", "female", "non-binary", "prefer not to say"]
ethnicities = ["Black or African American", "Hispanic or Latino", "White", "Native American", 
               "Pacific Islander", "Middle Eastern", "South Asian", "East Asian", "Southeast Asian", "Latino"]
educations = ["high school", "bachelor", "master", "phd"]
employments = ["employed", "student", "unemployed", "self-employed"]

# Function to generate fake data
def generate_fake_data(num_rows):
    data = []
    for _ in range(num_rows):
        row = {}
        
        # One-hot encoded conversation categories
        row.update({cat: np.random.randint(0, 2) for cat in conversation_categories})
        
        # Emotion categories (random float between 0 and 1)
        row.update({emotion: np.random.rand() for emotion in emotion_categories})
        
        # Demographics
        row['age'] = np.random.randint(18, 65)  # Random age between 18 and 65
        
        # One-hot encoding for gender
        gender = np.random.choice(genders)
        row['gender_male'] = 1 if gender == 'male' else 0
        row['gender_female'] = 1 if gender == 'female' else 0
        row['gender_non_binary'] = 1 if gender == 'non-binary' else 0
        row['gender_prefer_not_to_say'] = 1 if gender == 'prefer not to say' else 0
        
        # One-hot encoding for ethnicity
        ethnicity = np.random.choice(ethnicities)
        for ethnic in ethnicities:
            row[f'ethnicity_{ethnic.replace(" ", "_").lower()}'] = 1 if ethnic == ethnicity else 0
        
        # One-hot encoding for education
        education = np.random.choice(educations)
        for edu in educations:
            row[f'education_{edu.replace(" ", "_").lower()}'] = 1 if edu == education else 0
            
        # One-hot encoding for employment
        employment = np.random.choice(employments)
        for emp in employments:
            row[f'employment_{emp.replace(" ", "_").lower()}'] = 1 if emp == employment else 0
        
        # Ratio and feedback
        row['ratio'] = np.random.choice([0, 25, 50, 75, 100])
        row['feedback'] = np.random.randint(1, 6)  # Random feedback between 1 and 5
        
        data.append(row)
    
    return pd.DataFrame(data)

# Generate the fake data
df_fake = generate_fake_data(3000)

print(df_fake.head())  # Show the first few rows of the generated data

# Splitting the dataframe into train, validation, and test sets
from sklearn.model_selection import train_test_split

# First, split into train (80%) and temp (20% which will later be split into validation and test)
df_train, df_temp = train_test_split(df_fake, test_size=0.2, random_state=42)

# Split the temp set further into validation (10%) and test (10%)
df_validate, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

# Save the splits to CSV files
df_train.to_csv('backend/neural_net/data/train.csv', index=False)
df_validate.to_csv('backend/neural_net/data/validate.csv', index=False)
df_test.to_csv('backend/neural_net/data/test.csv', index=False)
