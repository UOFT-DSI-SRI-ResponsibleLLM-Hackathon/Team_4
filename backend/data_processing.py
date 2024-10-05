import pandas as pd

def process_data(category_data, demographics_data, emotion_data, ratio):
    # Initialize a dictionary to hold the final data
    final_data = {
        'exams': [0],
        'family': [0],
        'friends': [0],
        'internship': [0],
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
        'optimism': [0],
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
        'age': [demographics_data['age']],
        'gender_male': [1 if demographics_data['gender'] == 'male' else 0],
        'gender_female': [1 if demographics_data['gender'] == 'female' else 0],
        'gender_non_binary': [1 if demographics_data['gender'] == 'non-binary' else 0],
        'gender_prefer_not_to_say': [1 if demographics_data['gender'] == 'prefer not to say' else 0],
        'ethnicity_black_or_african_american': [1 if demographics_data['ethnicity'] == 'Black or African American' else 0],
        'ethnicity_hispanic_or_latino': [1 if demographics_data['ethnicity'] == 'Hispanic or Latino' else 0],
        'ethnicity_white': [1 if demographics_data['ethnicity'] == 'White' else 0],
        'ethnicity_native_american': [1 if demographics_data['ethnicity'] == 'Native American' else 0],
        'ethnicity_pacific_islander': [1 if demographics_data['ethnicity'] == 'Pacific Islander' else 0],
        'ethnicity_middle_eastern': [1 if demographics_data['ethnicity'] == 'Middle Eastern' else 0],
        'ethnicity_south_asian': [1 if demographics_data['ethnicity'] == 'South Asian' else 0],
        'ethnicity_east_asian': [1 if demographics_data['ethnicity'] == 'East Asian' else 0],
        'ethnicity_southeast_asian': [1 if demographics_data['ethnicity'] == 'Southeast Asian' else 0],
        'ethnicity_latino': [1 if demographics_data['ethnicity'] == 'Latino' else 0],
        'education_high_school': [1 if demographics_data['education'] == 'high school' else 0],
        'education_bachelor': [1 if demographics_data['education'] == 'bachelor' else 0],
        'education_master': [1 if demographics_data['education'] == 'master' else 0],
        'education_phd': [1 if demographics_data['education'] == 'phd' else 0],
        'employment_employed': [1 if demographics_data['employment'] == 'employed' else 0],
        'employment_student': [1 if demographics_data['employment'] == 'student' else 0],
        'employment_unemployed': [1 if demographics_data['employment'] == 'unemployed' else 0],
        'employment_self-employed': [1 if demographics_data['employment'] == 'self-employed' else 0],
        'ratio': [ratio]
    }

    # Update specific columns based on the input data
    for key, value in category_data.items():
        final_data[key] = [value]
    
    for emotion, value in emotion_data.items():
        if emotion in final_data:
            final_data[emotion] = [1 if value > 0 else 0]
    
    # Create the DataFrame
    df = pd.DataFrame(final_data)

    return df

# Example usage
category_data = {'exams': 0, 'family': 1, 'friends': 0, 'internship': 1, 'sleep': 0, 
                 'trauma': 0, 'relationships': 0}

demographics_data = {
    "age": 25,
    "gender": "female", 
    "ethnicity": "White", 
    "education": "bachelor", 
    "employment": "employed"
}

emotion_data = {'joy': 0.560, 'admiration': 0.440, 'excitement': 0.190, 'love': 0.065, 
                'approval': 0.023, 'neutral': 0.013, 'pride': 0.013, 'gratitude': 0.010, 
                'amusement': 0.008, 'surprise': 0.007, 'relief': 0.005, 'realization': 0.005, 
                'annoyance': 0.003, 'desire': 0.003, 'optimism': 0.003, 'caring': 0.002, 
                'curiosity': 0.002, 'disapproval': 0.002, 'sadness': 0.002, 'anger': 0.001, 
                'disappointment': 0.001, 'confusion': 0.001, 'nervousness': 0.001, 
                'disgust': 0.001, 'fear': 0.0009, 'grief': 0.0009, 'embarrassment': 0.0007, 
                'remorse': 0.0004}

# Call the function
new_data = process_data(category_data, demographics_data, emotion_data, 0.2)

print(new_data)
