# MoodIQ

### An Intelligent Adaptive Chatbot for Emotional Support and Logical Advice

MoodIQ is an advanced AI-driven chatbot tool designed to provide personalized conversational experiences. It adapts responses in real-time based on a combination of conversation topics, sentiment analysis, and demographic information. Using state-of-the-art models and neural networks, MoodIQ finds the perfect balance between emotional support and practical, logical advice, offering an empathetic, human-like experience.

---

## Features

- **Adaptive Conversational AI**: Utilizes OpenAI’s LLM to generate intelligent, context-aware responses.
- **Sentiment & Emotion Analysis**: Leverages `roberta-base-go_emotions` to detect emotions and sentiment behind user inputs.
- **Dynamic Topic Categorization**: Uses `bart-large-mnli` to classify conversation topics, particularly those emotionally sensitive to adolescents.
- **Neural Network-Driven Adaptation**: A neural network refines prompts in real-time based on:
  - Conversation topics
  - Sentiment analysis and emotional states
  - Demographic information such as age, gender, and cultural background
- **Balanced Responses**: MoodIQ tailors responses to provide an optimal mix of emotional support and validation, alongside practical, logical advice.
- **Demographic Personalization**: Adjusts chatbot behavior to provide relatable, culturally-sensitive interactions.

---

## How It Works

1. **Conversation Input**: Users engage with MoodIQ via a chat interface.
2. **Emotion & Sentiment Recognition**: The chatbot analyzes the emotional tone of the conversation using `roberta-base-go_emotions` to understand user emotions.
3. **Topic Categorization**: The `bart-large-mnli` model categorizes conversation topics, especially recognizing emotionally sensitive topics (e.g., topics affecting adolescents).
4. **Neural Network Processing**: A neural network processes conversation topics, emotions, and demographic data to predict the best response strategy.
5. **LLM Prompt Generation**: OpenAI’s LLM generates emotionally intelligent and logically sound responses based on the neural network’s predictions.
6. **Personalized Response Delivery**: Users receive real-time adaptive responses, with the right mix of emotional empathy and logical advice.

---

## Tech Stack

- **Frontend**: 
  - React 
  - Next.js (for server-side rendering and routing)
- **Backend**: 
  - Flask (for API and backend logic)
  - OpenAI LLM for response generation
  - `roberta-base-go_emotions` for emotion and sentiment recognition
  - `bart-large-mnli` for topic categorization
- **Machine Learning Frameworks**: 
  - Hugging Face Transformers (for integrating `roberta-base-go_emotions` and `bart-large-mnli` models)
  - PyTorch (for neural network processing)
- **Database**: 
  - MongoDB (for storing user interaction data)
- **APIs**: Flask REST API for serving chat data and managing the LLM interaction pipeline
