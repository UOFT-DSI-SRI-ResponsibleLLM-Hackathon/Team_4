from transformers import pipeline
import collections

def identify_emotions(msg: str):
    """
    """
    emotions_dict = collections.defaultdict(int)
    
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    sentences = msg.split('.')

    for i in range(len(sentences)):
        output = classifier(sentences)[i]
        for emotion_score in output:
            emotions_dict[emotion_score['label']] += emotion_score['score'] / max(len(sentences), 1)

    return emotions_dict

print(identify_emotions('I ate chocolate cake and it was delicious'))
