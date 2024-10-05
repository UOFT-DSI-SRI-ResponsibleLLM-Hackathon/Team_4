from transformers import pipeline

categories = ["exams", "family", "friends", "internship", "sleep", "trauma", "relationships"]

def categorize(msg: str, categories: list[str]):
    """ Return a dictionary that is 1 for the present category and 0 for all others.
    """
    pipe = pipeline(model="facebook/bart-large-mnli")
    category_scores = pipe(msg, candidate_labels=categories)

    result = {category: 0 for category in categories}

    sorted_categories = category_scores['labels']
    result[sorted_categories[0]] = 1

    return result
    
#print(categorize("I am stressed about applying to jobs and interviewing", categories))

