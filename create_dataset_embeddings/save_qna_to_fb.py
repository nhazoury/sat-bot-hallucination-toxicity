import os
import json
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dotenv
import torch
import openai
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def preprocess_text(text):
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return " ".join(filtered_words)




def save_qna_dict_to_firestore(db, vectorizer):
    filepath = 'UpdatedKnowledgeBaseQ_A.json'
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)


    # Preprocess the questions
    questions_original = [qna['Question'] for qna in data['QnAs']]
    questions = [preprocess_text(qna['Question']) for qna in data['QnAs']]
    answers = [qna['Answer'] for qna in data['QnAs']]

    # Initialize the Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode(questions, convert_to_tensor=False)

    # Convert embeddings to list for storage
    question_embeddings_list = question_embeddings.tolist()

    
    # Store the data in Firestore
    db.collection('knowledge_base').document('QnAs').set({
        'questions': questions_original,
        'answers': answers,
        'vectorized_questions': json.dumps(question_embeddings_list)
    })

    print("successfully stored vectorized questions and the answer dict to firestore")
    qna_dict = {"questions":questions, "answers":answers}
    return qna_dict

# Initialize Firebase Admin SDK with your credentials
def initialize_firebase():
    dotenv.load_dotenv()
    cred = credentials.Certificate(json.loads(base64.b64decode(os.environ["FB_ADMIN"]).decode("utf-8")))
    fb_app = firebase_admin.initialize_app(cred, name=__name__)
    db = firestore.client(fb_app)
    return db


    
def load_qna_from_firestore(db):
    doc_ref = db.collection('knowledge_base').document('QnAs')
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
       # Load embeddings as a tensor
        embeddings_tensor = torch.tensor(json.loads(data['vectorized_questions']))
        answers = data['answers']
        questions = data['questions']  # Load the questions as well
        return questions, embeddings_tensor, answers
    else:
        raise Exception("Failed to load data from Firestore")
    

def find_most_relevant_question_old(user_query, vectorized_questions, answers, vectorizer):
    # Assume vectorizer and other preprocessing steps are initialized and loaded similarly
    processed_query = preprocess_text(user_query)
    query_vec = vectorizer.transform([processed_query])
    
    similarities = cosine_similarity(query_vec, vectorized_questions)
    most_similar_idx = np.argmax(similarities)
    
    if similarities[0, most_similar_idx] > 0.5:
        return answers[most_similar_idx]
    else:
        return "Sorry, your question is very new to me."
    
def find_most_relevant_question(user_query, embeddings, questions, answers):
    # Assume vectorizer and other preprocessing steps are initialized and loaded similarly
    #processed_query = preprocess_text(user_query)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([user_query], convert_to_tensor=False)
    
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]  # Get the similarity scores for the query

    most_similar_idx = np.argmax(similarities).item()

    # Filter and collect all indices where similarity is above the threshold
    #relevant_indices = torch.where(similarities > 0.6)[0].tolist()

    # Collect all question-answer pairs that meet the threshold
    #relevant_pairs = [(questions[idx], answers[idx]) for idx in relevant_indices]

    if similarities[most_similar_idx] > 0.6:
        # Collect all question-answer pairs that meet the threshold and are not the most similar one
        relevant_indices = [idx for idx in torch.where(similarities > 0.5)[0].tolist() if idx != most_similar_idx]
        other_QA = [(questions[idx], answers[idx]) for idx in relevant_indices]
        return json.dumps({
            "most_relevant_Q": questions[most_similar_idx],
            "most_relevant_A": answers[most_similar_idx],
            "other_QA": other_QA[:2]
        }, indent=4 )
    else:
        return {}
    

def download_nltk_resources():
    # Define the required resources
    resources = {
        'corpora/wordnet.zip': 'wordnet',
        'tokenizers/punkt.zip': 'punkt',  # Needed for word tokenization
        'corpora/stopwords.zip': 'stopwords'  # Needed for removing stopwords
    }

    # Loop through the resources and download if not found
    for resource_path, resource in resources.items():
        try:
            _ = nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource)
            print(f"Downloaded NLTK resource: {resource}")
        else:
            print(f"NLTK resource '{resource}' is already installed.")


if __name__ == "__main__":
    db = initialize_firebase()
    vectorizer = TfidfVectorizer()


    save_qna_dict_to_firestore(db, vectorizer)

    questions, embeddings, answers = load_qna_from_firestore(db) 
    user_query = "What do you mean by scenario based?"
    answer = find_most_relevant_question(user_query, embeddings, questions, answers)
    print(answer)




    