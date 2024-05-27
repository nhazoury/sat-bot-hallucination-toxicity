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
import numpy as np

def query_gpt(query):

    intro_prompt="""
Prompt: Given a specific issue reported by a user, you have to see smartly that this problem is most relevant to which problem we have in Database, i have provided the database problem below, smartly identify and return the corresponding problem. Make sure you only identify a problem if you are 100 percent absolutely sure it belongs to a category. If you are even slightly uncertain about the category, return "I don't know" and nothing else.
Make sure you are smart what you determine belongs in a category and what doesn't, and always output "I don't know" if you aren't sure. For example, you should be able to see that just because a user is laughing doesn't mean it belongs to the category related to laughter.

Example 1: 
User: "Wow 5-10 mins is long hahah" 
You: "I don't know" 

Example 2:
User: "Idk it feels weird to force myself to laugh"
You: "User Problem: User feels that laughing at a situation is disrespectful, wrong, or not appropriate"

Example 3:
User: "Are there any exercises I can do against jealousy?"
You: "User Problem: User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity."


Database Problems:
User has a crisis in a personal relationships with their partner, friends, family, neighbor, or at work.
User doesn’t feel comfortable speaking out loud to their inner child
User feels that laughing at a situation is disrespectful, wrong, or not appropriate
User doesn’t understand the purpose behind the house building exercise
User doesn’t understand the purpose of imagining their inner child during protocols
User doesn’t feel a connection to their child
User is in distress or experiencing negative emotions
User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity.
User wants to reflect about their role in society and how society has shaped them.
User wants to nurture opposite traits and practice creativity and affirmations

Task: Match the user's issue smartly to the most relevant problem from the list above. And return only the statement of that problem not a single word more that that.
"""

    response = openai.ChatCompletion.create(
    #response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": intro_prompt}, {"role": "user", "content": f"User Problem: {query}"}],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content

# Function to preprocess text by removing punctuation, lowercasing, and lemmatizing
def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, and lemmatizing."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def parse_file_to_dict(file_path):
    data_dict = {}
    current_problem = None
    current_section = None

    # Regular expression patterns for identifying sections
    problem_pattern = re.compile(r'- Problem: (.+)')
    knowledge_segment_pattern = re.compile(r'- Knowledge Segment:')
    advice_pattern = re.compile(r'- Advice:')
    exercise_pattern = re.compile(r'- Exercise:')

    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Match problem statements
            problem_match = problem_pattern.match(line)
            #print("\n\nMatched Problem is: ",problem_match)
            if problem_match:
                current_problem = problem_match.group(1)
                #print("\n\nCurrent Problem is: ",current_problem)
                data_dict[current_problem] = {"Knowledge Segment": "", "Advice": "", "Exercise": []}
                current_section = None
                continue

            # Match knowledge segment section
            if knowledge_segment_pattern.match(line):
                current_section = "Knowledge Segment"
                continue

            # Match advice section
            if advice_pattern.match(line):
                current_section = "Advice"
                continue

            # Match exercise section
            if exercise_pattern.match(line):
                current_section = "Exercise"
                continue

            # Append content to the current section
            if current_problem and current_section:
                if current_section == "Exercise":
                    if line.startswith('- '):
                        data_dict[current_problem][current_section].append(line[2:])
                    else:
                        # Continue the last exercise item if the line does not start with '- '
                        if data_dict[current_problem][current_section]:
                            data_dict[current_problem][current_section][-1] += " " + line
                else:
                    data_dict[current_problem][current_section] += line + " "

    return data_dict


# Initialize Firebase Admin SDK with your credentials
def initialize_firebase():
    dotenv.load_dotenv()
    cred = credentials.Certificate(json.loads(base64.b64decode(os.environ["FB_ADMIN"]).decode("utf-8")))
    fb_app = firebase_admin.initialize_app(cred, name=__name__)
    db = firestore.client(fb_app)
    return db

# Function to compute and store embeddings
def compute_and_store_embeddings(db, data_dict):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #nltk.download('wordnet')


    # Process text and compute embeddings
    corpus = [preprocess_text(key) for key in data_dict.keys()]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    # Convert embeddings to list for storage
    embeddings_to_store = corpus_embeddings.cpu().numpy().tolist()
    # Store embeddings in Firestore
    db.collection('precomputed_embeddings').document('embeddings').set({'corpus': json.dumps(embeddings_to_store)})

def download_nltk_resources():
    try:
        _ = nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')


# Assuming db is already initialized and accessible
def get_precomputed_embeddings(db):
    embeddings_doc = db.collection('precomputed_embeddings').document('embeddings').get()
    if embeddings_doc.exists:
        # Load and parse the stored embeddings
        embeddings_data = embeddings_doc.to_dict()
        corpus_embeddings_list = json.loads(embeddings_data['corpus'])
        # Convert list back to tensor
        corpus_embeddings = torch.tensor(corpus_embeddings_list)
        return corpus_embeddings
    else:
        return None
    
def save_data_dict_to_firestore(data_dict, db):
    # Convert dictionary to a JSON string
    data_dict_json = json.dumps(data_dict)
    # Save JSON string to Firestore
    db.collection('application_data').document('problems_data').set({'data': data_dict_json})



def get_data_dict_from_firestore(db):
    doc_ref = db.collection('application_data').document('problems_data')
    doc = doc_ref.get()
    if doc.exists:
        data_dict_json = doc.to_dict().get('data')
        data_dict = json.loads(data_dict_json)
        return data_dict
    else:
        return None
    

# Function to format the output based on your example
def format_scenario_output(problem_description, problem_details):
    knowledge_segment = problem_details.get("Knowledge Segment", "").strip().replace('\\n', '\n')
    exercises = problem_details.get("Exercise", [])

    formatted_exercises = [exercise.strip() for exercise in exercises]

    formatted_output = {
        "Scenario": problem_description,
        "Knowledge_Segment": knowledge_segment,
        "Exercises": formatted_exercises
    }

    return formatted_output


# Usage of the function within your workflow
def process_user_query(user_query, db, model, corpus_embeddings, data_dict):

    #get semantic meaning of user_query
    user_exact_problem=query_gpt(user_query)

    if "I don't know" in user_exact_problem:
        print("This problem is pretty new for me.")
        return {}
    else:
        print("Exact Problem is : ", user_exact_problem)

    # find syntactically similar problem in dataset
    processed_problem = preprocess_text(user_exact_problem)

    # Convert the problem text to embeddings
    problem_embedding = model.encode(processed_problem, convert_to_tensor=True)
    
    
    
    
    # Compute cosine similarities
    cosine_similarities = util.pytorch_cos_sim(problem_embedding, corpus_embeddings)

    # Find the index of the highest similarity score
    max_sim_index = torch.argmax(cosine_similarities).item()
    
    # Consider using a different threshold for determining 'newness'
    if cosine_similarities[0, max_sim_index] < 0.2:
        return "This problem is pretty new for me."
    
    
    similar_problem = list(data_dict.keys())[max_sim_index]
    similar_solution = data_dict[similar_problem]
    formatted_scenario_info = format_scenario_output(similar_problem, similar_solution)
    formatted_scenario_info["Exercises"] = extract_exercise_labels(formatted_scenario_info["Exercises"]) 
    return formatted_scenario_info

def extract_exercise_labels(exercise_descriptions):
    # Pattern to capture "Exercise X" from strings
    pattern = re.compile(r"\*\*\[Exercise (\d+)(\.\d+)?\]")
    extracted_labels = []

    for description in exercise_descriptions:
        match = pattern.search(description)
        if match:
            # Construct the exercise label
            exercise_number = match.group(1)
            if match.group(2):  # Check if there's a decimal part
                exercise_number += match.group(2)
            exercise_label = f"Exercise {exercise_number}"
            extracted_labels.append(exercise_label)

    return extracted_labels



################## FUNCTIONS FROM QA #########################################
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
        #relevant_indices = [idx for idx in torch.where(similarities > 0.5)[0].tolist() if idx != most_similar_idx]
        #other_QA = [(questions[idx], answers[idx]) for idx in relevant_indices]
        return json.dumps({
            "most_relevant_Q": questions[most_similar_idx],
            "most_relevant_A": answers[most_similar_idx]
        }, indent=4 )
    else:
        return {}


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
    





if __name__ == "__main__":
    db = initialize_firebase()
    #download_nltk_resources()

    # Path to the file
    file_path = 'C:\\Users\\SaahitiP\\Desktop\\MasterThesis\\Repositories\\Extropolis\\ChatBE\\Scenario-Exercise-Mapping.md'

    # Parse the file to dictionary
    data_dict_init = parse_file_to_dict(file_path)


    #compute_and_store_embeddings(db, data_dict)

    #save_data_dict_to_firestore(data_dict_init, db)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Retrieve precomputed embeddings from Firestore
    corpus_embeddings = get_precomputed_embeddings(db)
    if corpus_embeddings is None:
        print("Failed to retrieve precomputed embeddings")

    data_dict = get_data_dict_from_firestore(db)
    if data_dict:
        print("Data Dictionary Loaded Successfully")
    else:
        print("Failed to Load Data Dictionary")



    



    response = process_user_query("i dont rly feel close to it", db, model, corpus_embeddings, data_dict)
    if response != {}:
        scenario = response["Scenario"]
        knowledge_segment = response["Knowledge_Segment"]
        scenario_exercises_to_add = response["Exercises"]

        print("Scenario: ", scenario)
        print("Knowledge Segment: ", knowledge_segment)
        print("Exercises: ", scenario_exercises_to_add)
