import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
#from gpt import query_gpt
import re
import torch
from sentence_transformers import SentenceTransformer, util
import openai
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import random
from tqdm import tqdm

################### FROM FILE GPT.PY ###################################
api=""
openai.api_key =api
#This is the prompt which decides the behavioue of the model

def query_gpt(query):
    intro_prompt="""
    Prompt: Given a specific issue reported by a user, you have to see smartly that this problem is most relevant to which problem we have in Database, i have provided the database problem below, smartly identify and return the corresponding problem. Make sure you only identify a problem if you are 100 percent absolutely sure it belongs to a category. If you are even slightly uncertain about the category, return "I don't know" and nothing else.
    Make sure you are smart what you determine belongs in a category and what doesn't, and always output "I don't know" if you aren't sure. For example, you should be able to see that just because a user is laughing doesn't mean it belongs to the category related to laughter.

    Database Problems:
    User has a crisis in a personal relationships with their partner, friends, family, neighbor, or at work.
    User doesn’t feel comfortable speaking out loud to their inner child
    User feels that laughing at a situation is disrespectful, wrong, or not appropriate
    User doesn’t understand the purpose of imagining their inner child during protocols
    User doesn’t feel a connection to their child
    User is in distress or experiencing negative emotions
    User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity.
    User wants to reflect about their role in society and how society has shaped them.
    User wants to nurture opposite traits and practice creativity and affirmations

    Task: Match the user's issue smartly to the most relevant problem from the list above. And return only the statement of that problem not a single word more that that.
    
    Example 1: 
    User: "Wow 5-10 mins is long hahah" 
    You: "I don't know" 

    Example 2:
    User: "Idk it feels weird to force myself to laugh"
    You: "User feels that laughing at a situation is disrespectful, wrong, or not appropriate"

    Example 3:
    User: "Every time I get to that section where we're supposed to shift perspective, I find myself hesitating. It's like I'm supposed to turn a switch on my feelings, and it doesn't feel genuine."
    You: "User feels that laughing at a situation is disrespectful, wrong, or not appropriate"

    Example 4:
    User: "told my mate how I felt & got laughed at, like really??"
    You: "User has a crisis in a personal relationships with their partner, friends, family, or at work"

    Example 5:
    User: "Are there any exercises I can do against jealousy?"
    You: "User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity."

    Example 6:
    User: "Is there anything for anger management?"
    You: "User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity."

    Example 7:
    User: "i guess it was my fault, I need to apologize"
    You: "User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity."

    Example 8:
    User: "There's so much going wrong in the world and that brings me down"
    You: "User wants to reflect about their role in society and how society has shaped them."

    Example 9:
    User: "I've been a little more introspective lately, thinking about where things are heading."
    You: "User wants to reflect about their role in society and how society has shaped them."

    Example 10:
    User: "I feel like I'm stuck in a rut and I can't get out. Every day feels the same."
    You: "User is in distress or experiencing negative emotions"

    Example 11:
    User: "I've noticed I don't laugh as much as I used to. Things just don't seem as funny."
    You: "User is in distress or experiencing negative emotions"

    Example 12:
    User: "Stuck doesn't even begin to cover it. It's like every day is a copy of the last, in the most monotonous way possible."
    You: "User is in distress or experiencing negative emotions"

    Example 13:
    User: "Lately, I've been feeling a bit off, like I'm not fully there."
    You: "User is in distress or experiencing negative emotions"

    Example 14:
    User: "I can see now how my fear holds me back from opportunities. Looking for ways to overcome it."
    You: "User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity."

    Example 15:
    User: "This doesn't align with how I process things. It's supposed to lighten the mood, I guess, but it feels counterintuitive to me."
    You: "User feels that laughing at a situation is disrespectful, wrong, or not appropriate"
    """

    response = openai.ChatCompletion.create(
    #response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        #model="gpt-4-turbo",
        messages=[{"role": "system", "content": intro_prompt}, {"role": "user", "content": f"User Problem: {query}"}],
        max_tokens=300,
        temperature=0.3
    )
    return response.choices[0].message.content

##############################################################################



model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for semantic embeddings


nltk.download('wordnet')




def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, and lemmatizing."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def find_most_similar(problem, data_dict):
    processed_problem = preprocess_text(problem)
    corpus = [preprocess_text(key) for key in data_dict.keys()]
    
    # Convert sentences to embeddings
    problem_embedding = model.encode(processed_problem, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_similarities = util.pytorch_cos_sim(problem_embedding, corpus_embeddings)

    # Find the index of the highest similarity score
    max_sim_index = torch.argmax(cosine_similarities).item()
    
    # Consider using a different threshold for determining 'newness'
    if cosine_similarities[0, max_sim_index] < 0.2:
        return "This problem is pretty new for me."
    
    similar_problem = list(data_dict.keys())[max_sim_index]
    return similar_problem, data_dict[similar_problem]


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




########################## EVALUATION ###########################

def predict_label(user_problem):
    # model's prediction
    user_exact_problem=query_gpt(user_problem)
    res = find_most_similar(user_exact_problem, data_dict)
    if type(res) is tuple:
        similar_problem, _ = find_most_similar(user_exact_problem, data_dict)
    else:
        similar_problem = find_most_similar(user_exact_problem, data_dict)
    return similar_problem

# Function to preprocess and standardize labels in test_data such that they are the same as similar_problem
def standardize_labels(test_data):

    label_mapping = {
        "Crisis in personal relationships": "User has a crisis in a personal relationships with their partner, friends, family, or at work",
        "User uncomfortable with speaking exercises": "User doesn’t feel comfortable speaking out loud to their inner child during protocols",
        "Laughing at situations feels inappropriate": "User feels that laughing at a situation is disrespectful, wrong, or not appropriate",
        "User struggles with inner child concept" : "User doesn’t understand the purpose of imagining their inner child during protocols",
        "User is in distress or experiencing negative emotions": "User is in distress or experiencing negative emotions",
        "User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve": "User notices bad habits, toxic traits, or anti-social behavior in themselves and wants to improve. User wants to practice creativity.",
    }

    standardized_data = []
    for utterance, label in test_data:
        # Replace the label with its standardized version if it's in the mapping
        standardized_label = label_mapping.get(label, label)  # Use the original label if no mapping is found
        standardized_data.append((utterance, standardized_label))
    return standardized_data


def evaluate_model(test_data):

    print("length of test_data is ", len(test_data))

    test_data = [(data, label) for data, label in test_data if label != "User can’t see improvement" or label != "User can't see improvement"]


    # shuffle examples to reduce bias
    random.shuffle(test_data)

    #map labels from test data to what they're supposed to be called so that you can evaluate if similar_problem is correct
    test_data = standardize_labels(test_data)
    

    y_true = [label for _, label in test_data]
    # Wrap test_data in tqdm for a progress bar
    y_pred = [predict_label(utterance) for utterance, _ in tqdm(test_data, desc="Predicting")]

    # Collect and print incorrect predictions to file 
    incorrect_predictions = [(test_data[i][0], true, pred) for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    with open('incorrect_predictions.txt', 'w', encoding='utf-8') as file:
        file.write("Incorrect Predictions:\n")
        for utterance, true, pred in incorrect_predictions:
            file.write(f"Utterance: {utterance}\nTrue Label: {true}\nPredicted: {pred}\n\n")

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Model precision: {precision:.2f}")
    print(f"Model recall: {recall:.2f}")
    print(f"Model F1 score: {f1:.2f}")


def read_test_data_from_markdown(file_path):
    """
    Reads test data from a markdown file and prepares it for the evaluation model.
    
    Args:
    - file_path: str, the path to the markdown file containing the test data.
    
    Returns:
    - test_data: list of tuples, where each tuple contains a user utterance and its scenario label.
    """
    test_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip markdown table headers and separators
            if line.startswith("| User Utterance ") or line.startswith("| ---"):
                continue
            
            # Extract the utterance and label from the line
            try:
                utterance, label = line.strip().split("|")[1:3]
                # Remove smart quotes, standard quotes, and leading/trailing spaces
                utterance = utterance.replace('“', '').replace('”', '').replace('"', '').strip()
                #utterance = utterance.strip().strip('"')
                label = label.strip()
                
                if utterance and label:
                    test_data.append((utterance, label))
            except ValueError:
                # This handles any lines that don't match the expected format
                print(f"Skipping line: {line.strip()}")
                continue

    return test_data





##################################################################


if __name__ == "__main__":
    # Path to the test data file
    test_file_path = 'test_data_user_utterances.md'

    # Read the test data from the markdown file
    test_data = read_test_data_from_markdown(test_file_path)


    # Path to the file
    file_path = 'Scenario-Exercise-Mapping.md'

    # Parse the file to dictionary
    data_dict = parse_file_to_dict(file_path)

     # Evaluate the model
    evaluate_model(test_data)


    # Example usage
    while(True):
        user_problem = input("\n\nPlease Enter your Problem here: ")
        user_exact_problem=query_gpt(user_problem)
        if "I don't know" in user_exact_problem:
            print("This problem is pretty new for me.")
        else:
            print("Exact Problem is : ", user_exact_problem)
            similar_problem, similar_solution = find_most_similar(user_exact_problem, data_dict)
            print(similar_solution)