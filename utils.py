
import openai
import re
import os
import pandas as pd
import json
import httpx
import urllib.parse
from sentence_transformers import SentenceTransformer, util
import re
import torch
import openai
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import string

def filter_emojis(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def is_valid_text(text):
    return len(text) > 3 and sum(c.isalpha() for c in text) >= 3

def call_gpt(messages):
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    result = response.choices[0]['message']['content'].strip()

    return result


     
# DO SOME CONTENT CHECKING TO SEE IF USER PASSED ANY CONFIDENTIAL INFORMATION!!!! IF YES, DELETE CONFIDENTIAL INFORMATION FROM THE QUERY
def check_sentence(input_string,min_sentence_length=5):
    # Initialize variables
    sentence = ""
    remaining_text = ""
    found_sentence = False

    # Split input string into words
    words = input_string.split()

    # Iterate through words
    for i, word in enumerate(words):
        # Check for sentence ending punctuation
        sentence = " ".join(words[:i+1])
        
        # Regex to check for sentence-ending punctuation that is not part of a decimal number
        if re.search(r'[.!?;](\s|$)', word) and len(sentence.split(' ')) >= min_sentence_length:
            found_sentence = True

            remaining_text = " ".join(words[i+1:])
            break

    split = sentence.split(',')
    if ',' in sentence and len(split[1].split(' ')) > 2 and len(split[0].split(' ')) > 2:
        sentence = split[0] 
        remaining_text = ", ".join(split[1:]) + remaining_text

    return found_sentence, sentence, remaining_text

from transformers import GPT2TokenizerFast
GPT_TOKEN_LIMIT = 4096
class MessageTracker():
    def __init__(self) -> None:
        self.init_message = {"role": "system", "content": ""}
        self.message_queue = [] # list of dict {"role": x, "content": y}
        self.cushion_token = 1000
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def get_message_token_count(self, message):
        tokens = self.tokenizer(message)
        return len(tokens['input_ids'])+1  # role is token as well
    
    def get_token_counts(self):
        token_counts = 0
        if len(self.message_queue) > 0:
            token_counts = self.message_queue[-1][1]
        return token_counts
    
    def update(self):
        delta = self.cushion_token + self.get_token_counts() - GPT_TOKEN_LIMIT
        if delta>0:
            cut = 0
            for idx, (_, cnt) in enumerate(self.message_queue):
                if cnt > delta:
                    self.message_queue = self.message_queue[idx:]
                    cut = cnt
                    break
            for idx, (msg, cnt) in enumerate(self.message_queue):
                self.message_queue[idx] = (msg, cnt-cut)
        return     
    def append(self, message):
        prev_count = 0
        if len(self.message_queue) > 0:
            prev_count = self.message_queue[-1][1]
        self.message_queue.append((message, prev_count+self.get_message_token_count(message["content"])))
        self.update()

    def set_message_queue(self, messages, init_message):
        self.init_message = init_message
        self.cushion_token = 1000 + self.get_message_token_count(self.init_message["content"])
        self.message_queue = []
        for msg in messages:
            self.append(message=msg)
    @property
    def messages(self):
        return [self.init_message] + [x[0] for x in self.message_queue]
    
def build_classifier_dataset(actions:dict, dataset_path="./data/test_data.csv", num_samples=10):
    msg_tmp = "You are an assistant and have {action_count} actions to react to user input: {actions_str}. Generate {num_samples} difficult samples of user input and proper reaction, in the format: 'User input: input content\nProper reaction: reaction label\n\n' Do not include Response."
    actions_key = list(actions.keys())
    action_count = len(actions_key)
    actions_str = ", ".join(actions_key[:-1]) + f", and {actions_key[-1]}"

    message = [{"role": "system", "content": msg_tmp.format(action_count=action_count, actions_str = actions_str, num_samples=num_samples)}]

    print("Calling GPT4...")
    completion = openai.ChatCompletion.create( 
        model="gpt-4",
        messages=message,
    )
    print("Calling complete!")
    raw_samples = completion['choices'][0]['message']['content']+"\n\n"
    inputs = re.findall("User input: (.*?)\\n", raw_samples)
    action_label = re.findall("Proper reaction: (.*?)\\n\\n", raw_samples)
    action_id = [actions[x] for x in action_label]
    df = pd.DataFrame({'inputs':inputs, "action": action_label, "id": action_id})
    if os.path.exists(dataset_path):
        df_prev = pd.read_csv(dataset_path,index_col=0)
        print(df_prev, df)
        df = pd.concat([df, df_prev], axis="index", ignore_index=True)
        print(df)
    df.to_csv(dataset_path,index=False)
    return 


###################### changes since deployment #########################

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
    

def get_data_dict_from_firestore(db):
    doc_ref = db.collection('application_data').document('problems_data')
    doc = doc_ref.get()
    if doc.exists:
        data_dict_json = doc.to_dict().get('data')
        data_dict = json.loads(data_dict_json)
        return data_dict
    else:
        return None
    
def preprocess_text(text):
    """Preprocess text by lowercasing, removing punctuation, and lemmatizing."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)
    

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
    import dotenv
    dotenv.load_dotenv()
    openai.api_key = os.environ['OPENAI-KEY']
    dataset_path = "./data/test_data.csv"
    actions = {"chitchat":0, "news search":1, "local search":2, "elaboration on the previous topic": 3, "searching for recent info": 4, "query": 5}
    
    #pd.read_csv(dataset_path).drop(columns=['Unnamed: 0']).to_csv(dataset_path)

    for i in range(1):
        build_classifier_dataset(actions, dataset_path=dataset_path, num_samples=10)