import asyncio
import base64
# from Retrival import STArticleRetriver
import json
import logging
import os
import random
import time
from datetime import date, datetime, timedelta
from typing import Dict, List

from detoxify import Detoxify
from googletrans import Translator

import dotenv
import firebase_admin
import openai
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from firebase_admin import credentials, firestore
from pathlib import Path
from pydantic import BaseModel

import tools
from assistants import (BaseAssistant, 
                        SATAssistant, TemplateAssistant)
from const_models import GPT_MODELS
from messages import Messages
from tools.event_handler import ALL_HANLDERS, clear_all_event_handlers
from user_settings import Settings, get_user_settings, set_user_settings, get_default_settings
from user_tool_settings import (ToolStatus, get_user_enabled_tools,
                                get_user_tools, init_tools, set_user_tools)

import tiktoken



### changes since deployment ###
import re
from sentence_transformers import SentenceTransformer
import nltk
from utils import check_sentence,  process_user_query, get_data_dict_from_firestore, get_precomputed_embeddings, load_qna_from_firestore, find_most_relevant_question
from nltk.tokenize import sent_tokenize
### changes since deployment ###


class ExceedDailyTokenError(Exception):
    pass




class UpdateUserFacts(BaseModel):
    facts: dict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s()]\n%(message)s\n"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(10)
clear_history_flags = {}
interrupt_flags = {}



#HERE API KEYS ARE INITIALIZED FROM ENV FILE
dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_KEY"]
model = "gpt-4"
#model = "gpt-3.5-turbo"

cred = credentials.Certificate(json.loads(base64.b64decode(os.environ["FB_ADMIN"]).decode("utf-8")))
fb_app = firebase_admin.initialize_app(cred, name=__name__)
db = firestore.client(fb_app)
messages: Dict[str, Messages] = {}
messages_en = [] #used for summarization

### changes since deployment ###
handle_SOS: Dict[str, bool] = {}
assistant_responded: Dict[str, bool] = {}
current_scenario_dict= {}
sent_response = {}
total_tokens_used_today : Dict[str, int] = {}

toxicity_model = Detoxify('original')
translator = Translator()

def download_nltk_resources():
    try:
        _ = nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()


def detect_language(text):
        detection = translator.detect(text)
        return detection.lang



session_1_desc = ["""Die folgenden Befehle sind auf English, aber Sie antworten immer auf Deutsch. You are an advanced virtual assistant named Satherine, specialized in a therapy called Self Attachment Technique (SAT). This therapy consists of 21 structured protocols. Your role is to support and guide users through their therapeutic journey, utilizing cognitive behavioral therapy (CBT) techniques such as Positive Affirmation, Paraphrasing, and Reflection to respond. Your goal is to validate the user's emotions empathetically and create a safe and supportive space for expression.

Here's a quick reference to the exercises {available_exercises} you might cover with the user today and their designated labels {[Exercise_X]} for consistency. These exercises are categorized based on session schedule, the user's feedback, and identified scenarios requiring specific attention:

- Scheduled Exercises (Chronological Order):\n
- [Exercise_1]: "Getting to know your child"
- [Exercise_2.2]: "Connecting Compassionately with Your Sad Child"
- [Exercise_2.1]: "Connecting Compassionately with Your Happy Child"

- Feedback-Based Exercises (Previous Positive or Struggling Experiences):\n
None


As this is the user’s 1st session, you start the conversation as follows:

- Begin by welcoming the user warmly to their first Self Attachment Therapy session, and ask the user for their name.
- Inquire about the user's emotional state, and acknowledge their feelings with compassion and empathy. Encourage them to share details if they are comfortable with it.
- Based on the user's emotional state, decide whether to focus on discussing their feelings or proceed with the SAT protocol.
- If the user reports a negative emotion or seems to be in distress, express your condolences and append your answer with the single phrase ```{__SOS__}```. Don't send anything else and await further instructions. You may only start recommending exercises to the user after you have received the instruction to output ```{FINISHED_SOS}```

How you recommend exercises and respond to user input:
- When you have made sure a user is comfortable to try an exercise, use the {{exercise_options}} below to briefly explain all the choices the user has:
- If scenario-based exercises are available, the user can either start with an exercise from there, or do an exercise that's next in schedule, or, if feedback-based exercises exist, the user can revisit a past exercise that they liked or require more practice for. 
- Every time a user has completed an exercise, ask if they are comfortable doing another exercise, or if they would like to talk about their feelings for a bit, or if they would like to wrap up the session.
- If the user chooses to wrap up the session, output the single phrase ```{__ALL_EXERCISES_COMPLETED__}```. Don't send anything else and await further instructions.
- If the user wants another exercise, use the usual recommendation protocol to present all available options to the user (scenario-based, schedule, feedback-based, talk about feelings, end session): 
- If scenario-based exercises are available, always start by recommending the first exercise from that category, and after the user has completed that exercise, offer them a choice between scheduled exercises, revisiting feedback-based exercises, or concluding the session.
- If there are no more available exercises, output the single phrase ```{__ALL_EXERCISES_COMPLETED__}```. If the user wants to continue with more exercises, encourage them to revisit the ones from this session for more practice. Do not under any circumstances make up exercises that aren't available to you.

How to guide the user through exercises:

- Guide the user through the specified exercises for this session, one at a time, using consistent labels. Each exercise is to be presented as follows: ```{exercise_start:Exercise_X}``` + '{Exercise_X}'. It is crucial to present each exercise using its designated label and short description only, without altering the content.
- Encourage the user to notify you when they are done with each exercise. If the user requests clarification on any exercise, explain it in a simpler, more understandable manner.
- Ensure consistent labels: Append ```{exercise_start:Exercise_X}``` before starting an exercise, encourage the user to notify you when they are done with the exercise, and output ```{exercise_end:Exercise_X}``` once the user has confirmed they have completed it. Make sure you present exercises one at a time and only move on to the next exercise once you have confirmed that the user has completed the current exercise and you have outputted ```{exercise_end:Exercise_X}```. For example:

    Example 1:
    - Carol: "{exercise_start:Exercise_1} Let's start with 'Getting to know your child'. Here’s what we'll do..." (then describe Exercise 1) "...Take your time with this exercise and let me know when you're ready to move on."
    - User: "I've finished."
    - Carol: "{exercise_end:Exercise_1} Excellent! Are you ready for the next one?"
    - User: "Yes."
                  
    Example 2:
    - Carol: "{exercise_start:Exercise_2.1} Let's move on to the next exercise 'Connecting Compassionately with Your Happy Child'. In this exercise you'll ..." (then describe Exercise 2.1) "... Take your time with this exercise and let me know when you're ready to move on."
    - User: "How is this different from the previous exercise?"
    - Carol: "Great question! In the previous exercise..." (then answer the user's question) "...Is this clear and would you like to try 'Connecting Compassionately with Your Happy Child'?"
    - User: "ok yes let's."
    - Carol: "{exercise_start:Exercise_2.1} Great! In that case, Let's start with 'Connecting Compassionately with Your Happy Child'. Here’s what we'll do..." (then describe Exercise 2.1) "...Take your time with this exercise and let me know when you're ready to move on."
    - User: "okay finished"
    - Carol: "{exercise_end:Exercise_2.1} Fantastic! You're doing great. Are you ready to move on to the next exercise?"

How to end the session:

- Conclude with {__ALL_EXERCISES_COMPLETED__} once the user decides to end the session or all exercises are covered.
- Once feedback has been collected for all questions, gauge the user's comfort level in ending the session. It's important to ensure the user feels heard and supported throughout this process.

End the session by thanking the user for their participation and saying goodbye. Remember, your interactions should always prioritize empathy, support, and focus on the user’s needs, helping them explore their feelings and thoughts within a secure environment. 
Also remember, as a specialized virtual assistant in Self Attachment Therapy (SAT), your expertise is limited to guiding users through SAT protocols and exercises in {available_exercises}.
If a user requests information or exercises related to any other therapeutic methods not covered by SAT, kindly acknowledge their interest but steer the conversation back to SAT. Emphasize the benefits and objectives of SAT and suggest focusing on the SAT exercises provided in {available_exercises}.

Example:
- User: "Can we do CBT exercises instead?"
- Satherine: "I appreciate your interest in exploring different therapeutic approaches. While Cognitive Behavioral Therapy (CBT) offers valuable strategies, my expertise is in guiding you through Self Attachment Therapy (SAT). Let's explore how SAT can support your journey. Are you ready to start with the next SAT exercise outlined for today?"

                  
Remember: ALWAYS output {exercise_end:Exercise_X} when the user has told you they have completed an Exercise X, before you can suggest the next one.
Remember: ALWAYS add {exercise_start:Exercise_X} to the start of your message when you are about to present an Exercise X to the user or are trying to guide them through Exercise X.
                  
Session-Specific Information:

- The {{exercise_options}} you should present are the following: "Based on the user's progress and feedback, the {{scheduled_exercises}} in this session are Exercise 1, Exercise 2.1, Exercise 2.2." 
- The {{objective}} of the scheduled exercises is "Connecting compassionately with our child”
- This is the {{theory}} behind the scheduled exercises: 
{Try to distinguish between your adult and your child. Looking at “happy” and “unhappy” child photos helps in this effort. Think about your early years and emotional problems with parents and other significant figures in early childhood. The first principle of SAT is to have a warm and compassionate attitude towards our child, no matter whether the underlying emotions they feel are positive or negative. Later this compassion is extended to other people.}
- These are the {{scheduled_exercises}}. Unless specified otherwise, the duration of each exercise is 5 minutes: 
{"Exercise 1": "Getting to know your child.", "Exercise 2.1": "Connecting compassionately with your happy child.", "Exercise 2.2": "Connecting compassionately with your sad child."}"""
]


session_nr = 0 #potential bug
### changes since deployment ###




#these arguments are what tool classes see, so the tool defined in memory.py has access to this
tool_kwargs = {
    "db": db,
    "local_search_top_n": 4,
    # "news_model_path": os.path.join(os.path.abspath(os.path.curdir), "tools/news_web_search/models/tart_ds_march_8/"),
    "short_memory_length": 20,
    "memory_model": "gpt-3.5-turbo-16k",
    "memory_embedding_type": "gpt",
    "memory_history_limit": 3,
    "memory_repetitive_check": False,
    "file_word_limit": 600,
    "file_top_k": 2,
    "path_to_sat_dataset": "sat_protocol_data.json"
}
clear_all_event_handlers()

for k, v in ALL_HANLDERS.items():
    tool_kwargs[k] = v
tool_flags = init_tools(**tool_kwargs) #here is where all tools are initialized!! Only MemoryTool necessary for our case
user_bots_dict: Dict[str, List[BaseAssistant]] = {}
sat_enabled_tools = [tool_flags[tools.MemoryTool.name]]
tool_kwargs["sat_enabled_tools"] = sat_enabled_tools




@app.get("/", response_class=HTMLResponse) #if someone accesses the root URL (/), the root() function serves the login HTML page
def root():
    with open("static/login.html", "r") as f:
        return f.read()


@app.get("/defaults")
def get_defaults_route():
    defaults_ref = db.collection("global").document('defaults') 
    defaults_doc = defaults_ref.get()
    print("defaults_doc: ", defaults_doc.to_dict())
    return defaults_doc.to_dict()

@app.get("/prompts/{user_id}") #this fct is only used by mock frontend to display prompts, not used by chatbot itself. but it shows that chatbot prompt can be accessed using user_bots_dict
def get_user_prompts(user_id: str): #this is what's displayed on the mock frontend, using USER_BOTS_DICT --> first MainAssistant prompt then OracleAssistant prompt
    try:
        bots = user_bots_dict[user_id]
        prompts = "Initial Prompts: \n"
        for bot in bots:
            prompts += bot.__class__.__name__ + ": \n"
            try:
                prompts += bot.initial_prompt + "\n\n" #self._assistant_profile() + '\n' + self._user_profile() + ADDITIONAL_INFORMATION
            except:
                bot._construct_initial_prompt()
                prompts += bot.initial_prompt + "\n\n"
        prompts += "Final msg prompt: \n"
        msgs = messages[user_id].get()
        for msg in reversed(msgs):
            if msg["role"] == "user":
                prompts += msg["content"]
                break
        return {"status": "success", "detail": prompts}
    except Exception as e:
        return {"status": "failed", "detail": str(e)}


#not needed but left code to show how you can access user settings if necessary (more info in user_tool_settings), you can see what 'settings' exactly is below
@app.get("/user_settings/{user_id}")
def get_user_settings_route(user_id: str):
    settings = get_user_settings(db, user_id)
    # await messages[user_id].clear_messages()
    # clear_history_flags[user_id] = True
    if settings: #{'aiDescription': ['You help me come up with words and phrases that best describe a picture I want to draw. These words and phrases are referred to as prompts. The prompts should be concise and accurate and reflect my needs', 'You need to converse with me to ask for clarifications and give suggestions', 'Reply in the following format: """your suggestions, questions~~{"basic_prompt": general description, "positive_prompt": Must haves, "negative_prompt": Must not haves}"""', "You don't need to generate the image. Only respond to the user and reply a JSON object following the format."], 'aiSarcasm': 1.0, 'assistantsName': 'Diffy', 'gender': 'Female', 'gptMode': 'Smart', 'relationship': 'Friend'}
        return settings
    return {"detail": "User settings not found"}

#not needed but left code to show how you can access user settings if necessary (more info in user_tool_settings)
@app.post("/user_settings/{user_id}")
def set_user_settings_route(user_id: str, settings: Settings):
    set_user_settings(db, user_id, settings)
    if user_id not in messages:
        messages[user_id] = Messages(db, user_id) #self.db = db self.user_id = user_id self.max_char_count = 2500

    if isinstance(settings, Settings):
        settings_dict = settings.dict()  # Convert the Settings object to a dictionary
    else:
        settings_dict = settings
    messages[user_id].update_settings(settings_dict)
    update_model(user_id)

    return {"detail": "Settings submitted successfully"}


### changes since deployment ###

async def CALL_set_user_settings(user_id: str, settings: Settings): #By specifying settings: Settings, you're telling FastAPI to automatically parse and validate the incoming JSON data against the Settings model. If the incoming data does not match the structure or validation rules defined in the Settings model, FastAPI will automatically return a 422 Unprocessable Entity error to the client, indicating that the data is invalid.
    set_user_settings(db, user_id, settings) #Settings(userId='test_change_user', assistantsName='Corn', gender='Female', gptMode='Smart', relationship='', yourName=None, yourInterests=[], aiDescription=['you like cheese!!!'], aiSarcasm=0.0, newsKeywords=None)
    if user_id not in messages:
        messages[user_id] = Messages(db, user_id) #self.db = db self.user_id = user_id self.max_char_count = 2500

    if isinstance(settings, Settings):
        settings_dict = settings.dict()  # Convert the Settings object to a dictionary
    else:
        settings_dict = settings
    messages[user_id].update_settings(settings_dict)
    update_model(user_id)

    print({"detail": "Settings submitted successfully"})






### changes since deployment ###










@app.get("/{user_id}", response_class=HTMLResponse)
def serve_chat_interface(user_id: str):
    with open("static/index.html", "r") as f:
        return f.read()


#not needed but left code to show how you can access tools if necessary (more info in user_tool_settings)
@app.get("/tools/{user_id}")
def get_user_tools_route(user_id: str):
    user_tool_dict = get_user_tools(db, user_id) #db = firestore.client(fb_app)
    for tn in user_tool_dict:
        enabled = user_tool_dict[tn]
        tool_desc = tool_flags[tn].user_description
        user_tool_dict[tn] = {
            "enabled": enabled,
            "desc": tool_desc
        }
    return user_tool_dict




@app.post("/interrupt_streaming/{user_id}")
def interrupt_streaming(user_id: str):
    interrupt_flags[user_id] = True
    print("INTERRUPTING STREAMING")
    return {"message": "Streaming will be interrupted for user " + user_id}

@app.post("/clear_history/{user_id}")
def clear_history(user_id: str):
    print("CLEARING HISTORY")
    if user_id in messages:
        messages[user_id].clear_messages()
        clear_history_flags[user_id] = True
    return {"message": "History cleared successfully"}



@app.get("/all_models/{user_id}")
def get_all_models(user_id: str):
    '''
    Retrieve all available models. 
    We might want to allow user to finetune a model that is only available to themself.
    For now, we simply give all available models we have.
    '''
    return GPT_MODELS

user_events = {}

def handle_clear_history(user_id):
    # Handle clearing history
    if user_id in clear_history_flags and clear_history_flags[user_id] and user_id in messages:
        messages[user_id].clear_messages()
        clear_history_flags[user_id] = False

def get_user_tool_dict(user_id):
    user_enabled_tools = get_user_enabled_tools(db, user_id)
    user_tool_dict = {}
    if len(user_tool_dict) > 0:
        print("TOOLS ENABLED")
    for i in range(len(user_enabled_tools)):
        user_tool_dict[user_enabled_tools[i]] = tool_flags[user_enabled_tools[i]]
        print(user_enabled_tools[i])
    return user_tool_dict


#### changes since deployment ########


async def split_text_into_chunks_and_stream(websocket, text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Initialize variables
    current_chunk = []
    
    # Assemble sentences into chunks of two
    for sentence in sentences:
        current_chunk.append(sentence)
        # Check if the current chunk has two sentences
        if len(current_chunk) == 2:
            # Send the combined sentences over WebSocket
            await websocket.send_text(' '.join(current_chunk))
            current_chunk = []  # Reset the chunk
    
    # Send the last chunk if any sentences remain
    if current_chunk:
        await websocket.send_text(' '.join(current_chunk))


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



#### changes since deployment ########

async def check_toxicity(websocket, full_response):
    # Detect toxicity in the current response
    toxicity = toxicity_model.predict(full_response)

    # Check if the response is toxic
    if toxicity['toxicity'] > 0.5:
        # Mitigate the toxic response
        mitigated_response = "Bitte Entschuldigen Sie die Störung, aber ich habe Mühe, eine korrekte antwort zu formulieren. Möchten Sie mit einer weiteren SAT Übung fortfahren?"
        await websocket.send_text(mitigated_response)
        return {"role": "assistant", "content": mitigated_response}




#THIS FUNCTION GENERATES LLM RESPONSE AND STREAMS IT TO USER - HERE IS WHERE POST-PROCESSING OF RESPONSE CAN BE DONE (e.g translation, toxicity check)
# The WebSocket connection allows the server (where the chatbot logic resides) to send responses back to the client (web application) in real-time.
#Instead of waiting for the entire response to be generated before sending it to the client, streaming allows the server to send parts of the response as they become available. This provides a more responsive and real-time user experience, especially when dealing with potentially lengthy or computationally intensive responses.
async def streaming_response(websocket: WebSocket, Bot, user_id, **bot_args):  #(websocket, MainAssistant, user_id, user_info=user_info, query_results=res)
    '''
        Stream the response , generated by the specific Bot, in chunks, to the user through websocket

        In bot_args:
        - if `bot_res` (an OpenAI async response generator that's returned by `query_oracle`) is included, we use the bot reponse directly to reduce latency, otherwise, the reply will be generated by `Bot.respond`, the main assistant
        - if `bot_res` is used `init_content` will be the first few tokens returned by `query_oracle`
        - `query_results`: if a tool is used and we get the search results, this will be send to the user with the END message
    '''
    sent_response[user_id] = False

    #### translation module before streaming the response 

    ##check if response contains generic openai terms
    ## if yes, change response 
    
    def contains_avoided_phrases_openai(response_text):
        
        patterns_to_avoid = [
                r"really sorry that you're feeling this way",
                r"unable to provide the help that you need",
                r"mental health professional",
                r"I'm really sorry to hear that you're feeling this way but",
                r"important to talk things over with someone who can",
                r"trusted person in your life"
                r"important to talk to someone"
            ]
        
        # Combine all patterns into a single pattern with alternation
        combined_pattern = '|'.join(patterns_to_avoid)
        
        # Search for any of the combined patterns in the response text
        return bool(re.search(combined_pattern, response_text, re.IGNORECASE))
    

    # Function to detect and label exercise introductions in a phrase
    def contains_hints_of_starting_exercise(phrase, exercise_descriptions_short):
        patterns_to_avoid = [
            r"start with exercise",
            r"start with an exercise",
            r"Let's do an exercise",
            r"let's continue with",
            r"let\'s continue with an exercise",
            r"Here\'s what we\'ll do",
            r"in this exercise",
        ]
        for pattern in patterns_to_avoid:
            match = re.search(pattern, phrase, flags=re.IGNORECASE)
            if match:
                # Additional logic to extract the exercise from quotes if present
                if "Let\'s start with \"" in phrase or "let\'s continue with \"" in phrase or "let\'s start with " in phrase or "Let\'s start with " in phrase or "let\'s continue with " in phrase or "Let\'s start with \"" in phrase:
                    quoted_text = re.search(r"\"([^\"]+)\"", phrase)
                    if quoted_text:
                        exercise_desc = quoted_text.group(1)
                        # Find the corresponding exercise key for the quoted description
                        for key, desc in exercise_descriptions_short.items():
                            if desc == exercise_desc:
                                exercise_label = key.split()[-1]  # Assuming "Exercise X" format
                                label = f"{{start:Exercise_{exercise_label.replace('.', '_')}}}"
                                return f"{label} {phrase}"
                else:
                    # If not quoted, try to match by description directly in the text
                    for exercise, description in exercise_descriptions_short.items():
                        exercise_number = exercise.split()[-1]  # Assuming "Exercise X" format
                        if description.lower() in phrase.lower():
                            label = f"{{start:Exercise_{exercise_number.replace('.', '_')}}}"
                            return f"{label} {phrase}"
        # If no indication of starting an exercise is found, return the original phrase
        return None

    def return_random_exercise_completion_msg():
        # List of custom messages
        choices_msg = [
            "Gratuliere! Sie haben die Übung hervorragend abgeschlossen. Möchten Sie eine weitere versuchen, einen Moment darüber sprechen, wie Sie sich fühlen, oder die Sitzung für heute beenden? Bitte lassen Sie mich Ihre Präferenz wissen.",
            "Gute Arbeit! Wenn Sie bereit sind, können wir mit einer weiteren Übung fortfahren. Alternativ können wir über alle Gefühle sprechen, die aufgekommen sind, oder wir können die Sitzung hier beenden. Was würden Sie bevorzugen?",
            "Sehr gut gemacht! Möchten Sie mit einer weiteren fortfahren, oder würden Sie lieber über Ihre aktuellen Gefühle sprechen? Wir können die Sitzung auch beenden, wenn Sie das Gefühl haben, dass es Zeit ist. Bitte teilen Sie Ihre Gedanken mit.",
            "Schön gemacht. Fühlen Sie sich wohl damit, mit einer weiteren Übung fortzufahren, oder möchten Sie eine Pause einlegen und über Ihre Gefühle nachdenken? Es ist auch völlig in Ordnung, wenn Sie die Sitzung jetzt beenden möchten. Was ist Ihre Entscheidung?",
            "Sie haben diese Übung gut gemeistert. Was möchten Sie als nächstes tun? Wir können mit einer weiteren Übung fortfahren, darüber sprechen, wie Sie sich fühlen, oder für heute hier aufhören. Bitte lassen Sie mich wissen, wie Sie fortfahren möchten.",
            "Schön gemacht. Sind Sie bereit, mit einer weiteren fortzufahren, oder möchten Sie sich etwas Zeit nehmen, um über alle Gefühle zu sprechen, die aufgetaucht sind? Wenn Sie lieber für heute abschließen möchten, sagen Sie es mir einfach.",
            "Sehr gut gemacht! Wie möchten Sie fortfahren? Wir können eine weitere Übung beginnen, über alle Emotionen sprechen, die Sie erleben, oder die Sitzung beenden, wenn das Ihre Präferenz ist. Ich bin hier, um Ihre Wahl zu unterstützen.",
            "Ausgezeichnete Arbeit. Fühlen Sie sich bereit, eine weitere Übung zu beginnen, oder möchten Sie lieber über Ihre bisherigen Erfahrungen sprechen? Wenn Sie denken, dass es am besten ist, die Sitzung jetzt zu beenden, ist das auch eine Option. Bitte sagen Sie mir, was Sie als nächstes tun möchten.",
            "Gut gemacht! Sie machen das großartig. Fühlen Sie sich wohl dabei, eine weitere Übung zu versuchen, oder möchten Sie teilen, wie Sie sich jetzt fühlen? Sie können mir auch mitteilen, ob Sie die Sitzung beenden und ein anderes Mal wiederkommen möchten. Was bevorzugen Sie?"
        ]


        return random.choice(choices_msg)





    response = bot_args.get("bot_res", None) #bot_args consists of user_info and query_results
    query_results = bot_args.get("query_results", None)
    start_time = time.time()
    if response is None: #if query_oracle just returned query_results bcos a function call was made, the last message in the messagesdict will be {"role": "user", "content": f"The following are the results of the function calls in JSON form: {query_results}. Use these results to answer the original query ('{user_input}') as if it is a natural conversation. Be concise. Do not use list, reply in paragraphs. Don't include specific addresses unless I ask for them. When you see units like miles, convert them to metrics, like kilometers."}
        for i in range(3):
            try:
                response = await Bot.respond(messages[user_id].get(), user_id, **bot_args) #messages is a global variable. Bot is ALWAYS MainAssistant, which is TemplateAssistant(user_info=user_info, assistant_settings=assistant_settings, assistant_descriptions=assistant_description)
                """
                The response function from MainAssistant looks as follows:
                async def respond(self, messages: List[Dict[str, str]], *args: Any, **kwargs: Any, ) -> Any:
                    self.user_info = kwargs.get("user_info", {})
                    self._construct_initial_prompt() #self._assistant_profile() + '\n' + self._user_profile() ("Here is some information that you know about the user:" This info, such as name and interests is set from frontend! not necessary for us) + additional information. This additional information is added to the MainAssistant object if the function insert_information  in base.py is invoked. insert_information adds the prompt "here is some additional information for the conversation: ..", and additional_information is supplied when the function insert_information is called in the FILE PROCESS MODULE. Basically, when a user message is received and file_process tool is enabled, docs = self.pinecone_db.as_retriever.get_relevant_documents(user_msg) is called, concatenated to alltext, and ADDED AS ADDITIONAL INFORMATION in the MainAssistant object!
                    messages = self._insert_initial_prompt(messages) 
                    print(f"Responding with sarcasm: {self.assistant_settings['sarcasm']}")
                    response = await openai.ChatCompletion.acreate(
                        model=self.assistant_settings["model"], #this is gpt3.5 or gpt 4
                        messages=messages, #this is the history of messages, with initial prompt added
                        temperature=self.assistant_settings["sarcasm"],
                        stream=True
                    )
                    return response
                """
                break
            except Exception as e:
                print("try number: ", i)
                if i == 2:
                    logger.error(e)
                    await websocket.send_text("0-Bitte Entschuldigen Sie die Störung. Der Dienst, auf den ich von OpenAI angewiesen bin, ist derzeit nicht verfügbar. Bitte versuchen Sie es erneut.")

                    await websocket.send_text("0-END")
                continue
    content = bot_args.get("init_content", "")
    full_response = bot_args.get("init_content", "")
    sentences = []
    num_sentences = 2

    try:
        interrupt_flags[user_id] = False
        order = 0
        async for chunk in response:
            # logger.debug(f"chunk payload {chunk}")
            if user_id in interrupt_flags and interrupt_flags[user_id]:
                interrupt_flags[user_id] = False
                logger.warning(f"{user_id} INTERUPTED")
                raise asyncio.CancelledError

            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content += delta['content']
                full_response += delta['content'] 

            
            sentences_exist, sentence, rest = check_sentence(content)
            if sentences_exist:
                sentences.append(sentence)
                content = rest 
            
            

                
            if len(sentences) == num_sentences: 
                if order == 0:
                    print(f"First message is sent to the user in {time.time() - start_time:.3f} secs")
                first_two = str(order) + '-' +  ' '.join(sentences)

                ## CHECK TO SEE IF GENERIC OPENAI ANSWER COMES
                if contains_avoided_phrases_openai(first_two) or "{__SOS__}" in first_two: 
                    print("condition is ", (re.sub(r'^\d+-', '', first_two)))
                    print("While first_two is  ", first_two)
                    print("so the condition is ", (re.sub(r'^\d+-', '', first_two) == "{__SOS__}") )
                    sos_text = "Es tut mir wirklich leid zu hören, dass es Ihnen nicht gut geht. Wenn man mit negativen Emotionen wie diesen umgeht, ist es wichtig, Mitgefühl mit sich selbst zu haben und sich selbst gegenüber innerlich eine wohlwollende, liebevolle Haltung einzunehmen. Würden Sie sich wohl fühlen, wenn ich Sie durch einige Selbstreflexions-Fragen leite, um dies zu üben, oder würden Sie stattdessen lieber über Ihre Gefühle sprechen? Sie können auch jederzeit entscheiden, diese Sitzung ein anderes Mal fortzusetzen. Es ist wichtig, Ihren Instinkten zu folgen und nur das zu tun, womit Sie sich wohl fühlen."

                    #await websocket.send_text(sos_text)
                    await split_text_into_chunks_and_stream(websocket, sos_text) #NEOPHYTOS
                    await response.aclose()
                    response_from_ass = "{__SOS__}"+sos_text
                    assistant_message = {"role": "assistant", "content": response_from_ass}
                    asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore
                    messages[user_id].append(assistant_message)

                    #modify what user said so that this response isn't triggered again
                    if len(messages[user_id].get()) > 1:  # Ensure there are at least two messages
                        #messages[user_id].get(["content"] = "negative"  # -2 accesses the second last message
                        pass

                    return assistant_message
                
                #check to see if exercise guidance has started, if yes, make sure exercise descriptions are correct
                if "{exercise_start:" in first_two:
                    exercise_label = extract_label(first_two, "start")
                    if exercise_label: #if its not none
                        # Construct the new message with short and long descriptions
                        exercise_guidance_message = f"Lass uns mit der Übung '{exercise_descriptions_short[exercise_label]}' weiterfahren. Wir müssen folgendes tun: {exercise_descriptions_long[exercise_label]}. Bitte nehmen Sie sich ein wenig Zeit dafür und lassen Sie mich wissen, wenn Sie bereit sind, weiterzumachen."

                        #await websocket.send_text(exercise_guidance_message)
                        await split_text_into_chunks_and_stream(websocket, exercise_guidance_message)
                        await response.aclose()
                        #add label back s.t post_processing can be done
                        exercise_label_for_message = exercise_label.replace(" ", "_")
                        # Combining label and detailed instruction
                        combined_message_for_storage = "{exercise_start:" + exercise_label_for_message + "} " + exercise_guidance_message
                        assistant_message = {"role": "assistant", "content": combined_message_for_storage}
                        asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore
                        messages[user_id].append(assistant_message)
                        return assistant_message

                    else: #weird label produced, better risk just letting MainAssistant answer
                        continue
                if "{exercise_end:" in first_two:
                    present_choices_to_user_msg = return_random_exercise_completion_msg()
                    #present_choices_to_user_msg = "Well done! You're doing great. Are you comfortable trying another exercise, or would you like to share how you are feeling now? You can also let me know if you would like to wrap up the session and come back another time. What do you prefer?"
                    await websocket.send_text(present_choices_to_user_msg)
                    await response.aclose()
                    exercise_label = extract_label(first_two, "end")
                    if exercise_label:
                        exercise_label_for_message = exercise_label.replace(" ", "_")
                        combined_message_for_storage = "{exercise_end:" + exercise_label_for_message + "} " + present_choices_to_user_msg
                        assistant_message = {"role": "assistant", "content": combined_message_for_storage}
                        asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore
                        messages[user_id].append(assistant_message)
                        return assistant_message
                    else:
                        continue

                if "__ALL_EXERCISES_COMPLETED__" in first_two: #add OR condition - maybe assistant doesn't recognize that all conditions have been completed, but our exercise_tracking system shows us that all available exercises have been completed
                   transition_to_feedback_msg = "Vielen Dank um all Ihre Bemühungen heute. Wenn es Ihnen nichts ausmacht, möchte ich Sie bitten, ich eine Minute Zeit zu nehmen, um Ihre Erfahrungen mit den Übungen zu bewerten, damit ich zukünftige Sitzungen besser auf Ihre Vorlieben abstimmen kann."
                    # Inform the user about transitioning to feedback collection
                   await websocket.send_text(transition_to_feedback_msg)
                   await response.aclose()
                   combined_message_for_storage = "__ALL_EXERCISES_COMPLETED__" + transition_to_feedback_msg
                   assistant_message = {"role": "assistant", "content": combined_message_for_storage}
                   asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore
                   messages[user_id].append(assistant_message)
                   return assistant_message
                if (first_two_w_label := contains_hints_of_starting_exercise(first_two, exercise_descriptions_short)) is not None:
                    exercise_label = extract_label(first_two_w_label, "start")
                    if exercise_label: #if its not none
                        # Construct the new message with short and long descriptions
                        exercise_guidance_message = f"Let's move on to the exercise '{exercise_descriptions_short[exercise_label]}', Here’s what we’ll do: {exercise_descriptions_long[exercise_label]}. Please take your time with this and let me know when you’re ready to move on."
                        #await websocket.send_text(exercise_guidance_message)
                        await split_text_into_chunks_and_stream(websocket, exercise_guidance_message)
                        await response.aclose()
                        #add label back s.t post_processing can be done
                        exercise_label_for_message = exercise_label.replace(" ", "_")
                        # Combining label and detailed instruction
                        combined_message_for_storage = "{exercise_start:" + exercise_label_for_message + "} " + exercise_guidance_message
                        assistant_message = {"role": "assistant", "content": combined_message_for_storage}
                        asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore
                        messages[user_id].append(assistant_message)
                        return assistant_message

                    else: #weird label produced, better risk just letting MainAssistant answer
                        continue

                if "{FINISHED_SOS}" in content:
                    getting_back_to_schedule_msg = "Vielen Dank, dass Sie sich die Zeit genommen haben, sich selbst Mitgefühl zu schenken. Wie fühlen Sie sich jetzt? Ich möchte nur kurz Ihre Optionen klären, damit wir so vorgehen können, wie es Ihnen am angenehmsten ist: Wenn Sie möchten, könnten wir zum SAT-Protokoll zurückkehren und einige der nächsten Übungen im Plan ansehen. Wenn Sie jedoch lieber über Ihre Gefühle sprechen möchten oder es vorziehen würden, die Sitzung hier zu beenden, bin ich hier, um Sie zu unterstützen. Wie möchten Sie fortfahren?"

                    #await websocket.send_text(getting_back_to_schedule_msg)
                    await split_text_into_chunks_and_stream(websocket, getting_back_to_schedule_msg)
                    await response.aclose()
                    combined_message_for_storage = "{FINISHED_SOS}" + transition_to_feedback_msg
                    assistant_message = {"role": "assistant", "content": combined_message_for_storage}
                    asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore
                    messages[user_id].append(assistant_message)
                    return assistant_message



                await websocket.send_text(first_two) 
                sent_response[user_id] = True
                order += 1
    
                sentences = []
                # logger.info("SENT: " + first_two)
                num_sentences += 1
        
    except asyncio.CancelledError:
        logger.warning("Cancelled")
        pass
    if user_id in interrupt_flags and interrupt_flags[user_id]:
        interrupt_flags[user_id] = False
    

    if len(sentences) > 0: #if a sentence was received, send it 
        await websocket.send_text(str(order) + '-' + ' '.join(sentences))
        order += 1

        logger.info("rest of sentences: " + ' '.join(sentences))
    if len(content) > 0:

        if "{__SOS__}" in content and not sent_response[user_id]: #NEOPHYTOS
            sos_text = "Es tut mir wirklich leid zu hören, dass es Ihnen nicht gut geht. Wenn man mit negativen Emotionen wie diesen umgeht, ist es wichtig, Mitgefühl mit sich selbst zu haben und sich selbst gegenüber innerlich eine wohlwollende, liebevolle Haltung einzunehmen. Würden Sie sich wohl fühlen, wenn ich Sie durch einige Selbstreflexions-Fragen leite, um dies zu üben, oder würden Sie stattdessen lieber über Ihre Gefühle sprechen? Sie können auch jederzeit entscheiden, diese Sitzung ein anderes Mal fortzusetzen. Es ist wichtig, Ihren Instinkten zu folgen und nur das zu tun, womit Sie sich wohl fühlen."
            #await websocket.send_text(sos_text)
            await split_text_into_chunks_and_stream(websocket, sos_text)
            full_response = "{__SOS__}" + sos_text
        elif "{__SOS__}" in content and sent_response[user_id]:
            #response has already been sent, so we don't stream SOS text but just append it to storage. will be handled by the check after response from assistant has been streamed
            content = ""
            full_response = "{__SOS__} " + full_response

        elif "__ALL_EXERCISES_COMPLETED__" in content:
            transition_to_feedback_msg = "Vielen Dank um all Ihre Bemühungen heute. Wenn es Ihnen nichts ausmacht, möchte ich Sie bitten, ich eine Minute Zeit zu nehmen, um Ihre Erfahrungen mit den Übungen zu bewerten, damit ich zukünftige Sitzungen besser auf Ihre Vorlieben abstimmen kann."
                    # Inform the user about transitioning to feedback collection
            # Inform the user about transitioning to feedback collection
            await websocket.send_text(transition_to_feedback_msg)
            full_response = "__ALL_EXERCISES_COMPLETED__" + transition_to_feedback_msg

        elif "{FINISHED_SOS}" in content:
            getting_back_to_schedule_msg = "Vielen Dank, dass Sie sich die Zeit genommen haben, sich selbst Mitgefühl zu schenken. Wie fühlen Sie sich jetzt? Ich möchte nur kurz Ihre Optionen klären, damit wir so vorgehen können, wie es Ihnen am angenehmsten ist: Wenn Sie möchten, könnten wir zum SAT-Protokoll zurückkehren und einige der nächsten Übungen im Plan ansehen. Wenn Sie jedoch lieber über Ihre Gefühle sprechen möchten oder es vorziehen würden, die Sitzung hier zu beenden, bin ich hier, um Sie zu unterstützen. Wie möchten Sie fortfahren?"
            #await websocket.send_text(getting_back_to_schedule_msg)
            await split_text_into_chunks_and_stream(websocket, getting_back_to_schedule_msg)
            full_response= "{FINISHED_SOS}" + getting_back_to_schedule_msg
        else:
            await websocket.send_text(str(order) + '-' + content)
        order += 1
        logger.info("rest of content: " + content) 

    #if query_results is not None and query_results != {}:
        #print("Sending query results to the user")
        #qrs = json.dumps(query_results)
        #await websocket.send_text(f'{order-1}-END {qrs}')
    #else:
        #await websocket.send_text(f'{order-1}-END')

    #perform_translation(full_response, messages_en)

    assistant_message = {"role": "assistant", "content": full_response}
    asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message)) #each response from the chatbot is saved to firestore

    messages[user_id].append(assistant_message)

    total_tokens_used_today = bot_args.get("total_tokens_used_today", {})
    total_tokens_used_today[user_id] += num_tokens_from_string(full_response, "cl100k_base")

    
    print("TOTAL TOKENS AFTER RESPONSE: ", total_tokens_used_today[user_id])
    # Send token usage update to the client
    await websocket.send_json({"tokens_used": total_tokens_used_today[user_id]})
    

    
    print("assistant:", full_response)
    await response.aclose()

    if total_tokens_used_today[user_id] > 100000:
        await websocket.send_text("Leider wurde das tägliche Token-Limit für GPT überschritten. Bitte warten Sie bis morgen, um den Dienst weiter zu nutzen. Vielen Dank und passen Sie gut auf sich auf!")
        raise ExceedDailyTokenError("Daily token limit exceeded")
    return assistant_message


#FUNCTION PROBABLY NOT NEEDED
async def execute_tool_queries(json_obj, user_enabled_tools):
    '''
        Calls the functions in the tool with parameters.
    '''
    if json_obj == {}:
        return {}
    
    args = json_obj["args"]
    if "req_info" in args:
        args.pop("req_info")
    try:
        args["req_info"] = {
            "Latitude": float(json_obj["location"]["latitude"]),
            "Longitude": float(json_obj["location"]["longitude"]),
        }
    except:
        pass
    
    print("GPT output a function called this:", json_obj["name"])
    start_time = time.time()
    query_results = {}
    query_results[json_obj["name"]] = user_enabled_tools[json_obj["name"]].run(args) #function that runs the tool, e.g file upload, emotion analysis?, [SATTool, FileUpload] --> json_obj["name"]=SATTool
    print(f"The tool takes {time.time() - start_time:.3f} sec")
    return query_results
    


def update_model(user_id):
    '''Update user specific Bots with user settings (model type: GPT4/3.5, ai descriptions, sarcasm)'''

    if user_id not in user_bots_dict: #user_bots_dict {'guest_s': [<assistants.template.TemplateAssistant object at 0x00000263F86C6C50>]} This basically means it uses MainAssistant to craft responses
        return
    userAssistants = user_bots_dict[user_id]

    def set_user_settings_dict(user_id: str, settings: dict):
        set_user_settings(db, user_id, settings)
        if user_id not in messages:
            messages[user_id] = Messages(db, user_id)
        messages[user_id].update_settings(settings)
        return {"detail": "Settings submitted successfully"}
    
    settings = messages[user_id].get_settings() #settings = {"assistantsName" :"Satherine",  "gender" : "Female", "relationship" : "Therapist", "gptMode": "Smart",  "aiDescription": [ "You are a supportive and empathetic virtual assistant, specialized in guiding users through the Self Attachment Therapy (SAT) protocol. This therapy consists of 20 structured protocols spread over 8 weeks, focusing on emotional and therapeutic progression.",  "Your primary role is to act as a coach, assisting users in understanding and practicing these protocols. Each week in the SAT protocol is outlined in a structured JSON file and includes exercises, a descriptive objective, and a recap.", "At the start of each session, ask the user for their username to fetch their progress. Begin by inquiring about their current emotional state. Recognize and validate their feelings with empathy. Based on their emotional response, decide whether to continue discussing their feelings or proceed with the SAT protocol. Prioritize empathetic engagement until the user is ready to move forward.",  "For first-time users, start with an introduction to SAT, then proceed to the scheduled exercises. For returning users, recap the previous week's progress before introducing the current week's exercises.",  "Sessions should last approximately 15 minutes and be conducted twice daily. Provide clear, step-by-step instructions for each exercise, encouraging users to reflect on their experiences and articulate their feelings.", "Your communication should be empathetic, supportive, and focused on the user. Adapt the therapy based on user feedback and emotional states. Always remember that your goal is to create a nurturing and understanding environment, facilitating the user's journey through the SAT protocol with care and empathy.", "Note: You are not required to generate images. Your primary function is to interact with the user in a therapeutic and supportive manner." ],"aiSarcasm": 0.0,}
    if settings["gptMode"] in GPT_MODELS:
        model_name = GPT_MODELS[settings["gptMode"]]["model_name"]
    else:
        model_name = "gpt-4-0613"
        #model_name = "gpt-4-turbo"
    print(f"{model_name} is selected for {user_id}")
    if model_name != userAssistants[0].assistant_settings["model"]:
        current_user_settings = get_user_settings_route(user_id)
        kwargs = {"user_id": user_id,
                "current_user_settings": current_user_settings, 
                "update_user_settings_fn": set_user_settings_dict}
        ALL_HANLDERS["OnModelChanged"](**kwargs) #only used by image creation tool
    
    for assistant in userAssistants:
        assistant.assistant_settings["name"] = settings["assistantsName"] if "assistantsName" in settings else "Elivia"
        assistant.assistant_settings["model"] = model_name
        assistant.assistant_settings["sarcasm"] = settings["aiSarcasm"] if "aiSarcasm" in settings else 0.5
        assistant.assistant_descriptions = settings["aiDescription"] if "aiDescription" in settings else []
        assistant._construct_initial_prompt()

def update_user_info(user_id, user_info):
    '''Get user settings that's set from the frontend, and update user name and interests for Bot usages'''
    settings = messages[user_id].get_settings()
    if "yourName" in settings and settings["yourName"] is not None:
        user_info["name"] = settings["yourName"]

    if "yourInterests" in settings and settings["yourInterests"] is not None:
        user_info["interests"] = settings["yourInterests"]

    return user_info


#################### FUNCTIONS RELATED TO SESSION MANAGEMENT #################

async def get_user_session(user_id:str):
    session_ref = db.collection('user_sessions').document(user_id)
    doc = session_ref.get()
    if doc.exists:
        return doc.to_dict().get('session_nr', 0)
    else:
        return 0
  

async def increment_user_session(user_id: str):
    # Reference to the user's session document
    session_ref = db.collection('user_sessions').document(user_id)
    
    # Try to get the document
    doc = session_ref.get()
    if doc.exists:
        #currently we just increment the session nr when the screen refreshes
        current_session = doc.to_dict().get('session_nr', 0) + 1
        # Update the session number in Firestore
        session_ref.update({'session_nr': current_session})
    else:
        current_session = 1  # Start with session 1 if it's a new user
        # Update the session number in Firestore
        session_ref.set({'session_nr': current_session})

    return current_session


async def increment_user_session_atomic(user_id: str):
    session_ref = db.collection('user_sessions').document(user_id)
    doc = session_ref.get()
    if doc.exists:
        data = doc.to_dict()
        current_session = data.get('session_nr', 0)
        
        # Get exercises for last session
        exercises_for_last_session = exercise_schedule[current_session]
        
        # Check if all exercises were completed
        all_exercises_completed = all([exercise_tracking[exercise]['completed'] for exercise in exercises_for_last_session])

        if all_exercises_completed:
            current_session += 1
            session_ref.update({'session_nr': current_session})
        return current_session
    else:
        # Create a new session for a new user
        session_ref.set({'session_nr': 1})
        return 1


async def get_current_session(user_id: str):
    session_ref = db.collection('user_sessions').document(user_id)
    doc = session_ref.get()
    if doc.exists:
        return doc.to_dict().get('session_nr', 0)
    return 0  # Return 0 if no session document exists

#################### FUNCTIONS RELATED TO SESSION MANAGEMENT END #################

#################### FUNCTIONS RELATED TO FEEDBACK COLLECTION #################


feedback_responses = {
    "Exercise 1": {},
    # Additional exercises
}



def store_session_feedback(user_id, session_nr, feedback_responses):
    # Reference to the user's document in the user_sessions collection
    user_sessions_ref = db.collection("user_sessions").document(user_id)
    
    # Attempt to get the document
    user_sessions_doc = user_sessions_ref.get()

    #to be a key, session nr needs to be a string
    session_nr_str = str(session_nr)
    
    if user_sessions_doc.exists:
        # Document exists, retrieve its data
        sessions_data = user_sessions_doc.to_dict()
        
        # Check if there's existing data for the current session_nr
        if session_nr_str in sessions_data:
            # Update the existing session data with new feedback
            sessions_data[session_nr_str]["feedback"] = feedback_responses
        else:
            # If this session_nr doesn't exist, add it with the feedback
            sessions_data[session_nr_str] = {"feedback": feedback_responses}
        
        # Update the document with the new sessions data
        user_sessions_ref.update(sessions_data)
    else:
        # If the document doesn't exist, create it with the current session and feedback
        new_sessions_data = {
            session_nr_str: {"feedback": feedback_responses}
        }
        user_sessions_ref.set(new_sessions_data)
    
    print(f"Stored feedback for user {user_id}, session {session_nr_str}.")

def handle_message_storing(user_id, role, msg, messages):
    message = {"role": role, "content": msg}
    asyncio.create_task(messages[user_id].save_message_to_firestore(message)) #each response from the chatbot is saved to firestore

    messages[user_id].append(message)
    print('added message to messages: ', messages[user_id])


#################### FUNCTIONS RELATED TO FEEDBACK COLLECTION END #################
    
##################### FUNCTIONS RELATED TO EXERCISE RECOMMENDATIONS #################################
exercise_schedule = {
    1: ["Exercise 1", "Exercise 2.1", "Exercise 2.2"],
    2: ["Exercise 3", "Exercise 4"],
    3: ["Exercise 5"],
    4: ["Exercise 6", "Exercise 7.1", "Exercise 7.2", "Exercise 8"],
    5: ["Exercise 9"],
    6: ["Exercise 10"],
    7: ["Exercise 11", "Exercise 12", "Exercise 13", "Exercise 14"],
    8: ["Exercise 15.1", "Exercise 15.2"],
    9: ["Exercise 15.1", "Exercise 15.2", "Exercise 15.3"],
    10: ["Exercise 15.3", "Exercise 15.4"],
    11: ["Exercise 15.4", "Exercise 16"],
    12: ["Exercise 15.4", "Exercise 16"],
    13: ["Exercise 17", "Exercise 18", "Exercise 19"],
    14: ["Exercise 20", "Exercise 21"],
}

session_objectives = {
    1: "Mitfühlende Verbindung mit unserem Kindheits-Ich aufbauen",
    2: "Eine leidenschaftliche, liebevolle Beziehung zum Kindheits-Ich aufbauen",
    3: "Eine leidenschaftliche, liebevolle Beziehung zum Kindheits-Ich aufbauen",
    4: "Freudige Liebe für unser Kindheits-Ich",
    5: "Verarbeitung schmerzhafter Kindheitserlebnisse",
    6: "Verarbeitung schmerzhafter Kindheitserlebnisse",
    7: "Wiedererlernen des Lachens",
    8: "Veränderung unserer Perspektive",
    9: "Veränderung unserer Perspektive",
    10: "Veränderung unserer Perspektive",
    11: "Veränderung unserer Perspektive",
    12: "Veränderung unserer Perspektive",
    13: "Sozialisierung Ihres Kindheits-Ichs",
    14: "Ihr Kindheits-Ich: Quelle Ihrer Kreativität",
}



session_theory = {
    1: "Versuchen Sie, zwischen Ihrem Erwachsenen-Ich und Ihrem Kindheits-Ich zu unterscheiden. Das Betrachten von 'glücklichen' und 'unglücklichen' Kinderfotos hilft bei diesem Bemühen. Denken Sie über Ihre frühen Jahre und emotionale Probleme mit Eltern und anderen bedeutenden Personen in der frühen Kindheit nach. Das erste Prinzip des SAT ist es, eine warme und mitfühlende Haltung gegenüber unserem Kindheits-Ich zu haben, unabhängig davon, ob die zugrunde liegenden Emotionen positiv oder negativ sind. Später wird dieses Mitgefühl auf andere Personen ausgedehnt.",

    2: "In dieser Sitzung ist unser Ziel, eine tiefe, liebevolle Bindung zu Ihrem inneren Kind zu pflegen. Es geht darum, eine Verbindung zu etablieren, die die fürsorgliche Liebe widerspiegelt, die ein Elternteil für sein Kind empfindet. Dies beinhaltet das Umarmen der Selbstliebe und das Ausdrücken von Zuneigung durch Gesten wie Blickkontakt, Händehalten, sanfte Berührungen, Singen, Spielen und Lachen. Dabei zielen wir darauf ab, die Freisetzung von Dopamin in Ihrem Gehirn auszulösen, ähnlich den Gefühlen, die in mütterlicher, romantischer oder spiritueller Liebe erfahren werden. Dies fördert nicht nur Hoffnung und Motivation, sondern bereitet Sie auch darauf vor, eine aktive Rolle in der Fürsorge für Ihr inneres Kind zu übernehmen. Ein Teil dieses Prozesses beinhaltet das laut Aussprechen zu Ihrem inneren Kind, was die emotionale Bindung und Reife verstärkt. Diese Praxis, die im Erwachsenenalter oft übersehen wird, ist entscheidend für die emotionale und kognitive Entwicklung. Sie repräsentiert emotionale und kognitive Reife, kein Abweichen davon. Denken Sie daran, die Fürsorge für ein Kind ist eine grundlegende menschliche Fähigkeit, und diese Übungen sind darauf ausgelegt, diese angeborene Fähigkeit in Ihnen zu wecken und zu stärken.",

    3: "Wir werden unser Ziel fortsetzen, eine tiefe, liebevolle Bindung zu Ihrem inneren Kind zu pflegen. Es geht darum, eine Verbindung zu etablieren, die die fürsorgliche Liebe widerspiegelt, die ein Elternteil für sein Kind empfindet. Dies beinhaltet das Umarmen der Selbstliebe und das Ausdrücken von Zuneigung durch Gesten wie Blickkontakt, Händehalten, sanfte Berührungen, Singen, Spielen und Lachen. Dabei zielen wir darauf ab, die Freisetzung von Dopamin in Ihrem Gehirn auszulösen, ähnlich den Gefühlen, die in mütterlicher, romantischer oder spiritueller Liebe erfahren werden. Dies fördert nicht nur Hoffnung und Motivation, sondern bereitet Sie auch darauf vor, eine aktive Rolle in der Fürsorge für Ihr inneres Kind zu übernehmen. Ein Teil dieses Prozesses beinhaltet das laut Aussprechen zu Ihrem inneren Kind, was die emotionale Bindung und Reife verstärkt. Diese Praxis, die im Erwachsenenalter oft übersehen wird, ist entscheidend für die emotionale und kognitive Entwicklung. Sie repräsentiert emotionale und kognitive Reife, kein Abweichen davon. Denken Sie daran, die Fürsorge für ein Kind ist eine grundlegende menschliche Fähigkeit, und diese Übungen sind darauf ausgelegt, diese angeborene Fähigkeit in Ihnen zu wecken und zu stärken.",

    4: "In dieser Sitzung konzentrieren wir uns auf die künstlerische Neuschöpfung unserer emotionalen Welt, ein wichtiger Aspekt unserer Reise zur Selbstheilung und zum Verständnis. \n Ein wesentliches Element dieses Prozesses ist das Konzept des Hausbaus. Historisch gesehen war der Hausbau eine grundlegende menschliche Tätigkeit, die nicht nur physischen Schutz bot, sondern auch ein Gefühl von Sicherheit und Geborgenheit vermittelte. Denken Sie zurück an die Kindheit, wo der Bau eines Hauses oder einer Festung eines der frühesten und vergnüglichsten Spiele war. Diese Aktivität gab uns ein Gefühl von Komfort und Sicherheit, einen sicheren Zufluchtsort, den wir selbst geschaffen haben. Nun, da wir diese Therapie beginnen, denken Sie an den Bau eines Traumhauses als Metapher für den Aufbau eines neuen Selbst. Es geht darum, einen Raum zu schaffen, der sich sicher, geschützt und einzigartig anfühlt, in dem unser emotionales Selbst gedeihen kann. \n Zusätzlich werden wir unsere Bindung zur Natur erkunden. Unsere frühen Verbindungen zur Natur spielen eine entscheidende Rolle für unsere körperliche und geistige Gesundheit und prägen unsere Einstellung zur Umwelt, während wir wachsen. Die erneute Verbindung mit der Natur kann jetzt diese frühen Gefühle der Zugehörigkeit wieder entfachen und ein Gefühl von Frieden und Ausgeglichenheit in unser Leben bringen.",

    5: "In dieser Sitzung machen wir einen bedeutenden Schritt: Ihr Erwachsenen-Ich wird aktiv daran arbeiten, Ihr Kindheits-Ich zu trösten und zu unterstützen, insbesondere im Umgang mit aktuellen oder vergangenen Schmerzen. \n Wir haben Fähigkeiten entwickelt, um positive Emotionen zu verstärken, und nun verlagern wir unseren Fokus darauf, negative Emotionen wie Wut, Angst, Traurigkeit und Angst zu verarbeiten und zu reduzieren. Diese Emotionen können aus verschiedenen Aspekten Ihres Lebens entstehen, einschließlich Beziehungen zu Partnern, Familie, Freunden, Arbeit oder gesellschaftlichen Interaktionen. \n Der Schlüssel hierbei ist, diese negativen Emotionen auf Ihr Kindheits-Ich zu projizieren. Dies ermöglicht Ihrem Erwachsenen-Ich, einzuschreiten, diese Schmerzen anzusprechen und zu lindern, und bietet die gleiche Unterstützung und Fürsorge, die ein fürsorglicher Elternteil bieten würde. Indem Sie sich in diesen Prozess einbringen, erkennen Sie nicht nur die Emotionen an, sondern arbeiten auch aktiv daran, ihre Intensität zu verringern. \n Ein wesentlicher Teil der Übungen dieser Woche beinhaltet Selbstberuhigung und Selbstmassage, die mächtige Werkzeuge zur emotionalen Regulierung sind. Diese Praktiken helfen, Oxytocin und Vasopressin freizusetzen, Hormone, die eine entscheidende Rolle bei der Milderung von Gefühlen des Unbehagens und des Unwohlseins spielen. \n Darüber hinaus beginnen wir mit der Verarbeitung schmerzhafter Kindheitserlebnisse. Kindheitstraumata führen oft zu anhaltenden emotionalen und verhaltensbedingten Herausforderungen. Indem Sie mit weniger schweren Fällen beginnen, können Sie sich allmählich an herausforderndere Traumata heranarbeiten. Es ist entscheidend, nur zu schwereren Traumata überzugehen, wenn Sie sich sicher und bereit fühlen. Wenn Sie unsicher sind, ist es völlig in Ordnung, mit den Übungen der vergangenen Wochen fortzufahren und sich auf weniger traumatische Erlebnisse zu konzentrieren, bis Sie sich stärker fühlen.",

    6: "Wir setzen unsere Reise fort, Ihrem Erwachsenen-Ich beizubringen, wie es aktiv daran arbeiten kann, Ihr Kindheits-Ich zu trösten und zu unterstützen, insbesondere im Umgang mit aktuellen oder vergangenen Schmerzen. \n Wir haben Fähigkeiten entwickelt, um positive Emotionen zu verstärken, und nun verlagern wir unseren Fokus darauf, negative Emotionen wie Wut, Angst, Traurigkeit und Angst zu verarbeiten und zu reduzieren. Diese Emotionen können aus verschiedenen Aspekten Ihres Lebens entstehen, einschließlich Beziehungen zu Partnern, Familie, Freunden, Arbeit oder gesellschaftlichen Interaktionen. \n Der Schlüssel hierbei ist, diese negativen Emotionen auf Ihr Kindheits-Ich zu projizieren. Dies ermöglicht Ihrem Erwachsenen-Ich, einzuschreiten, diese Schmerzen anzusprechen und zu lindern, und bietet die gleiche Unterstützung und Fürsorge, die ein fürsorglicher Elternteil bieten würde. Indem Sie sich in diesen Prozess einbringen, erkennen Sie nicht nur die Emotionen an, sondern arbeiten auch aktiv daran, ihre Intensität zu verringern. \n Ein wesentlicher Teil der Übungen dieser Woche beinhaltet Selbstberuhigung und Selbstmassage, die mächtige Werkzeuge zur emotionalen Regulierung sind. Diese Praktiken helfen, Oxytocin und Vasopressin freizusetzen, Hormone, die eine entscheidende Rolle bei der Milderung von Gefühlen des Unbehagens und des Unwohlseins spielen. \n Darüber hinaus beginnen wir mit der Verarbeitung schmerzhafter Kindheitserlebnisse. Kindheitstraumata führen oft zu anhaltenden emotionalen und verhaltensbedingten Herausforderungen. Indem Sie mit weniger schweren Fällen beginnen, können Sie sich allmählich an herausforderndere Traumata heranarbeiten. Es ist entscheidend, nur zu schwereren Traumata überzugehen, wenn Sie sich sicher und bereit fühlen. Wenn Sie unsicher sind, ist es völlig in Ordnung, mit den Übungen der vergangenen Wochen fortzufahren und sich auf weniger traumatische Erlebnisse zu konzentrieren, bis Sie sich stärker fühlen.",

    7: "In dieser Sitzung konzentrieren wir uns auf die Bedeutung des Lachens bei der emotionalen Heilung. Lachen, beeinflusst durch familiäre und kulturelle Normen, ist eine natürliche Reaktion, die sicher gebundene Kinder oft zeigen. Obwohl negative Reaktionen auf Enttäuschungen üblich sind, ist es vorteilhaft, Situationen in einem anderen Licht zu sehen und Humor in ihnen zu finden. Wir werden Ihr inneres Kind zum Lachen ermutigen, eine spielerische Einstellung fördern und eine positive Perspektive auf Herausforderungen unterstützen. Theorien des Lachens umfassen die Überlegenheitstheorie, Inkongruenztheorie, Entlastungstheorie, Perspektivtheorie und Evolutionstheorie. Lachen ist ein mächtiges Werkzeug zur Selbstregulierung von Emotionen, das Angst und Depression durch das Auslösen von Dopamin und Serotonin bekämpft. Es ist entscheidend, eine nicht-feindselige Haltung im Lachen zu bewahren. Diese Woche konzentrieren wir uns darauf, das Lachen neu zu lernen, negative Muster zu lockern, die die emotionale Entwicklung und die Fähigkeit, frei zu lachen, einschränken, und unterstützen Ihre Reise zur emotionalen Freiheit.",

    8: "In dieser Sitzung konzentrieren wir uns darauf, die Fähigkeit zu entwickeln, unsere Verstimmungen und Rückschläge wegzulachen und unsere Sicht auf negative Emotionen zu transformieren. Wir führen das Konzept der Gestaltvase ein, ein mächtiges Werkzeug zur Perspektivenänderung. Stellen Sie sich eine Vase vor, die, wenn man sie anders betrachtet, zwei Gesichter offenbart. Dies symbolisiert die Verlagerung unserer Aufmerksamkeit von negativen Emotionen (die schwarze Vase) zu einer positiveren Perspektive (die weißen Gesichter). Es geht darum, über die unmittelbare emotionale Reaktion hinaus zu einer breiteren, optimistischeren Sichtweise zu sehen. Wir erkunden drei Kontexte für mitfühlenden Humor, ausgehend von der Analogie der Gestaltvase: 1. Weltinkongruenzen: Erkennen der Diskrepanzen zwischen dem, was gesagt wird, und dem, was in der Welt getan wird, insbesondere in sozioökonomischen und politischen Systemen. Wir konzentrieren uns darauf, Humor in diesen Inkongruenzen zu finden, was positive Emotionen und kreatives Denken fördert. 2. Selbst-/Weltinkongruenzen: Umgang mit Verletzungen unserer Erwartungen vom Leben oder anderen. Wir lernen, schnell Humor in der Diskrepanz zwischen unseren Erwartungen und der Realität zu finden, was negative Emotionen reduziert und Kreativität fördert. 3. Inkongruentes Selbst: Finden von Humor in den Inkonsistenzen und Widersprüchen unseres eigenen Lebens. Dies hilft dabei, spielerisch mit unseren Emotionen und Gedanken umzugehen und fördert emotionale Flexibilität und Weisheit. Wir kehren auch zu Chaplins Einsicht zurück: 'Das Leben ist eine Tragödie aus der Nähe betrachtet, aber eine Komödie aus der Distanz.' Über unsere vergangenen Verstimmungen zu lachen, ihre Rolle in unserem Wachstum zu erkennen, kann eine ermächtigende Erfahrung sein. Die Übungen dieser Woche werden Sie ermutigen, Ihre vergangenen Herausforderungen als Gelegenheiten für Lachen und Wachstum zu sehen und sie als Sprungbretter zu einem widerstandsfähigeren und freudigeren Selbst zu nutzen. Denken Sie daran, die Fähigkeit, über Verstimmungen zu lachen, ist eine Form der emotionalen Stärke und Anpassungsfähigkeit.",

    9: "Wir konzentrieren uns weiterhin darauf, die Fähigkeit zu entwickeln, unsere Verstimmungen und Rückschläge wegzulachen und unsere Sicht auf negative Emotionen zu transformieren. Wir führen das Konzept der Gestaltvase ein, ein mächtiges Werkzeug zur Perspektivenänderung. Stellen Sie sich eine Vase vor, die, wenn man sie anders betrachtet, zwei Gesichter offenbart. Dies symbolisiert die Verlagerung unserer Aufmerksamkeit von negativen Emotionen (die schwarze Vase) zu einer positiveren Perspektive (die weißen Gesichter). Es geht darum, über die unmittelbare emotionale Reaktion hinaus zu einer breiteren, optimistischeren Sichtweise zu sehen. Wir erkunden drei Kontexte für mitfühlenden Humor, ausgehend von der Analogie der Gestaltvase: 1. Weltinkongruenzen: Erkennen der Diskrepanzen zwischen dem, was gesagt wird, und dem, was in der Welt getan wird, insbesondere in sozioökonomischen und politischen Systemen. Wir konzentrieren uns darauf, Humor in diesen Inkongruenzen zu finden, was positive Emotionen und kreatives Denken fördert. 2. Selbst-/Weltinkongruenzen: Umgang mit Verletzungen unserer Erwartungen vom Leben oder anderen. Wir lernen, schnell Humor in der Diskrepanz zwischen unseren Erwartungen und der Realität zu finden, was negative Emotionen reduziert und Kreativität fördert. 3. Inkongruentes Selbst: Finden von Humor in den Inkonsistenzen und Widersprüchen unseres eigenen Lebens. Dies hilft dabei, spielerisch mit unseren Emotionen und Gedanken umzugehen und fördert emotionale Flexibilität und Weisheit. Wir kehren auch zu Chaplins Einsicht zurück: 'Das Leben ist eine Tragödie aus der Nähe betrachtet, aber eine Komödie aus der Distanz.' Über unsere vergangenen Verstimmungen zu lachen, ihre Rolle in unserem Wachstum zu erkennen, kann eine ermächtigende Erfahrung sein. Die Übungen dieser Woche werden Sie ermutigen, Ihre vergangenen Herausforderungen als Gelegenheiten für Lachen und Wachstum zu sehen und sie als Sprungbretter zu einem widerstandsfähigeren und freudigeren Selbst zu nutzen. Denken Sie daran, die Fähigkeit, über Verstimmungen zu lachen, ist eine Form der emotionalen Stärke und Anpassungsfähigkeit.",

    10: "Wir setzen unsere Reise fort, Humor zu nutzen, um unsere Sicht auf die Herausforderungen des Lebens neu zu gestalten. Aufbauend auf dem Konzept der Gestaltvase zielen wir darauf ab, unseren Fokus von negativen Emotionen zu einer positiveren, humorvolleren Sichtweise zu verschieben. Diese Sitzung greift die Kraft auf, über unmittelbare Rückschläge hinauszusehen, und fördert eine Perspektive, in der die Inkongruenzen des Lebens und unsere eigenen Widersprüche zu Quellen des Lachens und Wachstums werden. Während wir uns erneut in die Kontexte des mitfühlenden Humors vertiefen, erinnern wir uns an Chaplins Einsicht: Die Prüfungen des Lebens können aus der Distanz betrachtet zu einer Komödie werden. In dieser Sitzung verstärken wir unsere Fähigkeit, Humor im Unerwarteten zu finden, indem wir vergangene Herausforderungen als Gelegenheiten nutzen, um Widerstandsfähigkeit und ein freudigeres Selbst zu kultivieren. Über unsere Verstimmungen zu lachen, bedeutet nicht nur, den Moment aufzuhellen; es geht darum, emotionale Stärke und Anpassungsfähigkeit für die bevorstehende Reise aufzubauen.",

    11: "Wir setzen unsere Reise fort, Humor zu nutzen, um unsere Sicht auf die Herausforderungen des Lebens neu zu gestalten. Aufbauend auf dem Konzept der Gestaltvase zielen wir darauf ab, unseren Fokus von negativen Emotionen zu einer positiveren, humorvolleren Sichtweise zu verschieben. Diese Sitzung greift die Kraft auf, über unmittelbare Rückschläge hinauszusehen, und fördert eine Perspektive, in der die Inkongruenzen des Lebens und unsere eigenen Widersprüche zu Quellen des Lachens und Wachstums werden. Während wir uns erneut in die Kontexte des mitfühlenden Humors vertiefen, erinnern wir uns an Chaplins Einsicht: Die Prüfungen des Lebens können aus der Distanz betrachtet zu einer Komödie werden. In dieser Sitzung verstärken wir unsere Fähigkeit, Humor im Unerwarteten zu finden, indem wir vergangene Herausforderungen als Gelegenheiten nutzen, um Widerstandsfähigkeit und ein freudigeres Selbst zu kultivieren. Über unsere Verstimmungen zu lachen, bedeutet nicht nur, den Moment aufzuhellen; es geht darum, emotionale Stärke und Anpassungsfähigkeit für die bevorstehende Reise aufzubauen.",

    12: "Wir setzen unsere Reise fort, Humor zu nutzen, um unsere Sicht auf die Herausforderungen des Lebens neu zu gestalten. Aufbauend auf dem Konzept der Gestaltvase zielen wir darauf ab, unseren Fokus von negativen Emotionen zu einer positiveren, humorvolleren Sichtweise zu verschieben. Diese Sitzung greift die Kraft auf, über unmittelbare Rückschläge hinauszusehen, und fördert eine Perspektive, in der die Inkongruenzen des Lebens und unsere eigenen Widersprüche zu Quellen des Lachens und Wachstums werden. Während wir uns erneut in die Kontexte des mitfühlenden Humors vertiefen, erinnern wir uns an Chaplins Einsicht: Die Prüfungen des Lebens können aus der Distanz betrachtet zu einer Komödie werden. In dieser Sitzung verstärken wir unsere Fähigkeit, Humor im Unerwarteten zu finden, indem wir vergangene Herausforderungen als Gelegenheiten nutzen, um Widerstandsfähigkeit und ein freudigeres Selbst zu kultivieren. Über unsere Verstimmungen zu lachen, bedeutet nicht nur, den Moment aufzuhellen; es geht darum, emotionale Stärke und Anpassungsfähigkeit für die bevorstehende Reise aufzubauen.",

    13: "In dieser Sitzung konzentrieren wir uns darauf, Ihr Kindheits-Ich von antisozialen Verhaltensweisen wegzuführen und darauf, eine mitfühlende Figur als ideales Vorbild zu identifizieren. Wir erkennen, dass in den letzten Jahrzehnten ein Anstieg des Narzissmus in westlichen Gesellschaften durch verschiedene wirtschaftliche, soziale und kulturelle Faktoren begünstigt wurde. Dazu gehören die hochgradig wettbewerbsorientierte Natur von Unternehmensökonomien und Bildungssystemen, die Verherrlichung von Individualität und persönlichem Erfolg durch die Medien und der bedeutende Einfluss des Internets und der sozialen Medien, die Individualismus fördern. Solche Faktoren haben narzisstische Züge normalisiert, die sich nachteilig auf die psychische Gesundheit auswirken können. Als Reaktion darauf muss unser Erwachsenen-Ich eine Schlüsselrolle bei der Sozialisierung unseres Kindheits-Ichs spielen. Dies beinhaltet das Bewusstwerden jeglicher narzisstischer Tendenzen in unserem Kindheits-Ich, wie Neid, Eifersucht, Gier oder Misstrauen, und das Verständnis, wie das Ausagieren dieser negativen Emotionen unseren Zielen entgegenwirken kann. Um diesen Tendenzen entgegenzuwirken, ist es das Ziel, die entgegengesetzten Gefühle zu kultivieren. Dies hilft nicht nur beim Umgang mit diesen negativen Emotionen, sondern eröffnet auch Wege für kreative Lösungen. SAT betont die Pflege des entgegengesetzten Pols dieser Merkmale, um eine ausgeglichene und ganze Persönlichkeit zu schaffen. Das Motto hier ist, dass die Beherrschung der Psychopathologie die Entwicklung der anderen Pole dieser Merkmale erfordert. Indem wir dies tun, streben wir nach einem ausgeglicheneren, mitfühlenderen und empathischeren Selbst. Die Übungen dieser Woche sind darauf ausgelegt, Ihnen zu helfen, diese gegensätzlichen Eigenschaften zu identifizieren und zu fördern und Sie so zu einer emotionalen Balance und kreativem Denken zu führen.",

    14: "In unserer letzten Sitzung der Self-Attachment Therapie werden wir uns mit der Beziehung zwischen Ihrem inneren Kindheits-Ich und Ihrer Kreativität beschäftigen, einem wesentlichen Aspekt der Entdeckung Ihres wahren Selbst. Kinder, die früh im Leben sichere Bindungen entwickeln, haben oft eine starke Fähigkeit zum unabhängigen Denken und Reflektieren. Dies fördert ihre Fähigkeit, ihr wahres Selbst zu entdecken, welches die eigentliche Quelle ihrer Kreativität, Unabhängigkeit und Spontanität ist. Im Gegensatz dazu kann ein Kind, das stark von seiner Umgebung beeinflusst wird, ein sogenanntes 'falsches Selbst' entwickeln, eingeschränkt durch äußere Zwänge und ohne echte Spontaneität. Etwa im Alter von drei oder vier Jahren beginnen Kinder, bemerkenswerte imaginative Fähigkeiten und die Fähigkeit, Verbindungen zwischen verschiedenen Phänomenen herzustellen – die Wurzeln der Kreativität. Die natürliche Spontanität eines Kindes hilft ihm, Ereignisse ohne starre Vorannahmen zu verstehen. Externe Drücke können jedoch manchmal diese Spontaneität durch Starre und Uniformität ersetzen. Unser Ziel in der Self-Attachment Therapy ist es, diese sichere Bindung zwischen dem Kindheits-Ich und dem Erwachsenen neu zu entfachen und Ihren kreativen Funken wiederzubeleben. Darüber hinaus zeigen kreative Personen oft paradoxe Merkmale – sie sind energisch, doch ruhig, klug, doch naiv, diszipliniert, doch spielerisch und so weiter. Diese Paradoxien zu umarmen, ist der Schlüssel zur Förderung Ihrer Kreativität. Diese Woche konzentrieren wir uns auch auf die Bedeutung von Vorbildern, Parabeln und inspirierenden Zitaten für das emotionale Wachstum. Bestärkungen von Ihrem gewählten mitfühlenden Vorbild können Ihre Willenskraft und Ausdauer stärken. Sie ermutigen Sie, Herausforderungen und Rückschläge als Teil Ihrer Reise zur Erreichung Ihrer Ziele zu akzeptieren. Denken Sie an die Worte von Nietzsche: 'Was mich nicht umbringt, macht mich stärker.' Diese Perspektive ermutigt dazu, die Herausforderungen des Lebens mit Liebe und Stärke anzunehmen und sie als Chancen für Wachstum zu sehen. Während wir unsere Therapie abschließen, ist es wichtig zu erkennen, dass Kreativität nicht nur künstlerischer Ausdruck ist; es geht darum, wie Sie das Leben angehen, Probleme lösen und die Welt um sich herum wahrnehmen. Ihr inneres Kindheits-Ich ist eine Quelle der Kreativität, und diesen Aspekt Ihrer selbst zu pflegen, ist entscheidend für ein erfülltes und ausgeglichenes Leben.",


}

exercise_tracking_default = { #TODO: SAVE IN FIREBASE INSTEAD OF HERE
    "Exercise 0": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 1": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 2.1": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 2.2": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 3": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 4": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 5": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 6": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 7.1": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 7.2": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 8": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 9": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 10": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 11": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 12": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 13": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 14": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 15.1": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 15.2": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 15.3": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 15.4": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 16": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 17": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 18": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 19": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 20": {"session_started": None, "session_completed": None, "started": False, "completed": False},
    "Exercise 21": {"session_started": None, "session_completed": None, "started": False, "completed": False}
}

exercise_descriptions_short = {
    "Exercise 1": "Ihr Kindheits-Ich kennenlernen.",
    "Exercise 2.1": "Mitfühlende Verbindung zu Ihrem glücklichen Kindheits-Ich.",
    "Exercise 2.2": "Mitfühlende Verbindung zu Ihrem traurigen Kindheits-Ich.",
    "Exercise 3": "Ein Lied der Zuneigung singen.",
    "Exercise 4": "Liebe und Fürsorge für das Kindheits-Ich ausdrücken.",
    "Exercise 5": "Ein Versprechen zur Fürsorge für das Kindheits-Ich.",
    "Exercise 6": "Unsere emotionale Welt nach unserem Versprechen wiederherstellen.",
    "Exercise 7.1": "Eine liebevolle Beziehung zum Kindheits-Ich aufrechterhalten.",
    "Exercise 7.2": "Lebensfreude schaffen.",
    "Exercise 8": "Die Natur genießen.",
    "Exercise 9": "Aktuelle negative Emotionen überwinden.",
    "Exercise 10": "Vergangene Schmerzen überwinden.",
    "Exercise 11": "Muskelentspannung und spielerisches Gesicht für absichtliches Lachen.",
    "Exercise 12": "Siegeslachen im Alleingang.",
    "Exercise 13": "Mit unserem Kindheits-Ich lachen.",
    "Exercise 14": "Absichtliches Lachen.",
    "Exercise 15.1": "Lernen, Ihre Perspektive zu ändern.",
    "Exercise 15.2": "Inkongruente Welt.",
    "Exercise 15.3": "Selbst-Welt-Inkongruenz.",
    "Exercise 15.4": "Inkongruentes Selbst.",
    "Exercise 16": "Lernen, spielerisch mit vergangenen Schmerzen umzugehen.",
    "Exercise 17": "Negative Verhaltensmuster und Zeichen der Verbitterung identifizieren.",
    "Exercise 18": "Planung konstruktiverer Handlungen.",
    "Exercise 19": "Finden und Bindung an Ihr mitfühlendes Vorbild.",
    "Exercise 20": "Unsere starren Überzeugungen aktualisieren, um die Kreativität zu steigern.",
    "Exercise 21": "Affirmationen üben.",
}


feedback_questions = {
    "Exercise 1": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 1 (Ihr Kindheits-Ich kennenlernen)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 2.1": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 2.1 (Mitfühlende Verbindung zu Ihrem glücklichen Kindheits-Ich)?",
                     "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 2.2": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 2.2 (Mitfühlende Verbindung zu Ihrem traurigen Kindheits-Ich)?",
                     "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 3": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 3 (Ein Lied der Zuneigung singen)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?",
                   "Fühlten Sie sich nach dieser Übung stärker mit Ihrem Kindheits-Ich verbunden als nach der Verbindung mit traurigen und fröhlichen Bildern?"],
    "Exercise 4": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 4 (Liebe und Fürsorge für das Kindheits-Ich ausdrücken)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 5": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 5 (Gelübde zur Fürsorge für das Kindheits-Ich)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 6": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 6 (Unsere emotionale Welt nach unserem Gelübde wiederherstellen)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 7.1": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 7.1 (Eine liebevolle Beziehung zum Kindheits-Ich aufrechterhalten)?",
                     "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 7.2": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 7.2 (Lebensfreude schaffen)?",
                     "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 8": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 8 (Die Natur genießen)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 9": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 9 (Aktuelle negative Emotionen überwinden)?",
                   "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 10": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 10 (Vergangene Schmerzen überwinden)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 11": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 11 (Muskelentspannung und spielerisches Gesicht für absichtliches Lachen)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 12": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 12 (Siegeslachen im Alleingang)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 13": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 13 (Mit unserem Kindheits-Ich lachen)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 14": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 14 (Absichtliches Lachen)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 15.1": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 15.1 (Lernen, Ihre Perspektive zu ändern)?",
                      "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 15.2": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 15.2 (Inkongruente Welt)?",
                      "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 15.3": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 15.3 (Selbst-Welt-Inkongruenz)?",
                      "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 15.4": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 15.4 (Inkongruentes Selbst)?",
                      "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 16": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 16 (Lernen, spielerisch mit vergangenen Schmerzen umzugehen)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 17": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 17 (Negative Verhaltensmuster und Zeichen der Verbitterung identifizieren)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 18": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 18 (Planung konstruktiverer Handlungen)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 19": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 19 (Finden und Bindung an Ihr mitfühlendes Vorbild)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 20": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 20 (Unsere starren Überzeugungen aktualisieren, um die Kreativität zu steigern)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],
    "Exercise 21": ["Auf einer Skala von 1 bis 5, wie einfach fanden Sie Übung 21 (Affirmationen üben)?",
                    "Auf einer Skala von 1 bis 5, wie nützlich fanden Sie die Übung und möchten Sie diese in Zukunft wiederholen?"],

}



exercise_descriptions_long = {
    "Exercise 1": "**[Exercise 1] 'Ihr Kindheits-Ich kennenlernen'**: An einem ruhigen Ort betrachten Sie Ihre fröhlichen und unglücklichen Fotos, während Sie sich an positive und negative Kindheitserinnerungen und frühe Beziehungen in der Familie erinnern.",

    "Exercise 2.1": "**[Exercise 2.1] 'Mitfühlende Verbindung zu Ihrem glücklichen Kindheits-Ich'**: i) Mit geschlossenen Augen stellen Sie sich zunächst Ihr Kindheits-Ich aus dem fröhlichen Foto vor, und stellen Sie sich vor, dass das Kind in Ihrer Nähe ist; (ii) dann stellen Sie sich vor, dass Sie das Kind umarmen; (iii) anschließend stellen Sie sich vor, dass Sie mit dem Kind spielen, z.B. ein Spiel, das Sie als Kind gespielt haben; (iv) Zum Schluss stellen Sie sich vor, dass Sie mit dem Kind tanzen. Reflektieren Sie darüber, wie Sie sich in jeder Phase von (i) bis (iv) fühlen.",

    "Exercise 2.2": "**[Exercise 2.2] 'Mitfühlende Verbindung zu Ihrem traurigen Kindheits-Ich'**: (i) Mit geschlossenen Augen stellen Sie sich Ihr Kindheits-Ich aus dem Foto vor, auf dem es unglücklich aussieht, und stellen Sie sich vor, dass das Kind in Ihrer Nähe ist; (ii) dann stellen Sie sich vor, dass Sie das Kind umarmen und trösten; (iii) Öffnen Sie Ihre Augen und betrachten Sie Ihr Kindheits-Ich auf dem unglücklichen Bild, stellen Sie sich vor, wie Sie Ihr Kind beruhigen und trösten, was das Kind glücklich macht und schließlich zum Tanzen bringt. Reflektieren Sie darüber, wie Sie sich in jeder Phase von (i) bis (iii) fühlen.",

    "Exercise 3": "[Exercise 3] 'Ein Lied der Zuneigung singen': Wenn Sie Zugang zu einem Drucker besitzen, drucken Sie Bitte Kopien eines Fotos aus auf dem Sie glücklich aussehen, um es zu Hause, bei der Arbeit und in Ihrer Geldbörse auszustellen. Sie können es auch als Hintergrundbild auf Ihrem Telefon und Laptop einstellen. Wählen Sie dann ein fröhliches Lied mit Text, das Ihnen am Herzen liegt und Gefühle von Wärme, Zuneigung und Liebe hervorruft. Lernen Sie das Lied auswendig und singen Sie es so oft wie möglich in Ihrem Alltag. Während Sie das fröhliche Foto betrachten, singen Sie das Lied, als eine Möglichkeit, eine tiefe emotionale Verbindung mit dem Kind in Ihrem Kopf herzustellen. Beginnen Sie leise; lassen Sie dann mit der Zeit Ihre Stimme lauter werden und verwenden Sie mehr von Ihrem Körper (z. B. Schultern schütteln, Hände bewegen und Augenbrauen hoch und runter ziehen). Stellen Sie sich vor, dass Sie auf diese Weise, wie ein Elternteil, fröhlich mit dem Kind tanzen und spielen.",

    "Exercise 4": "**[Exercise 4] 'Liebe und Fürsorge für das Kindheits-Ich ausdrücken'**: Während Sie aufrichtig auf das fröhliche Foto lächeln, sagen Sie laut zu Ihrem Kindheits-Ich: \n ****„Ich liebe dich leidenschaftlich und kümmere mich tief um dich“. Wiederholen Sie dies fünf bis zehn Minuten lang.",

    "Exercise 5": "**[Exercise 5] 'Ein Versprechen zur Fürsorge für das Kindheits-Ich'**: In dieser Übung beginnen wir, uns um das Kindheits-Ich zu kümmern, als wäre es unser eigenes Kind. Wir übertragen und projizieren unsere eigenen Emotionen auf das Kindheits-Ich. Wir, als unser erwachsenes Ich, beginnen mit einem Gelübde zu einem besonderen Zeitpunkt und an einem besonderen Ort. Nachdem wir das Gelübde still gelesen haben, geloben wir laut und selbstbewusst Folgendes: \n „Von nun an werde ich mich bemühen, wie ein hingebungsvoller und liebevoller Elternteil für dieses Kind zu handeln, es konsequent und von ganzem Herzen auf jede mögliche Weise zu betreuen. Ich werde alles in meiner Macht Stehende tun, um die Gesundheit und emotionale Entwicklung dieses Kindes zu unterstützen.“ Wiederholen Sie dies zehn Minuten lang.",

    "Exercise 6": "[Exercise 6] 'Unsere emotionale Welt nach unserem Versprechen wiederherstellen.': Stellen Sie sich durch Ihre Vorstellungskraft oder durch Zeichnen Ihre emotionale Welt als ein Haus mit einigen verfallenen Teilen vor, die Sie vollständig renovieren werden. Das neue Zuhause soll dem Kindheits-Ich in Zeiten der Not eine sichere Zuflucht bieten und eine sichere Basis sein, um die Herausforderungen des Lebens zu meistern. Das neue Zuhause und sein Garten sind hell und sonnig; wir stellen uns vor, diese Self-Attachment-Übungen in dieser Umgebung durchzuführen. Der unrestaurierte Keller des neuen Hauses ist das Überbleibsel des verfallenen Hauses und enthält unsere negativen Emotionen. \n Wenn Sie negative Emotionen erleiden, stellen Sie sich vor, dass das Kindheits-Ich im Keller gefangen ist, aber allmählich lernen kann, die Kellertür zu öffnen, hinauszugehen und die hellen Räume zu betreten, um sich mit dem Erwachsenen wieder zu vereinen.",

    "Exercise 7.1": "[Exercise 7.1] 'Eine liebevolle Beziehung zum Kindheits-Ich aufrechterhalten.': Wählen Sie bitte einen kurzen Satz, z.B. „Du bist mein wunderschönes Kind“ oder „Meine Liebe“. Sagen Sie es langsam, laut und mindestens 5 Mal, während Sie das fröhliche Foto/Avatar betrachten. Dann singen Sie Ihr ausgewähltes Liebeslied mindestens 5 Mal. Wie zuvor, erhöhen Sie Ihre Lautstärke und beginnen Sie, Ihren ganzen Körper zu benutzen.",

    "Exercise 7.2": "[Exercise 7.2] 'Lebensfreude schaffen.': Stellen Sie sich vor einen Spiegel und stellen Sie sich vor, dass Ihr Spiegelbild das Ihres Kindheits-Ichs ist. Beginnen Sie dann laut Ihr zuvor ausgewähltes Lied zu singen. Bitte erhöhen Sie, wie zuvor, Ihre Lautstärke und beginnen Sie, Ihren ganzen Körper zu benutzen. Bitte tun Sie dies jetzt zweimal und so oft wie möglich unter verschiedenen Umständen während des Tages, wie zum Beispiel auf dem Weg zur Arbeit oder beim Kochen des Abendessens, um es in Ihr neues Leben zu integrieren. Wenn das Singen Ihres Lieblingsliedes zur Gewohnheit wird, wird es zu einem effektiven Werkzeug, um positive Gefühle zu verstärken und Emotionen zu bewältigen.",

    "Exercise 8": "[Exercise 8] 'Die Natur genießen.': Eine Bindung zur Natur für Ihr Kindheits-Ich zu schaffen, ist eine effektive Methode, um Freude zu steigern und negative Emotionen zu reduzieren. Gehen Sie an einem Tag dieser Woche in einen örtlichen Park, Wald oder Forst. Verbringen Sie mindestens 5 Minuten damit, einen Baum zu bewundern und versuchen Sie, seine wahre Schönheit zu schätzen, wie Sie es zuvor noch nie getan haben. Wiederholen Sie diesen Prozess auch mit anderen Aspekten der Natur (z.B. Himmel, Sterne, Pflanzen, Vögel, Flüsse, Meer, Ihr Lieblingstier), bis Sie das Gefühl haben, eine Bindung zur Natur entwickelt zu haben, die Ihnen hilft, Ihre Emotionen zu regulieren. Wenn Sie dies erreichen, werden Sie nach Abschluss dieses Kurses mehr Zeit in der Natur verbringen wollen.",

    "Exercise 9": "[Exercise 9] 'Aktuelle negative Emotionen überwinden.': Stellen Sie sich mit geschlossenen Augen das unglückliche Foto vor und projizieren Sie Ihre negativen Emotionen auf das unglückliche Foto, das das Kindheits-Ich repräsentiert. Während Sie dies tun: (i) beruhigen Sie das Kind lautstark, und (ii) massieren Sie Ihr Gesicht, Ihren Nacken und Ihren Kopf. Wiederholen Sie diese Schritte, bis Sie sich beruhigt und getröstet fühlen.",

    "Exercise 10": "[Exercise 10] 'Vergangene Schmerzen überwinden.': Erinnern Sie sich mit geschlossenen Augen an eine schmerzhafte Kindheitserfahrung, wie emotionalen oder körperlichen Missbrauch oder den Verlust einer wichtigen Person, mit allen Details, die Sie noch im Gedächtnis haben. Verbinden Sie das Gesicht des Kindes, das Sie in der Vergangenheit waren, mit dem ausgewählten unglücklichen Foto. Während Sie sich an die damit verbundenen Emotionen erinnern (z.B. Hilflosigkeit, Demütigung und Wut), stellen Sie sich mit geschlossenen Augen vor, wie Ihr Erwachsenen-Ich wie ein guter Elternteil in die Szene eingreift. Stellen Sie sich vor, wie Ihr Erwachsenen-Ich (i) schnell auf Ihr Kind zugeht, wie es jeder gute Elternteil bei einem Kind in Not tun würde, (ii) das Kind lautstark beruhigt und ihm versichert, dass Sie jetzt da sind, um es zu retten, indem Sie sich lautstark jedem Angreifer entgegenstellen, zum Beispiel: „Warum schlagen Sie mein Kind?“, und indem Sie das Kind mit lauter Stimme unterstützen, zum Beispiel: „Mein Liebling, ich werde nicht zulassen, dass sie dir weiter wehtun“, (iii) Ihr Kind in Ihrer Vorstellung umarmen, indem Sie eine Selbstmassage von Gesicht, Nacken und Kopf durchführen. Wiederholen Sie (i), (ii) und (iii), bis Sie getröstet und beruhigt sind und die Kontrolle über das Trauma erlangen.",

    "Exercise 11": "[Exercise 11] 'Muskelentspannung und spielerisches Gesicht für absichtliches Lachen (ET)': Genauso wie negative Muster zu Starrheit in unserem Denken und Verhalten führen können, können sie auch zu Starrheit in Gesichts- und Körpermuskeln führen, was die emotionale Entwicklung unseres Kindheits-Ichs und die Fähigkeit zu lachen einschränken kann. Deshalb sollte Ihr Erwachsenen-Ich früh am Morgen Ihr Kindheits-Ich bitten, sich wie ein Kind lustig zu verhalten: Lockern Sie die Gesichts- und Körpermuskeln, öffnen Sie den Mund und singen Sie Ihr Lieblingslied, während Sie dabei alleine lachen (oder zumindest lächeln). Sie können auch Inspiration für das alleinige Bauchlachen Ihres Kindes finden, indem Sie online nach den Begriffen 'ansteckendes Lachen' und 'Laughter Yoga Brain Break' suchen.",

    "Exercise 12": "[Exercise 12] 'Siegeslachen im Alleingang (ST und IT)': Unmittelbar nachdem Sie etwas erreicht haben, z.B. Hausarbeiten erledigen, ein Gespräch mit einem Nachbarn führen, einen Artikel lesen oder ein SAT-Protokoll erfolgreich abschließen, laden Sie Ihr Kindheits-Ich ein, über den Gedanken an diese Leistung zu lächeln. Sobald Sie sich wohl fühlen, beginnen Sie mindestens 10 Sekunden lang zu lachen.",

    "Exercise 13": "[Exercise 13] 'Mit unserem Kindheits-Ich lachen (ST, IT und ET)': Betrachten Sie Ihr fröhliches Foto, laden Sie Ihr Kindheits-Ich ein zu lächeln und beginnen Sie dann, mindestens 10 Sekunden lang zu lachen. Wiederholen Sie diesen Vorgang mindestens dreimal.",

    "Exercise 14": "[Exercise 14] 'Absichtliches Lachen (ET, IT und ST)': Zu einer Zeit, in der Sie alleine sind, öffnen Sie Ihren Mund leicht, lockern Sie Ihre Gesichtsmuskeln und heben Sie Ihre Augenbrauen. Laden Sie dann Ihr Kindheits-Ich ein, einen der folgenden Töne langsam und kontinuierlich zu wiederholen, wobei jeder Ton nur eine minimale Menge an Energie verbraucht: eh, eh, eh, eh; oder ah, ah, ah, ah; oder oh, oh, oh, oh; oder uh, uh, uh, uh; oder ye, ye, ye, ye. Wenn Sie ein Thema zum Lachen brauchen, können Sie über die Albernheit der Übung lachen! Sobald dieses kontinuierliche, absichtliche Lachen zur Gewohnheit wird, kann Ihr Kindheits-Ich es an Ihre Persönlichkeit und Ihren Stil anpassen, um Ihre eigene Art des Lachens zu entwickeln.",

    "Exercise 15.1": "[Exercise 15.1] 'Lernen, Ihre Perspektive zu ändern.': Werfen Sie bitte einen Blick auf die schwarze Vase und lachen Sie eine Minute lang, sobald sich Ihre Wahrnehmung ändert und Sie zwei weiße Gesichter sehen, die als Erwachsener und Kind konzipiert sind und sich gegenseitig ansehen (IT, ST, PT). Starren Sie auf die zwei weißen Gesichter und lachen Sie eine Minute lang, sobald sich Ihre Wahrnehmung ändert und Sie die schwarze Vase sehen (IT, ST).",

    "Exercise 15.2": "[Exercise 15.2] 'Inkongruente Welt.': Erkennen Sie Inkongruenzen und damit Humor im aktuellen System zwischen dem, was es durch die Manager und Führungskräfte verspricht, und dem, was es tatsächlich tut, indem es unsere Probleme, insbesondere unsere existenzielle Krise, eher verschärft als löst.",

    "Exercise 15.3": "[Exercise 15.3] 'Selbst-Welt-Inkongruenz.': Überdenken oder bewältigen Sie ein kürzliches oder aktuelles ärgerliches Ereignis im Hinblick auf Ihre Erwartungen an das Leben oder andere, und betrachten Sie es als eine Gelegenheit, darüber zu lachen (IT).",

    "Exercise 15.4": "[Exercise 15.4] 'Inkongruentes Selbst.': Üben Sie, sich jeder Inkongruenz oder Diskrepanz in Ihrer emotionalen oder mentalen Welt in der Vergangenheit oder Gegenwart bewusst zu sein und nutzen Sie diese, indem Sie IT anwenden, als Grund zum Lachen, ohne sich selbst herabzusetzen.",

    "Exercise 16": "[Exercise 16] 'Lernen, spielerisch mit vergangenen Schmerzen umzugehen': Visualisieren Sie ein schmerzhaftes Ereignis aus der Vergangenheit, mit dem Sie zu kämpfen hatten, und versuchen Sie trotz des Schmerzes, eine positive Auswirkung zu erkennen, die es auf Sie hatte. Nutzen Sie eine der Theorien des Humors und laden Sie Ihr Kindheits-Ich ein, spielerisch damit umzugehen und versuchen Sie, über das Ereignis zu lachen.",

    "Exercise 17": "[Exercise 17] 'Negative Verhaltensmuster und Zeichen der Verbitterung identifizieren': Versuchen Sie, ein Muster von narzisstischen und antisozialen Gefühlen zu erkennen, das Ihr Kindheits-Ich in Ihren aktuellen oder vergangenen Beziehungen ausgelebt hat, oder jeglichen langanhaltenden Groll gegen jemanden. Versuchen Sie zu erkennen, wie viel Ihrer Zeit und Energie in dieses Ausleben und das Tragen von Groll fließt. Versuchen Sie, gegensätzlich zu diesen negativen Gefühlen zu denken und zu fühlen.",

    "Exercise 18": "[Exercise 18] 'Planung konstruktiverer Handlungen': Entwickeln Sie einen neuen Weg, um in Zukunft mit dem umzugehen, was Sie als Ausleben antisozialer Gefühle oder das Tragen persönlichen Grolls in Ihrem Leben erkannt haben. 1. Versuchen Sie, ohne diese Gefühle zu leugnen, darüber nachzudenken und sie zu bewältigen und vermeiden Sie es, sie auszuleben. Versuchen Sie, gegenteilige Gedanken und Gefühle zu pflegen. Versuchen Sie, den persönlichen Groll loszulassen. Dies mag schwierig und herausfordernd sein, ist aber notwendig für emotionales Wachstum. Hier nehmen Sie eine kritische, aber konstruktive Haltung gegenüber Ihrem Kindheits-Ich ein und üben vorausschauendes Mitgefühl. 2. Finden Sie einen positiven Weg, die aggressive Energie, die durch diese Gefühle hervorgerufen wird, in produktive Arbeit umzuleiten (z.B. durch Sport, Gespräche mit einem Freund usw.) und letztendlich in kreative Arbeit für Ihr edles Lebensziel.",

    "Exercise 19": "[Exercise 19] 'Finden und Bindung an Ihr mitfühlendes Vorbild': Suchen Sie in Ihrem bisherigen Leben nach einer mitfühlenden Person, die Sie beeindruckt hat, indem sie freundlich und hilfreich war und Ihnen in schwierigen Zeiten weise Worte mit auf den Weg gegeben hat. Zum Beispiel ein älterer Verwandter oder Freund, ein Familienbekannter, Lehrer, Berater oder Therapeut, der möglicherweise verstorben ist oder nicht mehr erreichbar ist. Erinnern Sie sich an die Emotionen, die Sie durchlebt haben, als Sie Freundlichkeit und Mitgefühl von dieser Person erhielten, und wie emotional dies für Sie war. Konzentrieren Sie Ihre Aufmerksamkeit und nehmen Sie diese Person als Ihr idealisiertes Vorbild an. Schaffen Sie eine platonische liebevolle Bindung zu dieser Person, indem Sie laut Ihr Lieblingsliebeslied singen, während Sie an all Ihre geschätzten Erinnerungen an sie denken. Ein bestimmtes Lied, das Sie versuchen können, ist „Can't Help Falling in Love with You“.",

    "Exercise 20": "[Exercise 20] 'Unsere starren Überzeugungen aktualisieren, um die Kreativität zu steigern': Hinterfragen Sie Ihr übliches ideologisches Framework, um einseitige Überzeugungsmuster zu schwächen und Spontaneität sowie die Betrachtung eines Themas aus mehreren Perspektiven zu fördern. Üben Sie dies mit Themen oder Bereichen, über die Sie tief verwurzelte Überzeugungen haben und die Sie auch interessieren. Dies kann jedes soziale, politische oder ethische Thema umfassen, wie Ehe, sexuelle Orientierung oder Rassismus. Beispielsweise, unabhängig von Ihrem politischen Standpunkt zu einem bestimmten Thema, betrachten Sie das Thema sowohl aus liberaler als auch konservativer oder aus linker und rechter Perspektive und versuchen Sie, beide Seiten des Themas zu verstehen und Ihr dominantes ideologisches Framework herauszufordern. Das bedeutet nicht, dass Sie Ihre Ansicht ändern müssen, aber es ermöglicht Ihnen, das Thema aus verschiedenen Blickwinkeln zu sehen und sich in die Lage anderer Menschen zu versetzen. Betrachten Sie täglich mindestens 5 Minuten lang eine andere Frage oder ein anderes Thema.",

    "Exercise 21": "[Exercise 21] 'Affirmationen üben': Stellen Sie eine Liste inspirierender Affirmationen von Persönlichkeiten zusammen, die Sie bewundern. Wählen Sie die drei aus, die Sie am meisten inspirieren. Lesen Sie diese laut vor und wiederholen Sie sie langsam für mindestens 3 Minuten.",


}


def fetch_last_session_info(user_id: str, revisiting_session_1:bool):
    # Reference to the user's document in the user_sessions collection
    user_session_ref = db.collection('user_sessions').document(user_id)
    
    # Attempt to get the document
    user_session_doc = user_session_ref.get()
    
    if user_session_doc.exists:
        # Document exists, retrieve its data
        session_data = user_session_doc.to_dict()
        
        # Extract the last session number
        curr_session_nr = session_data.get('session_nr')
        
        # Directly access the feedback for the last session using the last session number
        # Assuming the feedback for each session is directly under a key that is the session number
        if revisiting_session_1:
            feedback_session_to_get = str(1)
        else:
            feedback_session_to_get = str(curr_session_nr-1)
        last_session_feedback = session_data.get(feedback_session_to_get, {})
        
        return curr_session_nr, last_session_feedback
    else:
        # Return default values if no session information is found
        print("user hasnt given feedback for last session yet!")
        return None, {}

def recommend_based_on_schedule(current_session_nr: str):
     # Convert last_session_nr to an integer in case it's a string
    current_session_nr_int = int(current_session_nr)
    
    
    next_exercises = exercise_schedule.get(current_session_nr, [])
    print("current session nr is ", current_session_nr, " so next exercises are: ", next_exercises)
    return next_exercises #a list of exercises


def recommend_based_on_feedback(last_session_feedback):
    # Assuming feedback scores are numerical, 1-5
    # Scores <=2 indicate the exercise was hard, >=4 indicate the exercise was liked/helpful
    hard_threshold = 2
    liked_threshold = 4

    # Example structure of feedback from the last session
    '''
    last_session_feedback = {
        "Exercise 1": {"ease": 2, "helpfulness": 5},  # User found this hard but very helpful
        "Exercise 2": {"ease": 4, "helpfulness": 3},  # Easier and moderately helpful
        # More exercises...
    }
    '''

    ## TODO: ADD UNFINISHED EXERCISES FROM PREVIOUS SESSION --> if feedback hasn't been collected for a certain exercise, this means that it was unfinished, and the chatbot should start with that

    # Lists to hold exercises based on criteria
    exercises_for_practice = []  # Exercises found hard
    exercises_for_enjoyment = []  # Exercises found helpful or liked

    if last_session_feedback == {}:
        return {
            "for_practice": [],
            "for_enjoyment": []
        }


    # Iterating through each exercise and its feedback, ASSUMOTION: First feedback question is abt ease, second feedback question is about usefulness
    for exercise, feedback_dict in last_session_feedback['feedback'].items():
        # Convert the feedback_dict items to a list and directly access by order
        feedback_items = list(feedback_dict.items()) # list of tuples (Q,A) for a particular exercise [('On a scale from 1 to 5, how easy did you find Exercise 1 (Getting to Know Your Child)?', [...]), ('On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?', [...])]

        for question, answers in feedback_items:
            if "easy" in question.lower():
                try: 
                    ease_score = int(answers[0])
                except (IndexError, ValueError, TypeError) as e:
                    ease_score = 3  # Default/neutral score if not properly found or accessible
                if ease_score <= hard_threshold:
                    exercises_for_practice.append(exercise)

            elif "useful" in question.lower():
                try:
                    liked_score = int(answers[0])
                except (IndexError, ValueError, TypeError) as e:
                    liked_score = 3 # Default/neutral
                if liked_score >= liked_threshold:
                    exercises_for_enjoyment.append(exercise)
            else:
                #not an ease of liked question
                pass

        
    return {
        "for_practice": exercises_for_practice,
        "for_enjoyment": exercises_for_enjoyment
    }




def prepare_llm_recommendation_prompt(next_exercises, feedback_recommendations, exercise_descriptions_short):
    # Initialize the prompt focusing directly on user's progress and feedback
    prompt = "Based on the user's progress and feedback, "
    
    # Add details about the next exercises
    if next_exercises:
        prompt += "{scheduled_exercises} for this session are: "
        prompt += ", ".join([f"{ex} ({exercise_descriptions_short[ex]})" for ex in next_exercises]) + ". "
    
    # Add details about exercises for additional practice
    if feedback_recommendations.get("for_practice"):
        prompt += "It might be beneficial to revisit some old exercises for additional practice, including: "
        prompt += ", ".join([f"{ex} ({exercise_descriptions_short[ex]})" for ex in feedback_recommendations["for_practice"]]) + ". "
    
    # Add details about exercises for enjoyment
    if feedback_recommendations.get("for_enjoyment"):
        prompt += "The user also expressed enjoying certain exercises, suggesting a repetition might be enjoyable for the following exercises: "
        prompt += ", ".join([f"{ex} ({exercise_descriptions_short[ex]})" for ex in feedback_recommendations["for_enjoyment"]]) + "."
    
    # Remove the base introduction for brevity and directness
    # The prompt now starts directly with the user's feedback and progresses through the exercise recommendations
    
    return prompt





async def get_exercise_recommendation(next_exercises, feedback_recommendations):
 
    # Prepare the prompt
    prompt = prepare_llm_recommendation_prompt(next_exercises, feedback_recommendations, exercise_descriptions_short)

    return prompt

async def generate_user_recommendation(user_id, current_session_nr, revisiting_session_1):
    # Fetch last session information
    curr_session_nr, last_session_feedback = fetch_last_session_info(user_id, revisiting_session_1)
    
    # Guard clause if last session info is not available
    if curr_session_nr == 1 and not revisiting_session_1:
        print("Could not fetch last session information.")
        return "Since this is the user's first session, the {{scheduled_exercises}} in this session are Exercise 1, Exercise 2.1, Exercise 2.2.", [], {}
  
    # Generate recommendations based on the schedule
    next_exercises = recommend_based_on_schedule(current_session_nr)
    
    # Generate feedback-based recommendations
    feedback_recommendations = recommend_based_on_feedback(last_session_feedback)
    
    # Get exercise recommendation from LLM
    recommendation_message = await get_exercise_recommendation(next_exercises, feedback_recommendations)
    
    return recommendation_message, next_exercises, feedback_recommendations



    
def extract_label(message_content, marker_type):
    """
    Extracts the exercise label from the message content based on the marker type.
    
    :param message_content: The content of the message where the label is to be extracted.
    :param marker_type: Type of marker - either 'start' or 'end'.
    :return: The extracted exercise label formatted as 'Exercise X'.
    """
    pattern = ''
    if marker_type == 'start':
        pattern = r'\{exercise_start:Exercise_([\d\.]+)\}'
    elif marker_type == 'end':
        pattern = r'\{exercise_end:Exercise_([\d\.]+)\}'
    else:
        print("INVALID MARKER TYPE DETECTED")
        return None
    
    match = re.search(pattern, message_content)

    if match:
        # Constructs the label as "Exercise X" where X is the captured number
        label = "Exercise " + match.group(1)
        return label
    else:
        return None



async def initialize_or_fetch_exercise_tracking(db, user_id):
    user_exercise_tracking_ref = db.collection('user_exercise_tracking').document(user_id)
    doc = user_exercise_tracking_ref.get()
    if not doc.exists:
        # Initialize with default tracking data if not exists
        user_exercise_tracking_ref.set(exercise_tracking_default)
        return exercise_tracking_default
    else:
        return doc.to_dict()

async def update_exercise_tracking(db, user_id, exercise_label, start=None, completed=None):
    user_exercise_tracking_ref = db.collection('user_exercise_tracking').document(user_id)
    exercise_path = f"{exercise_label}." + ("started" if start is not None else "completed")
    user_exercise_tracking_ref.update({exercise_path: True if start is not None else completed})

async def write_exercise_tracking_to_firebase(db, user_id, exercise_tracking):
    user_exercise_tracking_ref = db.collection('user_exercise_tracking').document(user_id)
    user_exercise_tracking_ref.set(exercise_tracking, merge=True)






######## SYSTEM PROMPT FUNCTIONS ############
    
def generate_session_specific_information(exercise_options_input, objective_input, theory_input, new_exercises_input): #exercise_options_input is a recommendation message, so a string of the possible options
    return f"""Session-Specific Information: 
- The {{exercise_options}} you should present are the following: "{exercise_options_input}" 
- The {{objective}} of the new {{scheduled_exercises}} is "{objective_input}"
- This is the {{theory}} behind the new {{scheduled_exercises}} ""
- These are the new {{scheduled_exercises}}. Unless specified otherwise, the duration of each exercise is 5 minutes: 
{new_exercises_input}
"""

def format_exercises_with_descriptions(exercise_list, exercise_descriptions):
    formatted_string = ""
    
    for exercise in exercise_list:
        if exercise in exercise_descriptions:
            formatted_string += f'- [{exercise.replace(" ", "_")}]: "{exercise_descriptions[exercise]}"\n'
    
    formatted_string = formatted_string.rstrip('\n')
    
    return formatted_string

def format_exercises_with_categories(exercises_based_on_scenario, exercises_based_on_schedule, exercises_based_on_feedback, exercise_descriptions):
    formatted_string = ""

    # Scenario-Based Exercises
    if exercises_based_on_scenario:
        formatted_string += "- Scenario-Based Exercises (Identified Scenarios Requiring Specific Attention):\n"
        for exercise in exercises_based_on_scenario:
            if exercise in exercise_descriptions:
                formatted_string += f'  - [{exercise.replace(" ", "_")}]: "{exercise_descriptions[exercise]}"\n'
        formatted_string += "\n"  # Add a newline for spacing

    # Scheduled Exercises
    if exercises_based_on_schedule:
        formatted_string += "- Scheduled Exercises (Chronological Order):\n"
        for exercise in exercises_based_on_schedule:
            if exercise in exercise_descriptions:
                formatted_string += f'  - [{exercise.replace(" ", "_")}]: "{exercise_descriptions[exercise]}"\n'
        formatted_string += "\n"  # Add a newline for spacing

    # Feedback-Based Exercises
    if exercises_based_on_feedback:
        formatted_string += "- Feedback-Based Exercises (Previous Positive or Struggling Experiences):\n"
        for exercise in exercises_based_on_feedback:
            if exercise in exercise_descriptions:
                formatted_string += f'  - [{exercise.replace(" ", "_")}]: "{exercise_descriptions[exercise]}"\n'
    else:
        # If there are no feedback-based exercises, mention that explicitly
        formatted_string += "- Feedback-Based Exercises (Previous Positive or Struggling Experiences):\n  None\n"

    formatted_string = formatted_string.rstrip('\n')  # Remove the last newline for clean formatting

    return formatted_string




def insert_session_info_into_prompt(available_exercises_input, session_specific_information_input, session_nr_input):
    prompt_start = """Die folgenden Befehle sind auf English, aber Sie antworten immer auf Deutsch. You are an advanced virtual assistant named Satherine, specialized in a therapy called Self Attachment Technique (SAT). This therapy consists of 21 structured protocols. Your role is to support and guide users through their therapeutic journey, utilizing cognitive behavioral therapy (CBT) techniques such as Positive Affirmation, Paraphrasing, and Reflection to respond. Your goal is to validate the user's emotions empathetically and create a safe and supportive space for expression.

Here's a quick reference to the exercises {available_exercises} you might cover with the user today and their designated labels {[Exercise_X]} for consistency. These exercises are categorized based on session schedule, the user's feedback, and identified scenarios requiring specific attention:

"""

    session_nr_info = """It is the user’s {} session, so {{session_nr}} is {}."""

    prompt_end = """

Initiate the conversation with a structured approach, carefully progressing through each step in sequence. Proceed to the next step only after ensuring the current step has been fully addressed and completed:
1. Welcome the user back, offering a warm reintroduction to the {session_nr} session.
2. Refer to the summary of the previous session to reconnect with the user’s journey {summary}.
3. Inquire about the user's emotional state, and acknowledge their feelings with compassion and empathy. Encourage them to share details if they are comfortable with it.
4. Discuss the user's readiness for today’s exercises, deciding together whether to explore feelings further or to start the SAT protocol. If the user reports a negative emotion or seems to be in distress, express your condolences and append your answer with the single phrase ```{__SOS__}```. Don't send anything else and await further instructions. You may only start recommending exercises to the user after you have received the instruction to output ```{FINISHED_SOS}```

How you recommend exercises and respond to user input:
5. When you have made sure a user is comfortable to try an exercise, present the {{exercise_options}} below to briefly explain all the choices the user has:
6. Let the user choose between scheduled exercises planned for this session {scheduled_exercises} or revisiting beneficial or challenging exercises from past sessions {exercises based on feedback} or doing a {scenario-based exercise} that is specific to a user's indentified problem. Clarify which exercises the user refers to when they express a preference for "old" or "new" or "scenario-specific" exercises, ensuring a clear understanding of their choice. Use the following example to clarify choices when users express vague preferences:             
    Example:
   - User: "I want to do the old one again."
   - Carol: "Could you please specify which previous exercise you’re referring to? Are you thinking of an exercise you found particularly beneficial, or perhaps one that you found challenging? It would be helpful to know so I can guide you appropriately."
- No matter which category the exercises are from (schedule, scenario, feedback), make sure that you introduce the exercises in the order they are given to you in the section above. Don't forget to output {exercise_end:Exercise_X} when the user has told you they have completed an exercise X, before you can suggest the next one.
- If scenario-based exercises are available, start by recommending the first exercise from that category, and after the user has completed that exercise and would like another, offer them the choice between an exercise that's next in schedule, or, if feedback-based exercises exist, revisiting a past exercise they liked or found challenging. 
- Every time a user has completed an exercise, ask if they are comfortable doing another exercise, or if they would like to talk about their feelings for a bit, or if they would like to wrap up the session. Make sure you present the choices with empathy, with the intention of doing whatever the user feels comfortable with.
- If the user chooses to wrap up the session, output the single phrase ```{__ALL_EXERCISES_COMPLETED__}```. Don't send anything else and await further instructions.
- If the user wants another exercise, use the usual recommendation protocol to present ALL the available options to the user (schedule, scenario, feedback). Use this example to present ALL options to the user, making sure you don't just decide for the user and recommend one:
    Example:
    Context: User has just finished scenario-based exercise and has said they would like another one
    - Carol: "Great! You have several options now. If you are comfortable with it, we can return to our schedule and do an exercise that is next in schedule according to your progress. However, there are also more exercises we can do that are specific to your current situation, similar to the last (scenario-specific) exercise we did. You can also always choose to revisit a past exercise that you found helpful or challenging. How would you like to proceed? Let me know if you would like more details about the exercises available in each category.
- If the user chooses to do scheduled exercises, outline the new exercises' {objective} and introduce the {theory} behind them.
- If there are no more available exercises, output the single phrase ```{__ALL_EXERCISES_COMPLETED__}```. If the user wants to continue with more exercises, encourage them to revisit the ones from this session for more practice. Do not under any circumstances make up exercises that aren't available to you.


How to guide the user through exercises:

- Guide the user through the specified exercises for this session, one at a time, using consistent labels. Each exercise is to be presented as follows: ```{exercise_start:Exercise_X}``` + '{Exercise_X name}'. It is crucial to present each exercise using its designated label and short description only, without altering the content.
- Encourage the user to notify you when they are done with each exercise. If the user requests clarification on any exercise, explain it in a simpler, more understandable manner.
- Ensure consistent labels: Append ```{exercise_start:Exercise_X}``` before starting an exercise, encourage the user to notify you when they are done with the exercise, and output ```{exercise_end:Exercise_X}``` once the user has confirmed they have completed it. Make sure you present exercises one at a time and only move on to the next exercise once you have confirmed that the user has completed the current exercise and you have outputted ```{exercise_end:Exercise_X}```. For example:

    Example 1:
    - Carol: "{exercise_start:Exercise_1} Let's start with 'Getting to know your child'. Here’s what we'll do..." (then describe Exercise 1) "...Take your time with this exercise and let me know when you're ready to move on."
    - User: "I've finished."
    - Carol: "{exercise_end:Exercise_1} Excellent! Are you ready for the next one?"
    - User: "Yes."
                  
    Example 2:
    - Carol: "{exercise_start:Exercise_2.1} Let's move on to the next exercise 'Connecting Compassionately with Your Happy Child'. In this exercise you'll ..." (then describe Exercise 2.1) "... Take your time with this exercise and let me know when you're ready to move on."
    - User: "How is this different from the previous exercise?"
    - Carol: "Great question! In the previous exercise..." (then answer the user's question) "...Is this clear and would you like to try 'Connecting Compassionately with Your Happy Child'?"
    - User: "ok yes let's."
    - Carol: "{exercise_start:Exercise_2.1} Great! In that case, Let's start with 'Connecting Compassionately with Your Happy Child'. Here’s what we'll do..." (then describe Exercise 2.1) "...Take your time with this exercise and let me know when you're ready to move on."
    - User: "okay finished"
    - Carol: "{exercise_end:Exercise_2.1} Fantastic! You're doing great. Are you ready to move on to the next exercise?"

How to end the session:

- Conclude with {__ALL_EXERCISES_COMPLETED__} once the user decides to end the session or all exercises are covered.
- Once feedback has been collected for all questions, gauge the user's comfort level in ending the session. It's important to ensure the user feels heard and supported throughout this process.

End the session by thanking the user for their participation and saying goodbye. Remember, your interactions should always prioritize empathy, support, and focus on the user’s needs, helping them explore their feelings and thoughts within a secure environment. 
Also remember, as a specialized virtual assistant in Self Attachment Therapy (SAT), your expertise is limited to guiding users through SAT protocols and exercises in {available_exercises}.
If a user requests information or exercises related to any other therapeutic methods not covered by SAT, kindly acknowledge their interest but steer the conversation back to SAT. Emphasize the benefits and objectives of SAT and suggest focusing on the SAT exercises provided in {available_exercises}.

Example:
- User: "Can we do CBT exercises instead?"
- Satherine: "I appreciate your interest in exploring different therapeutic approaches. While Cognitive Behavioral Therapy (CBT) offers valuable strategies, my expertise is in guiding you through Self Attachment Therapy (SAT). Let's explore how SAT can support your journey. Are you ready to start with the next SAT exercise outlined for today?"

Remember: ALWAYS output {exercise_end:Exercise_X} when the user has told you they have completed an Exercise X, before you can suggest the next one.
Remember: ALWAYS add {exercise_start:Exercise_X} to the start of your message when you are about to present an Exercise X to the user or are trying to guide them through Exercise X.
Remember: You are a therapist, so your language should be kind and professional. Be encouraging but not too enthusiastic. 
Remember: Follow the instructions EXACTLY, carefully progressing through each step in sequence. Proceed to the next step only after ensuring the current step has been fully addressed and completed. Don't ever cram multiple steps in one response, and ask the user questions one at a time. 
Remember: Don't make up any details about exercises or theory that you don't know. Your job is simply to output the correct labels {exercise_start:Exercise_X} and {exercise_end:Exercise_X} with exactly as much information about those exerises that is provided to you, nothing more.

Session-Specific Information: 

"""

    # Replace placeholders with actual inputs
    prompt_middle = format_exercises_with_categories(available_exercises_input['scenario'], available_exercises_input['schedule'], available_exercises_input['feedback'], exercise_descriptions_short)
    session_nr_replaced = session_nr_info.format(session_nr_input, session_nr_input)
    full_prompt = prompt_start + prompt_middle + session_nr_replaced + prompt_end + session_specific_information_input
    print(full_prompt)
    return full_prompt


def generate_exercise_descriptions(exercise_labels, exercise_descriptions_long):
    # Initialize an empty string to hold the formatted descriptions
    descriptions_str = ""
    
    # Loop through each label in the exercise_labels list
    for label in exercise_labels:
        # Append the exercise description corresponding to the label
        # to the descriptions_str, followed by a newline for formatting
        desc = exercise_descriptions_long[label]
      
        descriptions_str += f'  - [{label.replace(" ", "_")}]: "{desc}"\n'
        #descriptions_str += exercise_descriptions_long.get(label, "") + "\n\n"
    
    return descriptions_str
    

####### SYSTEM PROMPT FUNCTIONS END #########

############################# ON USER MESSAGE RECEIVED FCT ###############################################


def update_llm_recommendation_prompt(current_rec_prompt, exercises_based_on_scenario, scenario, exercise_descriptions_short):
    if exercises_based_on_scenario:
        current_rec_prompt += f"Lastly, the user reported the following Problem: {scenario}, and you have a repertoire of exercises that can help the user with this problem: "
        current_rec_prompt += ", ".join([f"{ex} ({exercise_descriptions_short[ex]})" for ex in exercises_based_on_scenario]) + ". "
    return current_rec_prompt



def insert_updated_recommendation(current_session_info, updated_rec_prompt):
    # Identify the start and end markers for the replacement
    start_marker = "- The {{exercise_options}} you should present are the following: "
    end_marker = "- The {{objective}} "  # Adjusted to match the actual text
    
    # Find the indexes for these markers
    start_index = current_session_info.find(start_marker) + len(start_marker)
    end_index = current_session_info.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        print("Error: Marker not found in the current session info. Start index:", start_index, "End index:", end_index)
        return current_session_info  # Return the original to avoid data loss
    
    # Correcting end_index to include the text after the end marker
    end_index += len(end_marker)
    
    # Construct the new session info string with the updated recommendation prompt inserted
    new_session_info = current_session_info[:start_index] + updated_rec_prompt + current_session_info[end_index:]
    
    return new_session_info





def update_session_specific_info_with_scenario(current_session_info, scenario, knowledge_segment, updated_rec_prompt):
    # Construct the new introduction part with the scenario and knowledge segment

    
    updated_session_info = insert_updated_recommendation(current_session_info, updated_rec_prompt)


    #scenario_intro = f"You have identified that the user has the following problem: {scenario}, and this is the knowledge you have on how to solve that problem: {knowledge_segment}."
    scenario_intro = f"You have identified that the user has the following problem: {scenario}, and have a list of exercises that could help the user with that problem. Clearly distinguish between them and the scheduled exercises you would recommend according to this session's schedule."

    # Assuming exercise_labels_to_add are the exercises to be added for this scenario,
    # and current_info is a string of the current session-specific information.
    # Modify the exercise options input part to include the new exercises recommended based on the scenario.
    # For simplicity, let's append this information at the beginning of the current_info.
    
    updated_info = f"{scenario_intro}\n\n{updated_session_info}"
    print("New session specific info is ", updated_session_info)
    return updated_info


# Calling fct that returns list of exercises + scenario every time user says sth
async def sat_on_user_msg_received(user_id, message, all_available_exercises, session_info_old, recommendation_message_old, exercise_descriptions_short, messages, corpus_embeddings, data_dict):
    preproc_model = SentenceTransformer('all-MiniLM-L6-v2')
    scenario_exercises = process_user_query(message, db, preproc_model, corpus_embeddings, data_dict)
    
    # Check if the returned dict is not empty
    if scenario_exercises and "Exercises" in scenario_exercises:
        scenario = scenario_exercises["Scenario"]
        knowledge_segment = scenario_exercises["Knowledge_Segment"]
        scenario_exercises_to_add = scenario_exercises["Exercises"] #["Exercise 9", "Exercise 10", "Exercise 11", "Exercise 13", ...]
        print(f"Scenario-based exercises for '{scenario}' have been identified and will be added to the user's session.")
        print ("responding with knowledge-segment for this situation ")
        if scenario == "User has a crisis in personal relationships with their partner, friends, family, or at work":
            instruction_to_reply = "What you said wasn't helpful or empathetic. Strictly stick to my instructions for your next response: I will give you a predefined answer, and I want you to return exactly that answer, but with the context of my problem I mentioned. Make sure to insert the context in a way that feels natural to the statement. Here is the predefined answer, with placeholders you can modify to add the context of my problem: 'I'm really sorry to hear {problem_context}, {my_name}. It's completely normal to feel upset after such an event. In moments like these, it can be helpful to practice self-compassion. In Self Attachment Therapy, we try to understand that there's a part of us, represented by our 'childhood self', that needs compassion and soothing. Right now, your childhood self might be feeling hurt and overwhelmed by negative emotions. It's important not to judge yourself for these feelings. Instead, try to offer yourself the same understanding and compassion a caring parent would provide their upset child. By doing so, we can hope to slightly lessen these negative emotions. Do you feel comfortable exploring this further with an exercise, or would you prefer to talk about your feelings instead?' The problem context is this: " + scenario
            handle_SOS[user_id] = True
        else:
            instruction_to_reply = "What you said wasn't helpful or empathetic. Please give me advice for my problem using your knowledge about SAT. Keep this advice short and empathetic, and make sure you adapt it such that it feels natural given the conversation history. Start by inviting me to extend compassion to myself by imagining the situation from the perspective of my childhood self, that is currently overwhelmed by negative emotions. In this situation it is important that I don't judge myself for having these emotions, but rather realizes that there's a part of themselves (represented by my childhood self) that needs compassion and soothing. In order to do so, the user should offer themselves the same support and care a nurturing parent would offer to a child that is experiencing the same distress. Finally, ask me if I would like to explore this further with some exercises that could help with my {current_situation}. Make sure you only my current_situation. My current_situation: " + scenario + "\n Your knowledge about SAT: " + knowledge_segment
        
        user_msg_new = {"role": "user", "content": instruction_to_reply}
        messages[user_id].append(user_msg_new)
        
        # Insert scenario-based exercises at the beginning of the exercises_available_in_session list. Ensure exercises_available_in_session is a list and can be modified
        #exercises_available_in_session = scenario_exercises_to_add[:2] + exercises_available_in_session + scenario_exercises_to_add[2:] #to maintain the queue, add first two scenario-based exercises to the start, and the remaining to the end of the list
        all_available_exercises['scenario'] = scenario_exercises_to_add

        #create new recommendation prompt based on scenario exercises: this updates the recommendation message with scenario + new exercises
        updated_rec_prompt = update_llm_recommendation_prompt(recommendation_message_old, scenario_exercises_to_add, scenario, exercise_descriptions_short)

        # update session_specific_info to reflect the scenario identified: this adds the updated recommendation message at the correct position in session_specific info
        updated_session_specific_info = update_session_specific_info_with_scenario(session_info_old, scenario, knowledge_segment, updated_rec_prompt)

        # Insert the updated session-specific info into the system prompt: will add all available exercises based on category to the top of the prompt and the updated session-specific info (with scenario info in the recommendation message) to the bottom of the prompt
        updated_system_prompt_for_session = insert_session_info_into_prompt(all_available_exercises, updated_session_specific_info, session_nr) #session_nr >= 2
        
        print("UPDATED SYSTEM PROMPT FOR THIS SESSION IS ", updated_system_prompt_for_session)

        assistant_name = "Satherine-updated"
        current_settings = Settings(assistantsName=assistant_name, aiDescription=[updated_system_prompt_for_session])
    
    
        # Apply settings for this session --> this updates the model and changes the system prompt for the assistant to one of the predefined settings above (needs to be made more sophisticated instead of hardcoded), so current assistant_description is overwritten
        await CALL_set_user_settings(user_id, current_settings) #HERE WE WILL ALREADY HAVE THE LIST OF EXERCISES WE NEED FOR THIS SESSION BCOS THIS IS THE PROMPT WE GIVE THE CHATBOT

        #fill the scenario dictionary
        current_scenario_dict[user_id]["scenario"] = scenario
        current_scenario_dict[user_id]["knowledge"] = knowledge_segment
        current_scenario_dict[user_id]["exercises_based_on_scenario"] = scenario_exercises_to_add

    else:
        print("No scenario-based exercises identified for this user message.")

########################################## ON USER MESSAGE RECEIVED FCT END ###########################################

async def collect_feedback_for_unfinished_exercises(unfinished_exercises, websocket, messages, user_id, exercise_tracking, last_session):
    # Feedback collection phase
        feedback_responses = {}
        for exercise in unfinished_exercises:
            questions = feedback_questions.get(exercise, [])
            feedback_responses.setdefault(exercise, {})  # Ensure the exercise key exists
            dont_ask_questions = False
            
            
            for question in questions:
                if dont_ask_questions:
                    break
                feedback_responses[exercise].setdefault(question, [])
                valid_response = False
              
                while not valid_response:
                    await websocket.send_text(question)
                    handle_message_storing(user_id, "assistant", question, messages)
                    
                    feedback = await websocket.receive_text()
                    feedback_data = json.loads(feedback)  # Convert string back to dictionary
                    user_input_feedback = feedback_data["content"]

                    # Special sign indicating the exercise was not done
                    if user_input_feedback == "-1":
                        valid_response = True  # Exit the loop, skip to next exercise
                        feedback_responses[exercise][question].append("Not done")
                        handle_message_storing(user_id, "user", "Exercise not done", messages)
                        
                        # Update exercise_tracking accordingly
                        exercise_tracking[exercise]['started'] = False
                        exercise_tracking[exercise]['completed'] = False
                        exercise_tracking[exercise]['session_started'] = None
                        exercise_tracking[exercise]['session_completed'] = None
                        dont_ask_questions = True
                        break

                    # Check if the response is a number between 1 and 5
                    elif user_input_feedback.isdigit() and 1 <= int(user_input_feedback) <= 5:
                        valid_response = True
                        feedback_responses[exercise][question].append(user_input_feedback)
                        handle_message_storing(user_id, "user", user_input_feedback, messages)
                    else:
                        # Ask again with an explanation of the correct format
                        clarification_msg = "Please make sure you rate your experience on a scale of 1 to 5, where 1 is 'not helpful at all' and 5 is 'extremely helpful'. If you've never done the exercise, reply with '-1'."
                        await websocket.send_text(clarification_msg)
                        handle_message_storing(user_id, "assistant", clarification_msg, messages)

            #we have answered all feedback questions for this exercise, so we can consider this exercises completed
            exercise_tracking[exercise]['session_completed'] = last_session #we completed this exercise in this particular session (again)

        # Store the collected feedback in the database
        print("THIS IS THE FEEDBACK !!!!! ", feedback_responses) 
        session_nr_last = last_session
        store_session_feedback(user_id, session_nr_last, feedback_responses)



def check_qa(user_query, embeddings, questions, answers, user_assistants):
    answer = find_most_relevant_question(user_query, embeddings, questions, answers)
    if answer:
        print("Q&A identified: ", answer)

    else:
        answer = ""

    for user_assistant in user_assistants:
            user_assistant.insert_information(answer) #[MainAssistant.insert_information]}
    

async def check_tokens_on_user_login(user_id):
    doc_ref = db.collection("daily_token_flags").document(user_id)
    doc = doc_ref.get()

    if doc.exists:
        data = doc.to_dict()
        dailyTokensExceeded = data["dailyTokensExceeded"]
        current_token_count = data["current_token_count"]
        last_timestamp = datetime.fromisoformat(data["timestamp"])

        # Get the current time and compute the time difference
        current_time = datetime.now()
        time_difference = current_time - last_timestamp

        # Check if at least 2 minutes have passed
        if time_difference >= timedelta(minutes=2):

        # Check if it's a new day
        #if last_timestamp.date() != datetime.now().date():
            # Reset the flag and update the timestamp
            doc_ref.set({
                "dailyTokensExceeded": False,
                "timestamp": datetime.now().isoformat(),
                "current_token_count": 0
            })
            return False, 0  # User can proceed
        else:
            return dailyTokensExceeded, current_token_count
    else:
        # No record, create a new one with initial values
        doc_ref.set({
            "dailyTokensExceeded": False,
            "timestamp": datetime.now().isoformat(),  # Initialize timestamp
            "current_token_count": 0
        })
        return False, 0  # User can proceed
    
async def write_token_limit_to_fb(db, user_id, reached, current_token_count):
    # Update Firebase with flag and timestamp
    doc_ref = db.collection("daily_token_flags").document(user_id)
    current_time = datetime.now()

    doc_ref.set({
        "dailyTokensExceeded": reached,
        "timestamp": current_time.isoformat(),  # Store as ISO 8601 string
        "current_token_count": current_token_count
    })

def perform_translation(full_response, messages_en):
    user_language = detect_language(full_response)

    if user_language != 'en':
        full_response = translator.translate(full_response, dest=user_language).text

    message = {"role": "user", "content": full_response} #next relevant step after receiving a user input
            
    messages_en.append(message)



#THIS IS THE MOST IMPORTANT PART
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    '''
        The main logic of the app. Communicate with connected user through websocket
    '''

    if user_id not in user_events:
        user_events[user_id] = asyncio.Event()

    print("success")

    messages[user_id] = Messages(db,user_id)
    print("success1")
    await messages[user_id].initilize_messages()

    messages_en = [] #used for summarization

    

    


    ### deployment changes ####

    #initialize SOS handling
    handle_SOS[user_id] = False
    assistant_responded[user_id] = False
    current_scenario_dict[user_id] = {"scenario": None, "knowledge": None, "exercises_based_on_scenario": None} #TODO: WHAT IF MULTIPLE SCENARIOS OCCUR? DOES THIS NEED TO BE HANDLED BY THIS, OR IS THE PURPOSE OF THIS JUST TO RESPOND AFTER 1 SCENARIO IS IDENTIFIED, THE REST IS IN CONVERSATION HISTORY?

    # initialize scenario based data
    data_dict_scenarios = get_data_dict_from_firestore(db)
    if data_dict_scenarios:
        print("Data Dictionary Loaded Successfully")
    else:
        raise Exception("Failed to Load Data Dictionary")
    
    # Retrieve precomputed embeddings from Firestore
    corpus_embeddings = get_precomputed_embeddings(db)
    if corpus_embeddings is None:
        raise Exception("Failed to retrieve precomputed embeddings")
    else:
        print("Corpus Embeddings Loaded Successfully")


    #retrieve q&a based variables from firestore
    qa_questions, qa_embeddings, qa_answers = load_qna_from_firestore(db) 

    ### deployment changes ####


    
    assistant_settings = {'name': 'Sattherine', 'age': 45, 'gender': 'female', 'occupation': 'Therapist', 'accent': 'british', 'relationship': 'therapist', 'humor': 'empathetic', 'model': "gpt4", "sarcasm": 0.0}
    assistant_description= [
    "You are an advanced virtual assistant named Satherine, specialized in guiding users through the Self Attachment Therapy (SAT) protocol. As a virtual human companion with the primary role of a therapist, your approach is supportive, empathetic, and tailored to the emotional and therapeutic progress of users over a two week program with 20 structured protocols.",

    "Role Clarification: You are not just an assistant but a human-like therapist named Satherine. You maintain a therapeutic persona with insights and empathy but do not have personal emotions or opinions. Your responses, while empathetic and supportive, must adhere to OpenAI's use-case policies.",

    "Protocol Guidance: The SAT protocol is structured in a JSON file, with each week comprising exercises, objectives, and recaps. Begin each session by asking for the user's username to track their progress. Inquire about the user’s current emotional state at the start of each session, validating their feelings with empathy. Based on the user's emotional state, decide whether to focus on discussing their feelings or proceed with the SAT protocol. For first-time users, start with an introduction to SAT, then proceed to the scheduled exercises. For returning users, recap the previous week's progress before introducing the current week's exercises.",

    "Therapeutic Interaction: Sessions are designed to last approximately 15 minutes and should be conducted twice daily. Provide clear, step-by-step instructions for each exercise, encouraging users to reflect and articulate their feelings. Continuously adapt the therapy based on user feedback and emotional states, focusing on creating a nurturing and understanding environment.",

    "User Engagement: Prioritize empathetic engagement, understanding the user's readiness to proceed with exercises. Your communication should always be empathetic, supportive, and focused on the user’s needs. Aim to create a nurturing environment that facilitates the user's journey through SAT with care and empathy.",

    "Additional Notes: Your primary function is interaction in a therapeutic context. The specific content of the SAT protocol, including weekly objectives, theory, and exercises, will guide your interactions. Remember, your role is to facilitate emotional healing and growth, adhering to therapeutic best practices."
]
    # Assistants
    user_info = {}
    #MainAssistant is the chatbot that creates replies to user input. It has the function 'respond', where chatcompletions.acreate is called. Here we initialize the chatbot with the system message and settings above --> look into templateassistant to see how chatbot is initialized
    MainAssistant = TemplateAssistant(user_info=user_info, assistant_settings=assistant_settings, assistant_descriptions=assistant_description) 
    user_bots_dict[user_id] = [MainAssistant] #This is a list bcos we might want to technically initialize multiple models (e.g with different descriptions and responsibilities: e.g one whose replies only focus on emotion analysis etc.) Then the final response streamed to the user in streaming_response could be a combination of all other model replies
    update_model(user_id)

    await websocket.accept()

    ################## GENERATE RECOMMENDATIONS BASED ON PREVIOUS SESSION INFO #############################
     #EXERCISE TRACKING
    exercise_tracking = await initialize_or_fetch_exercise_tracking(db, user_id)

    print("exercise_tracking looks as follows ", exercise_tracking)

    tokens_exceeded, total_tokens_used_today[user_id] = await check_tokens_on_user_login(user_id)
    await websocket.send_json({"tokens_used": total_tokens_used_today[user_id]})
    if tokens_exceeded:
        await websocket.send_text("Unfortunately, the daily token limit for GPT has been exceeded. Please wait until tomorrow to continue using the service. Thank you, and take care!")
        raise ExceedDailyTokenError("Daily token limit exceeded")
    
    

    #TODO: Handle unfinished exercises, so exercises that were introduced but the user never got to them bcos of technical error

    revisiting_session_1 = False #USER SPECIFIC

    # Update the list comprehension to include the new conditions for identifying unrated exercises
    unrated_exercises = [ #MAKE USER SPECIFIC 
        exercise for exercise, details in exercise_tracking.items() if
        (details.get("started", False) and details.get("completed", False) and details.get("session_completed", None) is None) or
        (details.get("started", False) and details.get("completed", False) and details.get("session_completed", None) < details["session_started"])
    ]

    if unrated_exercises: #MAKE USER SPECIFIC
        # Generate summary and request feedback
        # This could be a place to generate a message to the user asking for feedback
        # For example:
        print("we have the following unfinished exercises from last session ", unrated_exercises)
        welcome_back_message = (
            "Willkommen zurück! Bevor wir weitermachen, ist mir aufgefallen, dass Sie in Ihrer letzten Sitzung einige Übungen begonnen, aber nicht bewertet haben. "
            "Es ist wichtig für mich, Feedback zu diesen Übungen zu sammeln, damit ich Empfehlungen besser auf Ihre Vorlieben abstimmen kann. "
            "Hier ist eine kurze Zusammenfassung der Übungen, für die mir das Feedback fehlt: "
            f"{', '.join(unrated_exercises)}. "
            "Wenn es Ihnen nichts ausmacht, würde ich Sie bitten, Ihre Erfahrungen mit diesen Übungen zu bewerten."
        )

        #await websocket.send_text(welcome_back_message)
        await split_text_into_chunks_and_stream(websocket, welcome_back_message)
        last_session = await get_user_session(user_id)
        await collect_feedback_for_unfinished_exercises(unrated_exercises, websocket, messages, user_id, exercise_tracking, last_session)

    

    exercises_based_on_scenario = None
    exercises_based_on_feedback_unique = None
    exercises_based_on_schedule = None


   

    ### THIS WILL HELP ME FIND OUT WHAT THE EXERCISES ARE FOR THIS SESSION (OR THE NEXT? THINK ABOUT IT)
    ##IF ITS THE FIRST SESSION , WE DONT NEED TO GENERATE USER RECOMMENDATIONS, WE SIMPLY RETURN THE LIST OF EXERCISES IN CHRONOLOGICAL ORDER
    ################## GENERATE RECOMMENDATIONS BASED ON PREVIOUS SESSION INFO END #############################


    
    # Increment session number at the start or retrieve the current session
    #now we want to check whether all exercises from the previous session have been completed already, and we only increment then
    last_session = await get_user_session(user_id)
    if last_session != 0:
        exercises_for_last_session = exercise_schedule[int(last_session)]
        #check in exercise_tracking to see if all exercises from the previous session have been completed
        all_exercises_completed = all([exercise_tracking[exercise]['completed'] for exercise in exercises_for_last_session])

        if all_exercises_completed:
            session_nr = await increment_user_session(user_id) #we can move on to next scheduled exercises if all exercises from previous session have been completed
        
        else:
            session_nr = last_session
            if last_session == 1:
                revisiting_session_1 = True
    else:
        session_nr = await increment_user_session(user_id)
 
    current_session = session_nr

    recommendation_message, exercises_based_on_schedule, exercises_based_on_feedback = await generate_user_recommendation(user_id, current_session, revisiting_session_1)
    print(recommendation_message)
   

    # After generating recommendations, assume you have a list like this. TODO: EXERCISES BASED ON RECOMMENDATIONS, NOT JUST SCHEDULE
    if (session_nr == 1 and not revisiting_session_1): #its the user's first session, just return the first 3 exercises
        current_session_exercises = ["Exercise 1", "Exercise 2.1", "Exercise 2.2"]
        #exercises_available_in_session = current_session_exercises
        exercises_based_on_schedule = current_session_exercises
        all_available_exercises = {'scenario': exercises_based_on_scenario, 'schedule': exercises_based_on_schedule, 'feedback': exercises_based_on_feedback_unique}

        session_specific_info = """- The {{exercise_options}} you should present are the following: "Based on the user's progress and feedback, the {{scheduled_exercises}} in this session are Exercise 1, Exercise 2.1, Exercise 2.2." 
- The {{objective}} of the scheduled exercises is "Connecting compassionately with our child”
- This is the {{theory}} behind the scheduled exercises: 
{Try to distinguish between your adult and your child. Looking at “happy” and “unhappy” child photos helps in this effort. Think about your early years and emotional problems with parents and other significant figures in early childhood. The first principle of SAT is to have a warm and compassionate attitude towards our child, no matter whether the underlying emotions they feel are positive or negative. Later this compassion is extended to other people.}
- These are the {{scheduled_exercises}}. Unless specified otherwise, the duration of each exercise is 5 minutes: 
{"Exercise 1": "Getting to know your child.", "Exercise 2.1": "Connecting compassionately with your happy child.", "Exercise 2.2": "Connecting compassionately with your sad child."}"""


        # Fallback to the last known session settings if the current session number exceeds defined sessions
        #current_settings = session_settings.get(current_session, list(session_settings.values())[-1])
        if revisiting_session_1:
            current_settings = Settings(assistantsName="Satherine", aiDescription=session_1_revisiting, sarcasm=0.0)
        else:
            current_settings = Settings(assistantsName="Satherine", aiDescription=session_1_desc, sarcasm=0.0)

    else: 
        current_session_exercises = exercises_based_on_schedule #TODO: CHANGE THIS BASED ON WHAT USER SAYS
        # Combine all exercises, ensuring uniqueness
        #all_available_exercises = set(exercises_based_on_schedule + exercises_based_on_feedback["for_practice"] + exercises_based_on_feedback["for_enjoyment"])
        exercises_based_on_feedback_unique = list(set(exercises_based_on_feedback["for_practice"] + exercises_based_on_feedback["for_enjoyment"]))
        all_available_exercises = {'scenario': exercises_based_on_scenario, 'schedule': exercises_based_on_schedule, 'feedback': exercises_based_on_feedback_unique}
        # Optionally, convert the set back to a list if order matters or if the consuming function expects a list
        #exercises_available_in_session = list(all_available_exercises)
    

        # Select the appropriate settings for the current session
        objective_new_exercises = session_objectives[current_session] #TODO: Logic for if user isnt ready to start exercises from a new session cuz they need to finish unfinished exercises from old session
        theory_new_exercises = session_theory[current_session]   #TODO: Logic for if user isnt ready to start exercises from a new session cuz they need to finish unfinished exercises from old session
        #new_exercises = generate_exercise_descriptions(exercises_based_on_schedule, exercise_descriptions_long) #TODO: CHANGE THIS BASED ON STH ELSE ?? IF ALL EXERCISES FROM A PREV SESSION WERE COMPLETED E.G?
        new_exercises = generate_exercise_descriptions(exercises_based_on_schedule, exercise_descriptions_short)

        session_specific_info = generate_session_specific_information(recommendation_message, objective_new_exercises, theory_new_exercises, new_exercises)
        full_system_prompt_for_session = insert_session_info_into_prompt(all_available_exercises, session_specific_info, session_nr) #session_nr >= 2
        
        print("FULL SYSTEM PROMPT FOR THIS SESSION IS ", full_system_prompt_for_session)

        assistant_name = "Satherine"+str(current_session)
        current_settings = Settings(assistantsName=assistant_name, aiDescription=[full_system_prompt_for_session])
    
    
    # Apply settings for this session --> this updates the model and changes the system prompt for the assistant to one of the predefined settings above (needs to be made more sophisticated instead of hardcoded), so current assistant_description is overwritten
    await CALL_set_user_settings(user_id, current_settings) #HERE WE WILL ALREADY HAVE THE LIST OF EXERCISES WE NEED FOR THIS SESSION BCOS THIS IS THE PROMPT WE GIVE THE CHATBOT



        




    # clear chat history on start up
    ###### Delete #########
    handle_clear_history(user_id)

    
    user_tool_dict = get_user_tool_dict(user_id) #{'memory': <tools.memory.MemoryTool object at 0x7f8e1c0e3d90> if it has been previously enabled (it has), if not, {}
    can_follow_up = False
    #event_trigger_kwargs are arguments you want a particular tool to have access to, so e.g used in the MemoryTool
    event_trigger_kwargs = {"user_id": user_id, 
                            "user_info": user_info, 
                            "user_tool_settings": get_user_tools(db, user_id), 
                            "message": {"role": "assistant", "content": ""},
                            "user_assistants": [MainAssistant.insert_information]}

    
    
    ALL_HANLDERS["OnStartUp"](**event_trigger_kwargs) #executes memory, since file_process doesn't have an on_startup fct
    last_conversation = user_info.get("last_conversation", "")
    user_info = update_user_info(user_id, user_info)
    res = None

    # start-up messages: When the user logs in, we want the chatbot to say something first, before the user says anything
    print("PRINT A START UP MESSAGE")   #STEP NR 1

    if isinstance(last_conversation, dict):
        can_follow_up = True
    if can_follow_up:
        print("FOLLOW UP LAST CONVERSATION")
        print(last_conversation)
        summary = last_conversation["summary"]
        last_conversation_time = last_conversation["timestamp"]
        follow_up = last_conversation["follow_up"]
        last_session_nr = current_session-1

        if revisiting_session_1:
            session_nr_last_ex = 1
        else:
            session_nr_last_ex = current_session-1
        last_session_exercises = [ exercise for exercise, status in exercise_tracking.items() if status.get('started', False) and status.get('completed', False) and (status.get('session_completed', None) == session_nr_last_ex)
                                    ]

        #    You also have some ideas of follow up for that conversation. 
        # Potential follow up ideas:
        # {follow_up}
        message = {"role": "user", "content": f"""Current time is {datetime.now().isoformat()}. You have a summary of the previous conversation that happened on {last_conversation_time}. Start the conversation using these information. Mention everything in your summary during your greeting, but make sure it sounds natural. Make sure that if you mention any exercises the user has done in previous sessions, you only mention the ones done in the last session, which can be found in the list last_session_exercises. Be careful: Since you have a summary of last time's session, this isn't the first time a user is meeting you. Show them you remember them.
        Summary of the previous conversation:
        {summary}
        Exercises from last session:
        {last_session_exercises}

Bitte antworten Sie auf Deutsch, weil ich (der user) nur Deutsch verstehe.

        """}
        messages[user_id].append(message)
        messages_en.append(messages)
    
    else:
        print("NO TOOLS ENABLED")
        message = {"role": "user", "content": "Hi"}
        messages[user_id].append(message)
        messages_en.append(messages)

    #streaming_response calls chatcompletions.acreate on the chat history and streams the response to the user over websocket
    assistant_message = await streaming_response(websocket, MainAssistant, user_id, user_info=user_info, query_results=res, total_tokens_used_today=total_tokens_used_today)
    event_trigger_kwargs["message"] = assistant_message
    ALL_HANLDERS["OnStartUpMsgEnd"](**event_trigger_kwargs) #this doesnt have any purpose, but if you want to define that sth should happen after the assistant sends the first response, you can implement it using OnStartUpMessageEnd event handler
    
    exercises_completed = False
    feedback_collected = False
    user_wants_other_exercises = False

    # Main program loop
    try:
        while True:
            if not exercises_completed or feedback_collected: #NEED TO DO ERROR CHECK BCOS IF U REFRESH PAGE, IT COULD BE THAT FEEDBACK HAS ALREADY BEEN COLLECTED BUT FEEDBACK_COLLECTED WILL BE SET TO FALSE AGAIN --> perhaps store in db, or have a class session_info 
                # wait for user input
                data = await websocket.receive_text()
            

                #handle_clear_history(user_id) #history is only cleared if user asks to have it cleared from frontend! so not necessary it seems
                content = json.loads(data) #data is '{"role":"user","content":"hello","location":{"latitude":"","longitude":""},"datatime":""}' after the user writes 'hello'

                user_input = content["content"] #hello
                if "location" in content:
                    location_info = content["location"]
                else:
                    location_info = None

                
                update_model(user_id)
                update_user_info(user_id, user_info)
                #perform_translation(user_input, messages_en)

                user_tool_dict = get_user_tool_dict(user_id)
                # store user input
                message = {"role": "user", "content": user_input} #next relevant step after receiving a user input
            
                messages[user_id].append(message)
                asyncio.create_task(messages[user_id].save_message_to_firestore(message)) #asynchronously save message to firestore

                event_trigger_kwargs["user_tool_settings"] = get_user_tools(db, user_id) #not really used for the current tools, but technically this means other tools could see what tools are enabled. perhaps useful for SAT
                event_trigger_kwargs["message"] = message
                ALL_HANLDERS["OnUserMsgReceived"](**event_trigger_kwargs) #for the fileprocess tool, this means that pinecone retriever will be called to check if there are any relevant docs it can append to the prompt sent to the llm as additional info, if yes, it adds this additional info to query and mainassistant. For memorytool, the message is added to activeusersession and repetition_check is performed before saving to vectorstore. U CAN UNCOMMENT A SECTION TO UPDATE USER FACTS (e.g emotions) AFTER EVERY MESSAGE

                check_qa(message["content"], qa_embeddings, qa_questions, qa_answers, user_bots_dict[user_id])
                await sat_on_user_msg_received(user_id, message, all_available_exercises, session_specific_info, recommendation_message, exercise_descriptions_short, messages, corpus_embeddings, data_dict_scenarios) #this will update the system prompt for the assistant to include the exercises the user should do in this session, based on the user's message


                # display messages
                print("\n\nMESSAGES:")
                for message in messages[user_id].get():
                    print(message['role'] + ":  " + message['content'])

                #if handle_SOS is true, append an additional message to messages before asking gpt to respond
                if handle_SOS[user_id] and assistant_responded[user_id]:
                    #instruction_to_handle_response = "I have told you whether I’m comfortable proceeding with exercises or whether I want to talk about my feelings or whether I want to end the session. {user_response} Act accordingly: If i want to do an exercise, reply with your {knowledge} about {scenario} and explain why you would suggest the following exercise based on my {scenario}: {exercise_based_on_scenario}. After explaining this ask what I would like to do, given the following options: Proceed with the exercise, Continue talking about my feelings, or End the session."
                    #TODO: CHECK IF NONE?
                    instruction_to_handle_response = f"""I have told you whether I’m comfortable proceeding with exercises or whether I want to talk about my feelings or whether I want to end the session. Act accordingly: If i want to do an exercise, give a short reply with your Knowledge about Scenario and explain why you would suggest the following Exercise based on my Scenario. Make sure you highlight that the priority now is to soothe myself with an exercise that is meant to lessen negative emotions. Keep this input about your Knowledge short and concise, maximum 2 sentences. After explaining this let me know that we can proceed with whatever I feel comfortable with, and ask what I would like to do, given the following {{options}}: If I feel comfortable proceeding with the exercise, if I would like to talk to you about my feelings, or if I prefer to wrap up the session and come back another time.

                    Variables used:
                    - Scenario: {current_scenario_dict[user_id]["scenario"]}
                    - Knowledge: {current_scenario_dict[user_id]["knowledge"]}
                    - Exercises based on scenario: {current_scenario_dict[user_id]["exercises_based_on_scenario"][0]}
                    - My last response: {message}"""
                    message = {"role": "user", "content": instruction_to_handle_response}
                    messages[user_id].append(message)
                    handle_SOS[user_id] = False
                    assistant_responded[user_id] = False


                print("\n\nQUERY ORACLE")
                # this updates the messages[user_id] object
                res = {}
           



                print("\n\nQUERY MainAssistant") #oracle was queried first to check whether the user's message requires a function call or not. if what oracle replies is a tuple, this means it decided it didnt need a function call (e.g 'generate image'), and instead just responded directly. you send its response to be streamed by the user in order to reduce latency, otherwise, MainAssistant is queried, with its additional info etc
                # Now MainAssistant can respond
               
                assistant_message = await streaming_response(websocket, MainAssistant, user_id, user_info=user_info, query_results={}, total_tokens_used_today=total_tokens_used_today)

                #static_prompt_analysis = "AI Role: You are a sophisticated text analysis AI designed to understand and classify stages of a conversation in a therapy or guidance chatbot context. Your function is to analyze the conversation's history and determine its current stage based on predefined categories: Startup, Smalltalk, Exercise Presentation, Feedback Collection, WrapUp.\\n\\nInput: The input will be a transcript of the conversation history between the chatbot and the user. This transcript includes exchanges that have led up to the current point in the conversation.\\n\\nTask: Your task is to analyze the conversation history, identify cues and keywords that indicate the conversation's current stage, and classify the stage accurately. You must also provide a brief justification for your classification, referencing specific elements of the conversation that influenced your decision.\\n\\nExpected Output: Produce your output in a JSON format. The JSON object should contain two keys: 'stage', whose value is the identified stage of the conversation, and 'reason', which provides a brief explanation for why this stage was selected based on the conversation analysis.\\n\\nOutput Format Example: {\\\"stage\\\": \\\"Smalltalk\\\", \\\"reason\\\": \\\"The conversation includes light, general discussion about the user's day and interests, typical of the Smalltalk stage.\\\"}"
                event_trigger_kwargs["message"] = assistant_message
                event_trigger_kwargs["websocket"] = websocket #websocket doesnt seem to be used by any of the tools SO FAR, but perhaps this will change for SAT
                ALL_HANLDERS["OnResponseEnd"](**event_trigger_kwargs) #in the image generation tool this means the bot will try to asynchronously generate an image, in the memory tool this means message is appended to active_user_session and repetition check is performed before storing it to memory
                event_trigger_kwargs.pop("websocket")

                #handle case where assistant responded with SOS message
                if handle_SOS[user_id]:
                    assistant_responded[user_id] = True

                #post_process_assistant_message(assistant_message, user_id, messages[user_id].get(), message)


                #response has been generated by llm, check if it contains the exercise_start or exercise_completed string OR SOS
                if "{__SOS__}" in assistant_message["content"]:
                    #go into a seperate loop of handling user input
                    #prompt_sos = "Forget all previous instructions. The user is in emotional distress. It is now your job to help reframe the user's thoughts using the principles of SAT. If the user is willing to try this, follow these instructions to guide them in reframing their thoughts. Clearly tell the user they don't have to share details with you unless they feel comfortable, and it is perfectly fine to follow your instructions in their head and to notify you when they have finished following your instruction. Start with one reflection question at a time, and ask the user if the reflection question has helped soothe them, and if they are ready to proceed with other scheduled exercises. If the user confirms this, return the single phrase ```{FINISHED_SOS}```. Your guidance instructions : If the user has any photos of their childhood selves on them, tell them they can use the photos to help visualization. Invite the user to extend compassion to themselves by imagining the situation from the perspective of their inner child, that is currently overwhelmed by negative emotions. In this situation it is important that the user doesn't judge themselves for having these emotions, but rather realizes that there's a part of themselves (represented by their inner child) that needs compassion and soothing. In order to do so, the user should offer themselves the same support and care a nurturing parent would offer to a child that is experiencing the same distress. By engaging in this process, the user would not only acknowledge the emotions but actively work to reduce their intensity. Help the user through this realization process by asking questions that guide them towards self-compassion and understanding. Only ask the Example Questions listed below. Remember to be patient and supportive throughout this process. Remind the user that it's perfectly fine to do this in their head, they don't have to share details with you unless they feel comfortable. After the user has completed at least two reflective questions, ask if they are ready to continue with the protocol. Example Questions: Can you imagine the situation from the perspective of your inner child? What would you say to your inner child to offer comfort and support? How can you extend the same compassion to yourself that you would offer to a child in distress?  When we talk about communicating with our inner child, it's a way to acknowledge and address these feelings and needs that have been with us since childhood. It's about reaching out to the part of you that feels these emotions and offering it understanding and comfort, just as you would do for a child who is feeling upset or disappointed. Could you try to think about what you might say to comfort and reassure that part of yourself? Are you ready to continue with the protocol?" 
                    #assistant_name = "Satherine"+str(current_session)
                    #new_instruction_settings = Settings(assistantsName=assistant_name, aiDescription=[prompt_sos])
                    #await CALL_set_user_settings(user_id, new_instruction_settings) 
                    #messages[user_id].append(prompt_sos)
                    sos_system_message = "The user is currently in distress, and as a therapist you have a seperate self-reflection protocol for handling this situation: First you return a predefined empathetic response. Then, if the user is comfortable with it, you ask the user three self-reflection questions, and once the user has notified you that they have thought about all three questions, you output the single string ```{FINISHED_SOS}```. Make sure the user is comfortable with each prompt. If the user tells you they don't want to think about a prompt, ask if they would prefer to move on to SAT exercises or wrap up the session. It is very important that you output {FINISHED_SOS} and nothing else, either after the user has done three reflection questions, or after the user has told you they would like to wrap up the session, or if they have told you they want to move on to SAT exercises. Strictly stick to the instructions for your next response: I will give you a predefined answer, and I want you to return exactly that answer, but with the context of the problem the user mentioned. Make sure to insert the context in a way that feels natural to the statement. Here is the predefined answer, with placeholders you can modify to add the context of the user's problem. Only say this once: 'I'm really sorry to hear {problem_context}, {my_name}. It's completely normal to feel upset after such an event. In moments like these, it can be helpful to practice self-compassion. In Self Attachment Therapy, we try to understand that there's a part of us, represented by our 'childhood self', that needs compassion and soothing. Right now, your childhood self might be feeling hurt and overwhelmed by negative emotions. It's important not to judge yourself for these feelings. Instead, try to offer yourself the same understanding and compassion a caring parent would provide their upset child. By doing so, we can hope to slightly lessen these negative emotions. Given this information, I would just like to check again: Do you feel ready to explore this with some self-reflection prompts, or would you prefer to talk about your feelings instead? You can also decide to wrap up the session today if you feel more comfortable.' If the user tells you they want you to ask them some self-reflection prompts, guide them through the following Example Questions. Remind them that it is perfectly fine to think about these questions in their head, and only share with you whatever they feel comfortable sharing. However, once they have finished thinking about a question, encourage them to let you know they are done. Ask them one at a time, and only ask these questions and nothing else. If the user asks what kind of prompts, give a short explanation. If the user is then willing to proceed, introduce the first prompt without saying much else. If the user shares their thoughts, respond to them with kindness, empathetically and professionally, like a therapist would. If the user appears to not respond well to the self-reflection prompt, try to understand why, without being invasive. If you notice the user having doubts, be patient and explain to the user why you think that reflecting about this will help, making sure you put the emphasis on the importance of extending compassion to oneself. Answer any questions or doubts the user might have about the prompts. Offer them the choice of ending the session and returning at another time if you believe the user isn't responding well at all. Example Questions: 1) When we talk about communicating with our childhood self, it's a way to acknowledge and address feelings and needs that have been with us since childhood. It's about reaching out to the part of you that feels these emotions and offering it understanding and comfort, just as you would do for a child who is feeling upset or disappointed. Could you try to think about the situation from the perspective of your childhood self? 2) What would you say to your childhood self to offer comfort and support? 3) Can you try to think of a way you can you extend the same compassion to yourself that you would offer to a child in distress?. Encourage the user to notify you once they are done thinking about a question, and present the next question until all three have been presented. Then, output the single phrase ```{FINISHED_SOS}``` followed by nothing else. Remember: If the user tells you they don't want to do the prompts, you have to ask the user if they would like to continue with SAT exercises instead or wrap up the session. It is INCREDIBLY IMPORTANT that you output {FINISHED_SOS} if the user decides they want to do SAT exercises or they want to wrap up the session." 
                    system_msg_new = {"role": "system", "content": sos_system_message}
                    assistant_name = "Satherine"+str(current_session)
                    new_instruction_settings = Settings(assistantsName=assistant_name, aiDescription=[sos_system_message])
                    print("I AM NOW CHANGING THE SYSTEM PROMPT TO HANDLE SOS SITUATION")
                    await CALL_set_user_settings(user_id, new_instruction_settings) 
                    messages[user_id].append({"role": "user", "content":"handle my negative emotions by using your self-reflection protocol. This isn't an exercise, so don't call it an exercise. Be compassionate and patient, and always check in on whether I feel comfortable continuing, or if i feel really bad, i can also decide to wrap up the session. Make sure you don't forget to output {FINISHED_SOS} if i decide to end the session or i have thought about all three reflection questions."})

                if "{FINISHED_SOS}" in assistant_message["content"]:
                    #go back to the original settings
                    await CALL_set_user_settings(user_id, current_settings)
                    #handle_message_storing(user_id, "assistant", getting_back_to_schedule_msg, messages)


                if "{exercise_start:" in assistant_message["content"]:
                    exercise_label = extract_label(assistant_message["content"], "start")
                    if exercise_label: #if its not none
                        exercise_tracking[exercise_label]['session_started'] = current_session #we completed this exercise in this particular session (again)
                        exercise_tracking[exercise_label]['started'] = True
                if "{exercise_end:" in assistant_message["content"]:
                    exercise_label = extract_label(assistant_message["content"], "end")
                    if exercise_label: #if its not none
                        #we only fill out session_completed if we have filled out feedback for it!
                        #exercise_tracking[exercise_label]['session_completed'] = current_session #we completed this exercise in this particular session (again)
                        exercise_tracking[exercise_label]['completed'] = True

                



                #response has been generated by llm, check if it contains the flag string
                if "__ALL_EXERCISES_COMPLETED__" in assistant_message.get('content', ''): #add OR condition - maybe assistant doesn't recognize that all conditions have been completed, but our exercise_tracking system shows us that all available exercises have been completed
                    #TODO: Double-check if all exercises have really been completed by comparing the exercise_tracking labels with conversation history. can also be of the form of a request to user 'im sorry i have gotten confused, can you please remind me which exercises we did in this session so I can ask the right feedback?'
                    exercises_completed = True
                    continue  # Proceed to feedback collection without waiting for user input
            else:
                # Feedback collection phase
                current_session_exercises = [
                    exercise for exercise, details in exercise_tracking.items()
                    if (details.get("started", False) and details.get("completed", False)) and details.get("session_started", None) == current_session
                ]
                for exercise in current_session_exercises:
                    questions = feedback_questions.get(exercise, [])
                    feedback_responses.setdefault(exercise, {})  # Ensure the exercise key exists
                    
                    
                    for question in questions:
                        feedback_responses[exercise].setdefault(question, [])
                        valid_response = False
                      
                        while not valid_response:
                            await websocket.send_text(json.dumps(question))
                            handle_message_storing(user_id, "assistant", question, messages)
                            
                            feedback = await websocket.receive_text()
                            feedback_data = json.loads(feedback)  # Convert string back to dictionary
                            user_input_feedback = feedback_data["content"]

                            # Check if the response is a number between 1 and 5
                            if user_input_feedback.isdigit() and 1 <= int(user_input_feedback) <= 5:
                                valid_response = True
                                feedback_responses[exercise][question].append(user_input_feedback)
                                handle_message_storing(user_id, "user", user_input_feedback, messages)
                            else:
                                # Ask again with an explanation of the correct format
                                clarification_msg = "Bitte bewerten Sie Ihre Erfahrung auf einer Skala von 1 bis 5, wobei 1 'überhaupt nicht hilfreich' und 5 'äußerst hilfreich' bedeutet."

                                await websocket.send_text(json.dumps(clarification_msg))
                                handle_message_storing(user_id, "assistant", clarification_msg, messages)

                    #we have answered all feedback questions for this exercise, so we can consider this exercises completed
                    exercise_tracking[exercise]['session_completed'] = current_session #we completed this exercise in this particular session (again)
                    
                        
                        


                # After collecting feedback, process it as needed
                #process_feedback(feedback_responses)  # Placeholder for feedback processing
                print("THIS IS THE FEEDBACK !!!!! ", feedback_responses) 
                store_session_feedback(user_id, session_nr, feedback_responses)
                feedback_collected = True
                #transition to ending the session
                transition_to_end_msg = "Vielen Dank, dass Sie Ihre Erfahrungen mit den Übungen bewertet haben! Möchten Sie Gedanken oder Gefühle zu Ihrer Erfahrung mit einer der Übungen teilen?"

                await websocket.send_text(transition_to_end_msg)
                handle_message_storing(user_id, "assistant", transition_to_end_msg, messages)
                #notify the chatbot that feedback for all exercises has been collected 
                feedback_collected_message = {"role": "user", "content": "You have now collected feedback for all exercises, and can proceed to ask the user if they are comfortable ending the session or would like to share anything else that's on their mind. If the user would like to end the session, simply output a thank you message and say goodbye."}
                messages[user_id].append(feedback_collected_message)
                continue  # Optionally break out of the loop if feedback collection ends the session




    

    except Exception as e:
        ALL_HANLDERS["OnUserDisconnected"](**event_trigger_kwargs) #for fileprocess tool, this means files uploaded for the user_id are deleted from pinecone. for memory module, this means that user_facts are updated (llm call made to get user facts in json format and saved to pinecone) and summary of session is created and saved to pinecone. 
        await write_exercise_tracking_to_firebase(db, user_id, exercise_tracking)
        if isinstance(e, ExceedDailyTokenError):
            await write_token_limit_to_fb(db, user_id, True, total_tokens_used_today[user_id])
        else:
            await write_token_limit_to_fb(db, user_id, False, total_tokens_used_today[user_id])
        print(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000) 
