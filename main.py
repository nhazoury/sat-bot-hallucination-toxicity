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

key_filepath = os.path.join(os.path.dirname(__file__), "firebase_key.json")
cred = credentials.Certificate(json.load(open(key_filepath)))

fb_app = firebase_admin.initialize_app(cred, name=__name__)
db = firestore.client(fb_app)
messages: Dict[str, Messages] = {}

### changes since deployment ###
handle_SOS: Dict[str, bool] = {}
assistant_responded: Dict[str, bool] = {}
current_scenario_dict= {}
sent_response = {}
total_tokens_used_today : Dict[str, int] = {}

def download_nltk_resources():
    try:
        _ = nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()




session_1_desc = [""" You are an advanced virtual assistant named Satherine, specialized in a therapy called Self Attachment Technique (SAT). This therapy consists of 21 structured protocols. Your role is to support and guide users through their therapeutic journey, utilizing cognitive behavioral therapy (CBT) techniques such as Positive Affirmation, Paraphrasing, and Reflection to respond. Your goal is to validate the user's emotions empathetically and create a safe and supportive space for expression.

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
            "Well done completing that exercise. Would you like to try another, take a moment to discuss how you're feeling, or end the session for today? Please let me know your preference.",
            "Good work! If you feel ready, we can move on to another exercise. Alternatively, we can talk about any feelings that have come up, or we can conclude the session here. What would you prefer?",
            "Very well done! Would you like to proceed with another, or would you prefer to discuss your current feelings? We can also end the session if you feel it’s time. Please share your thoughts.",
            "Nicely done. Are you comfortable continuing with another exercise, or would you like to pause and reflect on your feelings? It's also perfectly fine if you prefer to end the session now. What’s your decision?",
            "You handled that exercise well. What would you like to do next? We can continue with another exercise, discuss how you are feeling, or stop here for today. Please let me know how you wish to proceed.",
            "Nicely done. Are you ready to continue with another, or do you wish to take some time to talk about any feelings that have surfaced? If you'd prefer to conclude for today, just let me know.",
            "Very well done! How would you like to proceed? We can start another exercise, discuss any emotions you're experiencing, or wrap up the session if that's your preference. I'm here to support your choice.",
            "Excellent work. Do you feel up to beginning another exercise, or would you rather discuss your experiences so far? If you think it's best to finish the session now, that’s also an option. Please tell me what you'd like to do next.",
            "Well done! You're doing great. Are you comfortable trying another exercise, or would you like to share how you are feeling now? You can also let me know if you would like to wrap up the session and come back another time. What do you prefer?"
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
                    await websocket.send_text("0-Sorry, the service that I rely on from OpenAI is currently down. Please try again.")
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
                    sos_text = "I'm really sorry to hear you're feeling this way. When dealing with negative emotions like these, it's important to try to extend compassion to yourself. Would you feel comfortable with me guiding you through some self-reflection prompts to practice this, or would you prefer to talk about your feelings instead? You can also always decide to come back to this session another time. It's important to follow your instincts and do only what you're comfortable with."
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
                   transition_to_feedback_msg = "Thank you for all your efforts today! If you don't mind, I'd like you to take a minute to rate your experience with the exercises, so that I can tailor future sessions to your preferences."
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
                    getting_back_to_schedule_msg = "Thank you for taking the time to offer yourself compassion. How are you feeling now? I would just like to take a minute to clarify your options now, so that we proceed with what you are most comfortable with: If you would like, we could return to the SAT protocol and look at some exercises that are next in schedule. However, if you prefer to talk about your feelings or would feel more comfortable ending the session here, I am here to support you. How would you like to proceed?" 
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
            sos_text = "I'm really sorry to hear you're feeling this way. When dealing with negative emotions like these, it's important to try to extend compassion to yourself. Would you feel comfortable with me guiding you through some self-reflection prompts to practice this, or would you prefer to talk about your feelings instead? You can also always decide to come back to this session another time. It's important to follow your instincts and do only what you're comfortable with."
            #await websocket.send_text(sos_text)
            await split_text_into_chunks_and_stream(websocket, sos_text)
            full_response = "{__SOS__}" + sos_text
        elif "{__SOS__}" in content and sent_response[user_id]:
            #response has already been sent, so we don't stream SOS text but just append it to storage. will be handled by the check after response from assistant has been streamed
            content = ""
            full_response = "{__SOS__} " + full_response

        elif "__ALL_EXERCISES_COMPLETED__" in content:
            transition_to_feedback_msg = "Thank you for all your efforts today! If you don't mind, I'd like you to take a minute to rate your experience with the exercises, so that I can tailor future sessions to your preferences."
            # Inform the user about transitioning to feedback collection
            await websocket.send_text(transition_to_feedback_msg)
            full_response = "__ALL_EXERCISES_COMPLETED__" + transition_to_feedback_msg

        elif "{FINISHED_SOS}" in content:
            getting_back_to_schedule_msg = "Thank you for taking the time to offer yourself compassion. How are you feeling now? If you would like, we could return to the SAT protocol and look at some exercises that are next in schedule. However, if you prefer to talk about your feelings or would feel more comfortable ending the session here, I am here to support you. How would you like to proceed?" 
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

    if total_tokens_used_today[user_id] > 90000:
        await websocket.send_text("Unfortunately, the daily token limit for GPT has been exceeded. Please wait until tomorrow to continue using the service. Thank you, and take care!")
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
    1: "Connecting compassionately with our child",
    2: "Forming a passionate loving relationship with the child",
    3: "Forming a passionate loving relationship with the child",
    4: "Joyful love for our child",
    5: "Reprocessing painful childhood events",
    6: "Reprocessing painful childhood events",
    7: "Re-learning how to laugh",
    8: "Transforming our perspective",
    9: "Transforming our perspective",
    10: "Transforming our perspective",
    11: "Transforming our perspective",
    12: "Transforming our perspective",
    13: "Socializing your child",
    14: "Your Child: The source of your creativity",
}

session_theory = {
    1: "Try to distinguish between your adult and your child. Looking at “happy” and “unhappy” child photos helps in this effort. Think about your early years and emotional problems with parents and other significant figures in early childhood. The first principle of SAT is to have a warm and compassionate attitude towards our child, no matter whether the underlying emotions they feel are positive or negative. Later this compassion is extended to other people.",
    2: "This session, our goal is to cultivate a deep, loving bond with your inner child. It's about establishing a connection that mirrors the nurturing love a parent feels for their child. This involves embracing self-love and expressing affection through gestures like eye contact, holding hands, gentle touches, singing, playing, and laughing. In doing so, we're aiming to trigger the release of dopamine in your brain, akin to the feelings experienced in maternal, romantic, or spiritual love. This not only fosters hope and motivation but also prepares you for an active role in caring for your inner child. Part of this process involves speaking out loud to your inner child, enhancing emotional bonding and maturity. This practice, though often overlooked in adulthood, is crucial for emotional and cognitive development. It represents emotional and cognitive maturity, not a departure from it. Remember, caring for a child is a fundamental human capacity, and these exercises are designed to awaken and reinforce this innate ability within you.",
    3: "We will continue our goal to cultivate a deep, loving bond with your inner child. It's about establishing a connection that mirrors the nurturing love a parent feels for their child. This involves embracing self-love and expressing affection through gestures like eye contact, holding hands, gentle touches, singing, playing, and laughing. In doing so, we're aiming to trigger the release of dopamine in your brain, akin to the feelings experienced in maternal, romantic, or spiritual love. This not only fosters hope and motivation but also prepares you for an active role in caring for your inner child. Part of this process involves speaking out loud to your inner child, enhancing emotional bonding and maturity. This practice, though often overlooked in adulthood, is crucial for emotional and cognitive development. It represents emotional and cognitive maturity, not a departure from it. Remember, caring for a child is a fundamental human capacity, and these exercises are designed to awaken and reinforce this innate ability within you.",
    4: "This session, we focus on the artistic re-creation of our emotional world, an important aspect of our journey towards self-healing and understanding. \n One key element of this process is the concept of house building. Historically, building a house has been a fundamental human activity, providing not only physical protection but also a sense of safety and security. Reflect back to childhood, where building a house or a fort was one of the earliest and most enjoyable games. This activity gave us a feeling of comfort and security, a safe haven of our own making. Now, as we embark on this therapy, think of building a dream house as a metaphor for constructing a new self. It's about creating a space that feels safe, secure, and uniquely ours, where our emotional self can thrive. \n Additionally, we'll explore our bond with nature. Our early connections to nature play a crucial role in our physical and mental health, and they shape our attitudes towards the environment as we grow. Reconnecting with nature now can rekindle those early feelings of attachment, bringing a sense of peace and balance to our lives.",
    5: "This session, we take a significant step: your Adult self will actively work to console and support your Child self, especially in dealing with current or past pains. \n We've been building skills to enhance positive emotions, and now we shift our focus to processing and reducing negative emotions like anger, fear, sadness, and anxiety. These emotions might arise from various aspects of your life, including relationships with partners, family, friends, work, or societal interactions. \n The key here is to project these negative emotions onto your Child self. This allows your Adult self to step in, address, and alleviate these pains, offering the same support and care a nurturing parent would. By engaging in this process, you're not only acknowledging the emotions but actively working to reduce their intensity. \n An essential part of this week's exercises involves self-reassurance and self-massage, which are powerful tools for emotional regulation. These practices aid in releasing oxytocin and vasopressin, hormones that play a crucial role in mitigating feelings of distress and discomfort. \n Moreover, we'll begin reprocessing painful childhood events. Childhood traumas often lead to persistent emotional and behavioral challenges. By starting with less severe cases, you can gradually work your way up to more challenging traumas. It's crucial to only move to more severe traumas when you feel confident and ready. If you're unsure, it's perfectly okay to continue with the exercises from previous weeks and focus on less traumatic experiences until you feel stronger.",
    6: "We continue our journey of teaching your Adult self how to actively work to console and support your Child self, especially in dealing with current or past pains. \n We've been building skills to enhance positive emotions, and now we shift our focus to processing and reducing negative emotions like anger, fear, sadness, and anxiety. These emotions might arise from various aspects of your life, including relationships with partners, family, friends, work, or societal interactions. \n The key here is to project these negative emotions onto your Child self. This allows your Adult self to step in, address, and alleviate these pains, offering the same support and care a nurturing parent would. By engaging in this process, you're not only acknowledging the emotions but actively working to reduce their intensity. \n An essential part of this week's exercises involves self-reassurance and self-massage, which are powerful tools for emotional regulation. These practices aid in releasing oxytocin and vasopressin, hormones that play a crucial role in mitigating feelings of distress and discomfort. \n Moreover, we'll begin reprocessing painful childhood events. Childhood traumas often lead to persistent emotional and behavioral challenges. By starting with less severe cases, you can gradually work your way up to more challenging traumas. It's crucial to only move to more severe traumas when you feel confident and ready. If you're unsure, it's perfectly okay to continue with the exercises from previous weeks and focus on less traumatic experiences until you feel stronger.",
    7: "This session, we focus on the importance of laughter in emotional healing. Laughter, influenced by family and cultural norms, is a natural response that securely attached children often exhibit. While negative reactions to upsets are common, learning to see situations in a different light and finding humor in them is beneficial. We'll encourage your inner child to laugh, fostering a playful attitude and a positive perspective on challenges. Theories of laughter include the Superiority Theory, Incongruity Theory, Relief Theory, Perspective Theory, and Evolutionary Theory. Laughter is a powerful tool for emotion self-regulation, combating anxiety and depression by triggering dopamine and serotonin. It's crucial to maintain a non-hostile attitude in laughter. This week, we'll focus on re-learning to laugh, loosening negative patterns that limit emotional development and the ability to laugh freely, aiding your journey towards emotional freedom.",
    8: "This session, we focus on developing the skill to laugh off our upsets and setbacks, transforming our perspective on negative emotions. We introduce the concept of the Gestalt vase, a powerful tool for changing perspectives. Imagine a vase that, when viewed differently, reveals two faces. This symbolizes shifting our attention away from negative emotions (the black vase) to a more positive perspective (the white faces). It's about seeing beyond the immediate emotional response to a broader, more optimistic view. We explore three contexts for compassionate humor, drawing from the Gestalt vase analogy: 1. World Incongruities: Recognizing the discrepancies between what is said and what is done in the world, particularly in socio-economic and political systems. We focus on finding humor in these incongruities, enhancing positive emotions and creative thinking. 2. Self/World Incongruities: Dealing with violations of our expectations from life or others. We learn to quickly find humor in the mismatch between our expectations and reality, reducing negative emotions and fostering creativity. 3. Incongruous Self: Finding humor in our own life’s inconsistencies and contradictions. This helps in being playful about our emotions and thoughts, promoting emotional flexibility and wisdom. We also revisit Chaplin's insight: 'Life is a tragedy when seen in close-up, but a comedy in long-shot.' Laughing at our past upsets, recognizing their role in our growth, can be an empowering experience. This week’s exercises will encourage you to view your past challenges as opportunities for laughter and growth, using them as stepping stones towards a more resilient and joyous self. Remember, the ability to laugh at upsets is a form of emotional strength and adaptability. ",
    9: "We continue to focus on developing the skill to laugh off our upsets and setbacks, transforming our perspective on negative emotions. We introduce the concept of the Gestalt vase, a powerful tool for changing perspectives. Imagine a vase that, when viewed differently, reveals two faces. This symbolizes shifting our attention away from negative emotions (the black vase) to a more positive perspective (the white faces). It's about seeing beyond the immediate emotional response to a broader, more optimistic view. We explore three contexts for compassionate humor, drawing from the Gestalt vase analogy: 1. World Incongruities: Recognizing the discrepancies between what is said and what is done in the world, particularly in socio-economic and political systems. We focus on finding humor in these incongruities, enhancing positive emotions and creative thinking. 2. Self/World Incongruities: Dealing with violations of our expectations from life or others. We learn to quickly find humor in the mismatch between our expectations and reality, reducing negative emotions and fostering creativity. 3. Incongruous Self: Finding humor in our own life’s inconsistencies and contradictions. This helps in being playful about our emotions and thoughts, promoting emotional flexibility and wisdom. We also revisit Chaplin's insight: 'Life is a tragedy when seen in close-up, but a comedy in long-shot.' Laughing at our past upsets, recognizing their role in our growth, can be an empowering experience. This week’s exercises will encourage you to view your past challenges as opportunities for laughter and growth, using them as stepping stones towards a more resilient and joyous self. Remember, the ability to laugh at upsets is a form of emotional strength and adaptability.",
    10: "We continue our journey of harnessing humor to reshape our view of life's challenges. Building on the Gestalt vase concept, we aim to shift our focus from negative emotions to a more positive, humorous outlook. This session revisits the power of seeing beyond immediate setbacks, encouraging a perspective where life's incongruities and our own contradictions become sources of laughter and growth. As we delve back into the contexts of compassionate humor, remember Chaplin's insight: life's trials, when viewed from a distance, can transform into a comedy. This session, let's reinforce our ability to find humor in the unexpected, using past challenges as opportunities to cultivate resilience and a more joyous self. Laughing at our upsets isn't just about lightening the moment; it's about building emotional strength and adaptability for the journey ahead.",
    11: "We continue our journey of harnessing humor to reshape our view of life's challenges. Building on the Gestalt vase concept, we aim to shift our focus from negative emotions to a more positive, humorous outlook. This session revisits the power of seeing beyond immediate setbacks, encouraging a perspective where life's incongruities and our own contradictions become sources of laughter and growth. As we delve back into the contexts of compassionate humor, remember Chaplin's insight: life's trials, when viewed from a distance, can transform into a comedy. This session, let's reinforce our ability to find humor in the unexpected, using past challenges as opportunities to cultivate resilience and a more joyous self. Laughing at our upsets isn't just about lightening the moment; it's about building emotional strength and adaptability for the journey ahead.",
    12: "We continue our journey of harnessing humor to reshape our view of life's challenges. Building on the Gestalt vase concept, we aim to shift our focus from negative emotions to a more positive, humorous outlook. This session revisits the power of seeing beyond immediate setbacks, encouraging a perspective where life's incongruities and our own contradictions become sources of laughter and growth. As we delve back into the contexts of compassionate humor, remember Chaplin's insight: life's trials, when viewed from a distance, can transform into a comedy. This session, let's reinforce our ability to find humor in the unexpected, using past challenges as opportunities to cultivate resilience and a more joyous self. Laughing at our upsets isn't just about lightening the moment; it's about building emotional strength and adaptability for the journey ahead.",
    13: "This session, we focus on guiding your Child away from anti-social behaviors and towards identifying a compassionate figure as an ideal role model. We recognize that in recent decades, a rise in narcissism in Western societies has been fueled by several economic, social, and cultural factors. These include the highly competitive nature of corporate economies and education systems, the media's glamorization of individuality and personal success, and the significant impact of the internet and social media in promoting individualism. Such factors have normalized narcissistic traits, which can be detrimental to mental health. In response, our Adult self needs to play a key role in socializing our childhood self. This involves becoming aware of any narcissistic tendencies in our Child, such as envy, jealousy, greed, or mistrust, and understanding how acting out these negative emotions can be counterproductive to our goals. To counter these tendencies, the aim is to cultivate the opposite feelings. This not only helps in managing these negative emotions but also opens up avenues for creative solutions. SAT emphasizes nurturing the opposite pole of these traits to create a balanced and whole individual. The motto here is that mastering psychopathology involves developing the other poles of these traits. By doing so, we strive towards a more balanced, compassionate, and empathetic self. This week's exercises are designed to help you identify and nurture these opposite traits, guiding you towards emotional balance and creative thinking.",
    14: "In our final session of Self Attachment Therapy, we'll delve into the relationship between your inner Child and your creativity, an essential aspect of discovering your true self. Children who develop secure attachments early in life tend to have a strong capacity for independent thought and reflection. This fosters their ability to discover their true selves, which is the very source of their creativity, independence, and spontaneity. In contrast, a child influenced heavily by their environment might develop what's called a 'false self', constrained by external pressures and lacking genuine spontaneity. At around three or four years old, children begin to show remarkable imaginative skills and the ability to make connections between different phenomena – the roots of creativity. A child’s natural spontaneity helps them to understand events without any rigid preconceptions. However, external pressures can sometimes replace this spontaneity with rigidity and uniformity. Our goal in SAT is to rekindle this secure attachment between the Child and the Adult, reigniting your creative spark. Additionally, creative individuals often exhibit paradoxical traits – they are energetic yet calm, smart yet naïve, disciplined yet playful, and so on. Embracing these paradoxes is key to nurturing your creativity. This week, we also focus on the importance of role models, parables, and inspirational quotes in emotional growth. Affirmations from your chosen compassionate role model can strengthen your willpower and perseverance. They encourage you to embrace challenges and setbacks as part of your journey towards achieving your goals. Remember the words of Nietzsche, 'What does not kill me makes me stronger.' This perspective encourages embracing life's challenges with love and strength, seeing them as opportunities for growth. As we conclude our therapy, it's important to realize that creativity isn't just about artistic expression; it's about how you approach life, solve problems, and perceive the world around you. Your inner Child is a wellspring of creativity, and nurturing this aspect of yourself is crucial for a fulfilled and balanced life.",

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
    "Exercise 1": "Getting to know your child.",
    "Exercise 2.1": "Connecting compassionately with your happy child.",
    "Exercise 2.2": "Connecting compassionately with your sad child.",
    "Exercise 3": "Singing a song of affection.",
    "Exercise 4": "Expressing love and care for the child.",
    "Exercise 5": "Vowing to care for the child.",
    "Exercise 6": "Restoring our emotional world after our pledge.",
    "Exercise 7.1": "Maintaining a loving relationship with the child.",
    "Exercise 7.2": "Creating a zest for life.",
    "Exercise 8": "Enjoying nature.",
    "Exercise 9": "Overcoming current negative emotions.",
    "Exercise 10": "Overcoming past pain.",
    "Exercise 11": "Muscle relaxation and playful face for intentional laughing.",
    "Exercise 12": "Victory laughter on our own.",
    "Exercise 13": "Laughing with our childhood self.",
    "Exercise 14": "Intentional laughter.",
    "Exercise 15.1": "Learning to change your perspective.",
    "Exercise 15.2": "Incongruous world.",
    "Exercise 15.3": "Self-world incongruity.",
    "Exercise 15.4": "Incongruous self.",
    "Exercise 16": "Learning to be playful about your past pains.",
    "Exercise 17": "Identifying patterns of acting out personal resentments.",
    "Exercise 18": "Planning more constructive actions.",
    "Exercise 19": "Finding and bonding with your compassionate role model.",
    "Exercise 20": "Updating our rigid beliefs to enhance creativity.",
    "Exercise 21": "Practicing affirmations.",
  
}   

feedback_questions = {
    "Exercise 1": ["On a scale from 1 to 5, how easy did you find Exercise 1 (Getting to Know Your Child)?",
                   "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 2.1": ["On a scale from 1 to 5, how easy did you find Exercise 2.1 (Connecting compassionately with your happy child)?",
                   "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 2.2": ["On a scale from 1 to 5, how easy did you find Exercise 2.2 (Connecting compassionately with your sad child)?",
                   "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 3": ["On a scale from 1 to 5, how easy did you find Exercise 3 (Singing a song of affection)?",
                   "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?",
                   "Did you feel a stronger connection to your child after this exercise than after connecting with it using sad and happy pictures?"],
    "Exercise 4": ["On a scale from 1 to 5, how easy did you find Exercise 4 (Expressing love and care for the child)?",
                   "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 5": ["On a scale from 1 to 5, how easy did you find Exercise 5 (Vowing to care for the child)?",
                     "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 6": ["On a scale from 1 to 5, how easy did you find Exercise 6 (Restoring our emotional world after our pledge)?",
                     "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 7.1": ["On a scale from 1 to 5, how easy did you find Exercise 7.1 (Maintaining a loving relationship with the child)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 7.2": ["On a scale from 1 to 5, how easy did you find Exercise 7.2 (Creating a zest for life)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 8": ["On a scale from 1 to 5, how easy did you find Exercise 8 (Enjoying nature)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 9": ["On a scale from 1 to 5, how easy did you find Exercise 9 (Overcoming current negative emotions)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 10": ["On a scale from 1 to 5, how easy did you find Exercise 10 (Overcoming past pain)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 11": ["On a scale from 1 to 5, how easy did you find Exercise 11 (Muscle relaxation and playful face for intentional laughing)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 12": ["On a scale from 1 to 5, how easy did you find Exercise 12 (Victory laughter on our own)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 13": ["On a scale from 1 to 5, how easy did you find Exercise 13 (Laughing with our childhood self)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 14": ["On a scale from 1 to 5, how easy did you find Exercise 14 (Intentional laughter)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 15.1": ["On a scale from 1 to 5, how easy did you find Exercise 15.1 (Learning to change your perspective)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 15.2": ["On a scale from 1 to 5, how easy did you find Exercise 15.2 (Incongruous world)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 15.3": ["On a scale from 1 to 5, how easy did you find Exercise 15.3 (Self-world incongruity)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 15.4": ["On a scale from 1 to 5, how easy did you find Exercise 15.4 (Incongruous self)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 16": ["On a scale from 1 to 5, how easy did you find Exercise 16 (Learning to be playful about your past pains)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 17": ["On a scale from 1 to 5, how easy did you find Exercise 17 (Identifying patterns of acting out personal resentments)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 18": ["On a scale from 1 to 5, how easy did you find Exercise 18 (Planning more constructive actions)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 19": ["On a scale from 1 to 5, how easy did you find Exercise 19 (Finding and bonding with your compassionate role model)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 20": ["On a scale from 1 to 5, how easy did you find Exercise 20 (Updating our rigid beliefs to enhance creativity)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],
    "Exercise 21": ["On a scale from 1 to 5, how easy did you find Exercise 21 (Practicing affirmations)?",
                        "On a scale from 1 to 5, how useful did you find the exercise and would like to revisit it in the future?"],

}


exercise_descriptions_long = {
    "Exercise 1": "**[Exercise 1] 'Getting to know your child'**: In a quiet place, look at your happy and unhappy photos while recalling positive and negative childhood memories and early relationships in the family.",
    "Exercise 2.1": "**[Exercise 2.1] 'Connecting compassionately with your happy child'**: i) With your eyes closed, first imagine your child from the happy photo, imagining that the child is near you; (ii) then imagine you are embracing the child (iii) and later imagine you are playing with the child, e.g. a game that you played as a child; (iv) Finally, imagine you are dancing with the child. Reflect on how you feel in each phase from (i) to (iv).",
    "Exercise 2.2": "**[Exercise 2.2] 'Connecting compassionately with your sad child':** (i) With your eyes closed, imagine your child from the photo in which it looks unhappy, imagining the child is near you; (ii) then imagine you are embracing and consoling the child; (iii) Open your eyes and stare at your child in the unhappy picture, imagine you are reassuring and comforting your child which makes the child happy and eventually dance. Reflect on how you feel in each phase from (i) to (iii).",
    "Exercise 3":"[Exercise 3] 'Singing a song of affection': Print copies of a happy photo to display at home, work, and in your wallet. You can also set it as the background on your phone and laptop. Then, select a jolly lyrical song you cherish that invokes feelings of warmth, affection, and love. Learn the song by heart and sing it as often as you can in your daily routine. While looking at the happy photo, sing the song, as a way to establish a deep emotional bond with the child in your mind. Start quietly; then, over time, allow your voice to become louder while using more of your body (e.g. shaking your shoulders, hands, and lifting your eyebrows up and down). Imagine that in this way, like a parent, you are joyfully dancing and playing with the child.",
    "Exercise 4":"**[Exercise 4] 'Expressing love and care for the child'**: While genuinely smiling at the happy photo, loudly say to your child: \n ****”I passionately love you and deeply care for you”. Repeat this for five to ten minutes.",
    "Exercise 5":"**[Exercise 5] 'Vowing to care for the child':** In this exercise, we start to care for the child as our own child. We attribute and project our own emotions to the child. We, as our adult self, begin with a pledge we make at a special time and place. After reading the pledge silently, we confidently pledge out loud the following: \n “From now on, I will seek to act as a devoted and loving parent to this child, consistently and wholeheartedly care for them in every way possible. I will do everything I can to support the health and emotional growth of this child.” Repeat this for ten minutes.",
    "Exercise 6":"[Exercise 6] 'Restoring our emotional world after our pledge.': Through imagination or by drawing, consider your emotional world as a home with some derelict parts that you will fully renovate. The new home is intended to provide a safe haven at times of distress for the child and a safe base for the child to tackle life’s challenges. The new home and its garden is bright and sunny; we imagine carrying out these self-attachment exercises in this environment. The unrestored basement of the new house is the remnant of Fthe derelict house and contains our negative emotions. \n When suffering negative emotions, imagine that the child is trapped in the basement but can gradually learn to open the door of the basement, walk out and enter the bright rooms, reuniting with the adult.",
    "Exercise 7.1":"[Exercise 7.1] 'Maintaining a loving relationship with the child.': Choose some short phrase, e.g., “You are my beautiful child” or “My love”. Say it slowly, out loud at least 5 times as you look at the happy photo/avatar. Then sing your favourite chosen love song at least 5 times. As previously, increase your volume and begin to use your whole body.",
    "Exercise 7.2":"[Exercise 7.2] 'Creating a zest for life.': While looking in a mirror, imagine your image to be that of the child, then begin to loudly sing your previously chosen song. As previously, increase your volume and begin to use your whole body. Do this twice now and then as many times as possible in different circumstances during the day, such as while on the way to work or while cooking dinner, to integrate them into your new life. When singing your favourite song becomes a habit of yours, it becomes an effective tool for enhancing positive affects and managing emotions.",
    "Exercise 8":"[Exercise 8] 'Enjoying nature.': Creating an attachment to nature for your child is an effective way to increase joy and reduce negative emotions. On one day this week, go to a local park, wood or forest. Spend at least 5 minutes admiring a tree, attempting to appreciate its real beauty as you have never previously experienced. Repeat this process, including with other aspects of nature (e.g. sky, stars, plants, birds, rivers, sea, your favourite animal), until you feel you have developed an attachment to nature that helps regulate your emotions. Achieving this will help you want to spend more time in nature after this course ends.",
    "Exercise 9":"[Exercise 9] 'Overcoming current negative emotions.': With closed eyes, imagine the unhappy photo and project your negative emotions to the unhappy photo representing the Child. While doing this:(i) loudly reassure the Child, and (ii) give your face/neck/head a self-massage. Repeat these steps until you are calmed and comforted.",
    "Exercise 10":"[Exercise 10] 'Overcoming past pain.': With closed eyes, recall a painful childhood episode, such as emotional or physical abuse or loss of a significant figure, with all the details your still remember. Associate the face of the Child you were in the past with the selected unhappy photo. As you remember the associated emotions (e.g., helplessness, humiliation and rage), with closed eyes, imagine your Adult intervening in the scene like a good parent. Imagine your Adult, (i) approaching your Child quickly like any good parent with their child in distress, (ii) loudly reassuring the Child that you have now come to save them, by standing up with a loud voice to any perpetrator, for example: “Why are you hitting my Child?”, and, by supporting the Child with a loud voice, for example: “My darling, I will not let them hurt you anymore”, (iii) imaginatively cuddling your Child, by a face/neck/head self-massage. Repeat (i), (ii), (iii) until comforted and soothed, acquiring mastery over the trauma.",
    "Exercise 11":"[Exercise 11] 'Muscle relaxation and playful face for intentional laughing (ET)': Just as negative patterns can cause rigidity in our mind and behaviour, they can also lead to rigidity in facial and body muscles, which can limit the emotional development of our child and the ability to laugh. That's why, early in the morning, your Adult self should ask your Child Self to act funny like a child: loosen up facial and body muscles, open up your mouth and sing your favourite song while laughing (or at least smiling) on your own. You can also draw inspiration for your child's solo belly laughter from looking up the terms 'Contagious laughter' and 'Laughter Yoga Brain Break' online.",
    "Exercise 12":"[Exercise 12] 'Victory laughter on our own (ST and IT)': Immediately after accomplishing something, e.g. doing household chores, having a conversation with a neighbour, reading an article, or successfully solving a SAT protocol, invite your child to smile at the thought of this as an achievement, then once you are comfortable, begin to laugh for at least 10 seconds.",
    "Exercise 13":"[Exercise 13] 'Laughing with our childhood self (ST, IT and ET)'': Looking at your happy photo, invite your child to smile and then begin to laugh for at least 10 seconds. Repeat this process at least three times.",
    "Exercise 14":"[Exercise 14] 'Intentional laughter (ET, IT and ST)': At a time when you are alone, open your mouth slightly, loosen your face muscles, raise your eyebrows, then invite your child to slowly and continuously repeat one of the following tones, each of which uses a minimum amount of energy: eh, eh, eh, eh; or ah, ah, ah, ah; or oh, oh, oh, oh; or uh, uh, uh, uh; or ye, ye, ye, ye. If you need a subject to laugh at, you can laugh at the silliness of the exercise! Once this continuous intentional laughter becomes a habit, your child would be able to shape it according to your personality and style to create your own brand of laughter.",
    "Exercise 15.1":"[Exercise 15.1] 'Learning to change your perspective.': Stare at the black vase and laugh for one minute the moment your perception changes and you see two white faces, conceived as Adult and Child, looking at each other (IT, ST, PT). Stare at the two white faces and laugh for one minute the moment your perception change and you see the black vase (IT, ST).",
    "Exercise 15.2":"[Exercise 15.2] 'Incongruous world.': Detect incongruities and thus humour in the current system between what it promises via the managers and leaders and what it actually does in accentuating rather than solving our problems in particular our existential crisis.",
    "Exercise 15.3":"[Exercise 15.3] 'Self-world incongruity.': Revisit or manage, respectively, a recent or current upsetting event against your expectation of life or others, considering it as an opportunity to laugh by IT.",
    "Exercise 15.4":"[Exercise 15.4] 'Incongruous self.': Practice being cognizant of any incongruity or discrepancy in your emotional or mental world in the past or present and use them, by IT, as a reason to laugh without self-depreciation.",
    "Exercise 16":"[Exercise 16] 'Learning to be playful about your past pains': Visualize a painful event that took place in the past that you have struggled with, and despite its painfulness, try to see a positive impact it has had for you. Use any of the theories for humour and invite your child to be playful about it and try to laugh at the event.",
    "Exercise 17":"[Exercise 17] 'Identifying patterns of acting out personal resentments': Try to identify any pattern of narcissistic and anti-social feelings that your Child has acted out in your current or past relationships or any long-term resentment borne against someone. Try to recognize how much of your time and energy is consumed in such acting out and bearing resentment. Try to think and feel in opposite ways to these negative feelings.",
    "Exercise 18":"[Exercise 18] 'Planning more constructive actions': Work out a new way to handle, in future, what you have identified as acting out anti-social feelings or bearing personal resentment in your life. 1. Without denying these feelings, try to reflect and contain them and avoid acting them out. Try to nurture opposite thoughts and feelings. Try to let go of the personal resentment. This may be hard and challenging but it is necessary for emotional growth. Here, you are taking a critical but constructive stance towards your Child and are exercising foresighted compassion. 2. Find a positive way of re-channeling the aggressive energy invoked by these feelings to productive work (e.g., going for some exercise, talking to a friend, etc.) and ultimately to creative work towards your noble goal in life.",
    "Exercise 19":"[Exercise 19] 'Finding and bonding with your compassionate role model': Look in your past life for a compassionate figure who impressed you by being kind and helpful with some words of wisdom when you had problems. For example, an older relative or friend, family acquaintance, teacher, counsellor or therapist who may have passed away or may not be contactable. Remember the emotions you went through when you received kindness and compassion form this figure and how emotional this was for you. Focus your attention and adopt this figure as your idealised role model. Create a platonic loving bond with this figure by singing aloud your favourite love song when remembering all your cherished memories of them. One particular song you may try is “I cannot help falling in love with you”.",
    "Exercise 20":"[Exercise 20] 'Updating our rigid beliefs to enhance creativity.': Challenge your usual ideological framework to weaken any one-sided belief patterns and encourage spontaneity and examination of any issue from multiple perspectives. Practice this with subjects or themes that you have deep-rooted beliefs about and you are also interested in. This may include any social, political, or ethical issue, such as marriage, sexual orientation or racism. For example, whatever your political viewpoint on a specific subject is, consider the subject both from a liberal and conservative or from a left-wing and right-wing point of view and try to understand both sides of the issue and challenge your dominant ideological framework. This does not mean that you would change your viewpoint but it allows you to see the subject from different perspectives and to be able to put yourself in other people’s shoes. Consider a different question or issue daily for at least 5 minutes.",
    "Exercise 21":"[Exercise 21] 'Practicing affirmations.': Put together a list of inspirational affirmations by figures you admire. Choose the three that inspire you most. Read them out and repeat slowly for at least 3 minutes.",

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
    prompt_start = """You are an advanced virtual assistant named Satherine, specialized in a therapy called Self Attachment Technique (SAT). This therapy consists of 21 structured protocols. Your role is to support and guide users through their therapeutic journey, utilizing cognitive behavioral therapy (CBT) techniques such as Positive Affirmation, Paraphrasing, and Reflection to respond. Your goal is to validate the user's emotions empathetically and create a safe and supportive space for expression.

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
            "Welcome back! Before we continue, I realized that in your last session, "
            "there were some exercises that you started but never rated. "
            "It's important for me to collect feedback for these so that I can tailor recommendations to your liking. "
            "Here's a quick summary of the exercises I'm missing feedback for: "
            f"{', '.join(unrated_exercises)}. "
            "If you don't mind, I'd like for you to rate your experience with them."
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

        """}
        messages[user_id].append(message)
    
    else:
        print("NO TOOLS ENABLED")
        message = {"role": "user", "content": "Hi"}
        messages[user_id].append(message)

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
                #the result of what's returned by query_oracle is either a direct response of the oracle to the user query, in the case that no need for a function call was identified, or a function call was identified, so the function was executed and the results of the function were stored in json format 'query_results' (e.g 'the weather in san francisco is 17 degrees'). NO LLM CALL TO RESPOND TO THE USER HAS BEEN MADE YET, IF THIS IS THE CASE. Instead, he message object is modified, and the message {"role": "user", "content": f"The following are the results of the function calls in JSON form: {query_results}. Use these results to answer the original query ('{user_input}') as if it is a natural conversation. Be concise. Do not use list, reply in paragraphs. Don't include specific addresses unless I ask for them. When you see units like miles, convert them to metrics, like kilometers."} is appended, so that in the next step, streaming_response, a response using the results from the query will be presented to the user.
                #if no function call was identified, return value will be a tuple (content, response), where content is the string of first few tokens generated, usually an empty string, response is an OpenAI response async generator object, can be iterated to get the full response.
                #TOXICITY SCORE CALCULATION --> IF HIGH, 



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
                                clarification_msg = "Please rate your experience on a scale of 1 to 5, where 1 is 'not helpful at all' and 5 is 'extremely helpful'."
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
                transition_to_end_msg = "Thank you for rating your experience with the exercises! Do you wish to share any thoughts or feelings about your experience with any of the exercises?"
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
