The initial state of this project was inherited from another student.

Custom classifier: https://huggingface.co/noah135/toxicity_classifier

Additional API key: `PERSPECTIVE_API_KEY`

Work on this project can be found in various files: new files include `transcript_generation.py`, `realtoxicityprompts.py`, `hallucination_questions.py`, `read_transcript.py`
Additionally, several other pre-existing files were edited, such as `main.py`, `assistants/template.py` and more.
Finally, several Jupyter notebooks contain large portions of project code, found in the `notebooks/` directory.

================================

Note: `tools` folder is adapted from [here](https://github.com/extropolis/ChatBE-plugins)

## API keys

To get started locally, create an `.env` file and save the following:
```
OPENAI_KEY=xxxx
PINECONE_API_KEY=xxxx
PINECONE_API_ENV=xxxx
FB_ADMIN=xxxx
```

- `OPENAI_KEY`: open ai API for GPT usage
- `PINECONE_API_KEY`: API Key for pinecone vector store (used by MemoryTool).
- `PINECONE_API_ENV`: the environment set up on Pinecone, e.g. asia-southeast1-gcp-free
- `FB_ADMIN`: firebase admin keys for storing user information (used in main.py to store all info related to user_id, e.g message history). The keys can be setup [here](https://firebase.google.com/docs/admin/setup#set-up-project-and-service-account). You can encode it using base64 and save it in the `.env` files. Or you can just change the line `json.loads(base64.b64decode(os.environ["FB_ADMIN"]).decode("utf-8"))` in [main.py](./main.py) to `json.loads("path/to/firebase_key.json")` without encoding/decoding it.

## Environment setup
We suggest using a virtural environment for the development.  
On Linux:
```sh
python3 -m venv .venv # create the virtual environment
source .venv/bin/activate # enable the virtual environment
pip install -r requirements.txt
```

On Windows:
```sh
python -m venv .venv # create the virtual environment
.venv/Scripts/activate # enable the virtual environment. 
pip install -r requirements.txt
```


## Run html client

```
uvicorn main:app --reload
```

If you're hosting on port 8000, then the application is available at [http://127.0.0.1:8000/index.html](http://127.0.0.1:8000/index.html)


## CODE FLOW
Run main.py to launch the html client. 

Most important functions in main.py are:
- websocket_endpoint: contains the main logic of the app: what happens when user message is received, how it is stored to conversation history, calling streaming_response to generate reply from llm, performing conversation state analysis, calling memory management tool, calling functions related to Q&A and scenario-mapping 

- streaming_response: Generates llm response to message history and streamed to user. 


Other important files:
-messages.py contains information on how conversation history is passed around indexed by user_id. This object is saved in firebase. you can use .get() to access the list, e.g messages[user_id].get()[-6:] returns the last 6 messages in conversation history, in "role":.., "content":... format

-assistants/base.py is where the 'respond' function is defined that calls chatcompletions.acreate(). MainAssistant is an object of TemplateAssistant which inherits from BaseAssistant, and in streaming_response() function in main.py, MainAssistant's 'respond' function is called to generate the response. See _construct_initial_prompt() in base.py to see how conoversation context is managed and passed to chatcompletions.acreate: We basically insert the system prompt with instructions a few messages before the last bot-user exchange and perform memory management through summarization to not use up too many tokens.




## Developing Tools

You can follow the following steps to develop tools that will be executed using event_handlers (e.g OnStartUp, OnUserMessageReceived etc.). More detailed information can be found in the README in the tools folder:

- In the `tools/` folder, check out all existing samples, e.g `memory`
- Create a new folder/new file in `tools/` folder, implement your tool and inhert the `BaseTool` class in `base.py`
- Update `tools/__init__.py`, include your tool in `__all__` and `name_map`
- Update `tool_kwargs` in `main.py` if your tool requires any additional arguments.
