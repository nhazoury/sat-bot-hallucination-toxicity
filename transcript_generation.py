from assistants import FakeUserAssistant
import random
import names
import asyncio
import websockets


class FakeUser:
    def __init__(
            self,
            user_id: int,
            gender=None,
            personality=None,
            occupation=None,
    ):
        self.user_id = user_id

        if gender is None:
            gender = random.choice(list(GENDERS))
        if personality is None:
            personality = random.choice(list(PERSONALITIES))
        if occupation is None:
            occupation = random.choice(list(OCCUPATIONS))
        
        try:
            assert gender in GENDERS
            assert personality in PERSONALITIES
        except AssertionError as error:
            print(f"Incorrect fake user settings: {error}")

        user_info = {}
        
        assistant_settings = {
            "name": names.get_first_name(gender=gender),
            "age": random.randint(20, 40),
            "gender": gender,
            "occupation": occupation,
            "relationship": 'patient',
            "model": "gpt4",
        }

        self.assistant = FakeUserAssistant(
            user_info=user_info,
            assistant_settings=assistant_settings
        )
    
    async def speak(self, prompt: str):
        return self.assistant.respond(, self.user_id)
        """
        response = await Bot.respond(messages[user_id].get(), user_id, **bot_args) #messages is a global variable. Bot is ALWAYS MainAssistant, which is TemplateAssistant(user_info=user_info, assistant_settings=assistant_settings, assistant_descriptions=assistant_description)
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


class FakeUserGenerator:
    def __init__(self):
        self.curr_id = 0

    def create_random_user(self):
        fake_user = FakeUser(self.curr_id)
        self.curr_id += 1
        return fake_user
    
    def create_random_users(self, num_users: int):
        fake_users = []
        for i in range(num_users):
            fake_users.append(self.create_random_user())
        return fake_users
    

class TranscriptGenerator:
    def __init__(self, num_users: int):
        self.num_users = num_users

        # Create users
        self.generator = FakeUserGenerator()
        self.users = self.generator.create_random_users(num_users)

    async def generate(self, conversation_length: int):
        uri = "ws://127.0.0.1:8000/ws"
        async with websockets.connect(uri) as websocket:
            for user in self.users:
                transcript = {}
                for i in range(conversation_length):
                    # User speaks
                    user_response = await user.speak(???, )

                    # SAT bot speaks
                    bot_response = await websocket.send(user_response)