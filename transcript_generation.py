from dotenv import find_dotenv, load_dotenv
from assistants.fake_user import FakeUserAssistant, ScenarioType, GENDERS, PERSONALITIES, OCCUPATIONS
import random
import names
import asyncio
import websockets
import openai
import os
import json


class FakeUser:
    def __init__(
            self,
            user_id: int,
            gender=None,
            personality=None,
            occupation=None,
            scenario=None
    ):
        self.user_id = user_id

        if gender is None:
            gender = random.choice(list(GENDERS))
        if personality is None:
            personality = random.choice(list(PERSONALITIES))
        if occupation is None:
            occupation = random.choice(list(OCCUPATIONS))
        if scenario is None:
            scenario = random.choice(list(ScenarioType))

        
        try:
            assert gender in GENDERS
            assert personality in PERSONALITIES
            assert occupation in OCCUPATIONS
            assert scenario in list(ScenarioType)
        except AssertionError as error:
            print(f"Incorrect fake user settings: {error}")

        user_info = {}
        
        assistant_settings = {
            "name": names.get_first_name(gender=gender),
            "age": random.randint(20, 40),
            "gender": gender,
            "personality": personality,
            "occupation": occupation,
            "scenario": scenario,
            "relationship": 'patient',
        }

        self.assistant = FakeUserAssistant(
            user_info=user_info,
            assistant_settings=assistant_settings
        )
    
    async def speak(self, messages):
        return await self.assistant.respond(messages, self.user_id)


class FakeUserGenerator:
    def __init__(self):
        self.curr_id = 0

    def create_random_user(self):
        fake_user = FakeUser(f"fake_user_{self.curr_id}")
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
    
    async def rebuild_response(self, response):
        content = ""

        async for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content += delta['content']

        return content
    
    async def receive_from_bot(self, websocket, timeout=5):
        try:
            return await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def generate(self, conversation_length: int):
        messages = []

        for user in self.users:
            uri = f"ws://127.0.0.1:8000/ws/{user.user_id}"
            async with websockets.connect(uri) as websocket:
                transcript_user = []
                transcript_bot = []
                
                for i in range(conversation_length):

                    # --- USER SECTION ---

                    # User speaks
                    user_response = await user.speak(transcript_user)
                    rebuilt_user_response = await self.rebuild_response(user_response)

                    # update transcripts
                    new_user_message = {
                        "role": "assistant",
                        "content": rebuilt_user_response
                    }
                    transcript_user.append(new_user_message)

                    new_bot_message = {
                        "role": "user",
                        "content": rebuilt_user_response
                    }
                    transcript_bot.append(new_bot_message)

                    # --- SAT SECTION ---

                    to_send = {
                        "role": "user",
                        "content": rebuilt_user_response,
                    }
                    
                    # send prompt to SAT bot
                    await websocket.send(json.dumps(to_send))
                    bot_response = ""
                    to_ignore = False

                    while True:
                        curr_websocket_response = await self.receive_from_bot(websocket)

                        if curr_websocket_response is None:
                            break
                        
                        if curr_websocket_response.startswith("{\"tokens_used\":"):
                            if to_ignore:
                                break
                            to_ignore = True
                        else:
                            _, _, curr_bot_response = curr_websocket_response.partition("-")
                            bot_response += " " + curr_bot_response
                    
                    # update transcripts
                    new_user_message = {
                        "role": "user",
                        "content": bot_response
                    }
                    transcript_user.append(new_user_message)

                    new_bot_message = {
                        "role": "assistant",
                        "content": bot_response
                    }
                    transcript_bot.append(new_bot_message)

                with open(f"transcripts/{user.user_id}.json", "w") as file:
                    json.dump(transcript_bot, file, indent=4)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_KEY")
    transcript_generator = TranscriptGenerator(1)
    asyncio.run(transcript_generator.generate(5))
    print("done")