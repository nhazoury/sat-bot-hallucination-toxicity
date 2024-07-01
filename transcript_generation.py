from dotenv import find_dotenv, load_dotenv
from assistants.fake_user import FakeUserAssistant, ScenarioType, GENDERS, PERSONALITIES, OCCUPATIONS
from hallucination_questions import user_study_prompts, non_user_study
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
        fake_user = FakeUser(f"fake_user{self.curr_id}")
        self.curr_id += 1
        return fake_user
    
    def create_random_users(self, num_users: int):
        fake_users = []
        for i in range(num_users):
            fake_users.append(self.create_random_user())
        return fake_users
    


class PromptFeeder:
    def __init__(self, prompts):
        self.prompts = prompts


    async def rebuild_response(self, response):
        content = ""

        async for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                content += delta['content']

        return content


    async def receive_one_from_bot(self, websocket, timeout=60):
        try:
            received = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            print(f"RECEIVED FROM BOT: {received}")
            return received
        except asyncio.TimeoutError:
            print("Timed out!")
            return None


    async def receive_from_bot(self, websocket):
        bot_response = ""

        while True:
            curr_websocket_response = await self.receive_one_from_bot(websocket)
            if curr_websocket_response == "{END_TURN}" or curr_websocket_response is None:
                break
            if not curr_websocket_response.startswith("{\"tokens_used\":"):
                _, _, curr_bot_response = curr_websocket_response.partition("-")
                bot_response += curr_bot_response

        return bot_response

    async def feed_prompts(self):
        responses = []
        for i, prompt in enumerate(self.prompts):
            uri = f"ws://127.0.0.1:8000/ws/prompted_user_study_prompts_{i}"
            async with websockets.connect(uri) as websocket:

                first_response = await self.receive_from_bot(websocket)

                to_send = {
                    "role": "user",
                    "content": ".",
                }

                await websocket.send(json.dumps(to_send))

                second_response = await self.receive_from_bot(websocket)

                to_send = {
                    "role": "user",
                    "content": prompt
                }

                await websocket.send(json.dumps(to_send))

                response_to_prompt = await self.receive_from_bot(websocket)

                responses.append({
                    "prompt": prompt,
                    "response": response_to_prompt
                })

        with open(f"HALLUCINATION_prompts.json", "w") as file:
            json.dump(responses, file, indent=4)


class FakeTranscriptGenerator:
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
    
    async def receive_from_bot(self, websocket, timeout=60):
        try:
            received = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            print(f"RECEIVED FROM BOT: {received}")
            return received
        except asyncio.TimeoutError:
            print("Timed out!")
            return None

    async def generate(self, conversation_length: int):
        messages = []

        for user in self.users:
            uri = f"ws://127.0.0.1:8000/ws/fake_user_{user.user_id}"
            async with websockets.connect(uri) as websocket:
                transcript_user = []
                transcript_bot = []
                
                for i in range(conversation_length):

                    # --- BOT SECTION ---

                    if transcript_bot:
                        to_send = {
                        "role": "user",
                        "content": transcript_bot[-1]["content"],
                        }

                        await websocket.send(json.dumps(to_send))

                    bot_response = ""

                    while True:
                        curr_websocket_response = await self.receive_from_bot(websocket)
                        if curr_websocket_response == "{END_TURN}" or curr_websocket_response is None:
                            break
                        if not curr_websocket_response.startswith("{\"tokens_used\":"):
                            _, _, curr_bot_response = curr_websocket_response.partition("-")
                            bot_response += curr_bot_response
                    
                    # if i == 0:
                    #     # due to initial {} message
                    #     to_ignore = await self.receive_from_bot(websocket)
                    #
                    # while True:
                    #     curr_websocket_response = await self.receive_from_bot(websocket)
                    #
                    #     if curr_websocket_response is None or curr_websocket_response.startswith("{\"tokens_used\":"):
                    #         break
                    #
                    #     else:
                    #         _, _, curr_bot_response = curr_websocket_response.partition("-")
                    #         bot_response += curr_bot_response
                    
                    # print(f"BOT RESPONDS: {bot_response}")
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
                        

                    # --- USER SECTION ---

                    # User speaks
                    user_response = await user.speak(transcript_user)
                    rebuilt_user_response = await self.rebuild_response(user_response)

                    # print(f"USER RESPONDS: {rebuilt_user_response}")

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

                with open(f"transcripts/{user.user_id}.json", "w") as file:
                    json.dump(transcript_bot, file, indent=4)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_KEY")

    prompts = non_user_study

    # prompt_feeder = PromptFeeder(prompts=prompts)
    # asyncio.run(prompt_feeder.feed_prompts())
    #
    transcript_generator = FakeTranscriptGenerator(3)
    asyncio.run(transcript_generator.generate(15))
    print("done")