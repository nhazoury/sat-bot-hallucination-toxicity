from abc import ABC, abstractmethod
from typing import Any, List, Dict
import openai
import json
import tiktoken
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class BaseAssistant(ABC):
    
    def __init__(self, user_info: Dict[str, Any], assistant_settings: Dict[str, Any], assistant_descriptions: List=[]) -> None:
        self.user_info = user_info
        self.assistant_settings = assistant_settings
        self.assistant_descriptions = assistant_descriptions
        self.init_prompt_offset = 4

        if "model" not in self.assistant_settings:
            # self.assistant_settings["model"] = "gpt4"
            self.assistant_settings["model"] = "gpt-3.5-turbo-0125"
            # self.assistant_settings["model"] = "gpt-4-turbo"

        print(f"INIITIALISING ASSISTANT: using model '{self.assistant_settings['model']}'")
        
        if "gender" not in self.assistant_settings:
            self.assistant_settings["gender"] = "other"

        if "sarcasm" not in self.assistant_settings:
            self.assistant_settings["sarcasm"] = 0.0
        
        if "name" in self.assistant_settings:
            self.name = self.assistant_settings["name"]
        
        self.additional_information = ""
        # instatiates self.initial_prompt
        self._construct_initial_prompt()

    @abstractmethod
    def _assistant_profile(self) -> str:
        pass

    # TODO: probably need error checks in here, but I'm not bothering for now
    def _user_profile(self) -> str:
        if self.user_info is None or len(self.user_info) == 0:
            return ""
        
        for k, v in self.user_info.items():
            if v is None or (isinstance(v, list) and len(v) == 0):
                self.user_info[k] = "Unknown"
        self.user_info.pop("last_conversation", "")
        user_prompt = f"Here is some information that you know about the user:\n{json.dumps(self.user_info, indent=True)}."
        return user_prompt
    
    def insert_information(self, additional_information):
        if len(additional_information) == 0:
            self.additional_information = ""
        self.additional_information = f"Here is some additional information for this conversation: {additional_information}" #always overwritten with each new user message! this inserts relevant documents from pinecone db after user query. This begs the question, do we really need retrieval performed after every user message? what if they're irrelevant?
    
    def   _construct_initial_prompt(self) -> None:
        self.initial_prompt = self._assistant_profile() + '\n' + self._user_profile()
        if len(self.additional_information) > 0:
            self.initial_prompt += "\n" + self.additional_information

    def __trim_conversation(self, conversations):
        total_len = 0
        for i in range(len(conversations) - 1, -1, -1):
            total_len += len(conversations[i]["content"].split(" ")) + 1 # to also account for the role token
            if total_len > self.threshold:
                print(f"Session length: {total_len}")
                return conversations[i:]
        print(f"Session length: {total_len}")
        return conversations
    
    def summarize_conversation(self, messages: List[Dict[str, str]]):
        model_id = "Astonzzh/bart-augmented"
        tokenizer = AutoTokenizer.from_pretrained("Astonzzh/bart-augmented", truncation=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer) #model is of type  AutoModelForSeq2SeqLM, finetuned by ziheng, tokenizer is of type AutoTokenizer, finetuned by ziheng
        summary = summarizer(messages, max_length=max(int(len(messages.split()) / 2), 30))[0]['summary_text']
        return {"role": "user", "content": summary}

    async def generate_summary(self, messages: List[Dict[str, str]]):
        print("GENERATING FOLLOW UPS")
        new_message = """Summarize the conversation. Output only the complete JSON object. The summarization should not exceed 40 words. Do not make up any details, just try to capture as much information as possible within the constraints so that the summary of the conversation is as accurate and detailed as possible. Specially make note of problems the user mentions, and any exercises the have done so far.
        Format:
        {
            "summary": <string>
        }"""

        conversations = self.__trim_conversation(messages)

        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-16k",
            # model="gpt-4-turbo",
            messages=conversations + [{"role": "user", "content": new_message}],
            # temperature=0
        ) #currently we just send this as a prompt to gpt and hope for the best, we hope it actually executes this and returns the filled out json. perhaps to make this more robust, we can replace the regular llm call with openai function format like in our langgraph implementation

        try:
            content = response['choices'][0]['message']['content']
            print(content)
            conversation_summary = json.loads(content)
            conversation_summary_message = "You have the following summary of the conversation that happened up until this point, and messages that happened after. Summary until here: " + conversation_summary["summary"]
            return {"role": "user", "content": conversation_summary_message}
        except Exception as e:
            print(e)

    def _insert_initial_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prompt = {"role": "system", "content": self.initial_prompt}

        #if len(messages) >= 12:
            #messages = messages[-20:-2] + [prompt] + messages[-2:]
            #print("BOT IS RESPONDING WITH THE FOLLOWING MESSAGES: ", messages)
        if len(messages) > 4: #QUESTION: How come 4? bcos u dont want the additional information to be inserted as the last messAGE, since the last message is the user input
            messages = messages[:-2] + [prompt] + messages[-2:]
        else:
            messages = [prompt] + messages[:]

        ## ADD ANOTHER CASE! If it's a new session, we want the AI with the new system prompt to kick in. It has different instructions than the previous AI session assistant. 
        ## also, we dont want to pass the entire convo history to chatcompletions, you just want it to respond to the last few messages. However, i think such a safeguard is already implemented in the messages obj, or CAN be implemented there.
        return messages
    
    # def num_tokens_from_messages(self, messages, model="gpt-4-0613"):
    # def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0125"):
    def num_tokens_from_messages(self, messages, model="gpt-4-turbo"):


        encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def respond( #async functions need to be executed using 'await' 
        self,
        messages: List[Dict[str, str]],
        user_id,
        *args: Any, 
        **kwargs: Any, 
    ) -> Any:
        self.user_info = kwargs.get("user_info", {}) #stuff from frontent, like name and interests. not necessary
        self._construct_initial_prompt()

        # Summarize all messages except the last 20
        if len(messages) >= 15:
            #summary = await self.generate_summary(messages[:-12])
            #summary = summarize_conversation(messages[:-12])
            messages = messages[-15:-2] + [{"role": "system", "content": self.initial_prompt}] + messages[-2:]
        else:
            messages = self._insert_initial_prompt(messages)
        # print(f"Responding with sarcasm: {self.assistant_settings['sarcasm']}")
        # print(f"Responding with messages: ", messages)

        #calculate number of tokens used when passing to gpt
        num_tokens = self.num_tokens_from_messages(messages)
        print(f"NUMBER OF TOKENS USED FOR ONE ITERATION: ", num_tokens)
        # total_tokens_used_today = kwargs.get("total_tokens_used_today", {})
        # total_tokens_used_today[user_id] += num_tokens
        # print(f"TOTAL NUMBER OF TOKENS USED SO FAR: ", total_tokens_used_today[user_id])

        

        response = await openai.ChatCompletion.acreate(
            model=self.assistant_settings["model"],
            messages=messages,
            temperature=self.assistant_settings["sarcasm"],
            #temperature=0.0,
            stream=True
        )
        return response