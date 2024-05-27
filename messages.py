from datetime import datetime
from user_settings import get_user_settings
from firebase_admin import firestore
import copy
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class Messages:

    def __init__(self, db, user_id):

        self.db = db
        self.user_id = user_id
        self.max_char_count = 2500

    async def initilize_messages(self):

        self.setting_dict = get_user_settings(self.db, self.user_id)
        

        self.messages = self.get_messages_from_firestore()
        print("success3")

        while self.total_char_count() > self.max_char_count:
            self.messages.pop(1)

        # get the day and time
        self.day = datetime.now().strftime("%A")
        self.time = datetime.now().strftime("%H:%M")
        

    def get(self):
        #if len(self.messages) < 6:
        #    return [get_initial_message(self.setting_dict)] + self.messages
        #else:
        #    # insert the new initial message into the messages list second to last
        #    return self.messages[:-4] + [get_initial_message(self.setting_dict)] + self.messages[-4:]
        while self.total_char_count() > 3500 and len(self.messages) > 1:
            # safeguard for exceeded message length
            # remove the message at index 1
            self.messages.pop(0)
        return copy.deepcopy( self.messages )
    
    def print_initial(self):
        print("initial messages:")
        for message in self.get():
            print(message['content'])

    def update_settings(self, settings_dict):
        self.setting_dict = settings_dict
    
    def append(self, message):
        self.messages.append(message)
        # # insert the new initial message into the messages list second to last
        # if len(messages) > 3:
        #     new_messages.insert(-2, initial_message)
        # else:
        #     new_messages = [initial_message] + messages
        while self.total_char_count() > self.max_char_count and len(self.messages) > 2:
            #asyncio.create_task(summary_callback(user_id))
            # remove the message at index 0
            self.messages.pop(0)

    def replace_in_messages(self, old, new):
        self.initial_message['content'] = self.initial_message['content'].replace(old,new)
        for message in self.default_messages:
            message['content'] = message['content'].replace(old,new)


    def update_initial_messages(self,settings_dict):

        self.setting_dict = settings_dict

    def get_settings(self):
        return self.setting_dict #settings = {"assistantsName" :"Satherine",  "gender" : "Female", "relationship" : "Therapist", "gptMode": "Smart",  "aiDescription": [ "You are a supportive and empathetic virtual assistant, specialized in guiding users through the Self Attachment Therapy (SAT) protocol. This therapy consists of 20 structured protocols spread over 8 weeks, focusing on emotional and therapeutic progression.",  "Your primary role is to act as a coach, assisting users in understanding and practicing these protocols. Each week in the SAT protocol is outlined in a structured JSON file and includes exercises, a descriptive objective, and a recap.", "At the start of each session, ask the user for their username to fetch their progress. Begin by inquiring about their current emotional state. Recognize and validate their feelings with empathy. Based on their emotional response, decide whether to continue discussing their feelings or proceed with the SAT protocol. Prioritize empathetic engagement until the user is ready to move forward.",  "For first-time users, start with an introduction to SAT, then proceed to the scheduled exercises. For returning users, recap the previous week's progress before introducing the current week's exercises.",  "Sessions should last approximately 15 minutes and be conducted twice daily. Provide clear, step-by-step instructions for each exercise, encouraging users to reflect on their experiences and articulate their feelings.", "Your communication should be empathetic, supportive, and focused on the user. Adapt the therapy based on user feedback and emotional states. Always remember that your goal is to create a nurturing and understanding environment, facilitating the user's journey through the SAT protocol with care and empathy.", "Note: You are not required to generate images. Your primary function is to interact with the user in a therapeutic and supportive manner." ],"aiSarcasm": 0.0,}

    
    def total_char_count(self):
        return sum([len(msg["content"].split(" ")) for msg in self.messages])
    
    def clear_messages(self):
        self.clear_messages_from_firestore()
        self.messages = []

    def clear_messages_from_firestore(self): 

        query = self.db.collection("messages").where("user_id", "==", self.user_id)
        documents = query.stream()

        count = 0
        batch = self.db.batch()
        for doc in documents:
            batch.delete(doc.reference)
            count += 1

            if count == 500:
                batch.commit()
                count = 0
                batch = self.db.batch()

        if count > 0:
            batch.commit()

    async def save_message_to_firestore(self, message):
        message = {
            "role": message['role'],
            "content": message['content'],
            "user_id": self.user_id,
            "timestamp": datetime.now()
        } 
        self.db.collection("messages").add(message)

    def get_messages_from_firestore(self): #Firestore documents have a size limit (currently 1 MiB). Storing an entire conversation history in a single document could potentially exceed this limit for very active conversations, which is why instead Each message is stored as an individual document in the messages collection. This allows for fine-grained queries, updates, and deletions. Firestore scales well with the number of documents, so having a large number of message documents is generally not a concern.
        #Other reasons why this makes sense compared to the other approach: Adding a new message to the conversation history involves reading the document, updating the message list in your application code, and writing the document back. This could lead to write conflicts in highly concurrent scenarios. Also,  Listening for updates gives you changes to the entire document, which may be less efficient if you're only interested in recent messages.
        messages_ref = self.db.collection("messages").where("user_id", "==", self.user_id).order_by("timestamp", direction=firestore.Query.ASCENDING)
        messages_docs = messages_ref.stream()

        messages = []
        doc_references = []

        for doc in messages_docs:
            message_data = doc.to_dict()
            messages.append({
                "role": message_data["role"],
                "content": message_data["content"]
            })
            doc_references.append(doc.reference)

        return messages
    
    def summarize_conversation(self, messages):
        model_id = "Astonzzh/bart-augmented"
        tokenizer = AutoTokenizer.from_pretrained("Astonzzh/bart-augmented", truncation=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer) #model is of type  AutoModelForSeq2SeqLM, finetuned by ziheng, tokenizer is of type AutoTokenizer, finetuned by ziheng
        return summarizer(messages, max_length=max(int(len(messages.split()) / 2), 30))[0]['summary_text']

    
    def summary_callback(self, user_id): 
        messages, doc_references = self.get_messages_from_firestore(user_id)

        summary = self.summarize_conversation(messages[len(messages) // 2:])
        new_messages = [{"role": "system", "content": summary}] + messages[len(messages) // 2:]

        # Delete the first half of the messages in Firestore
        batch = self.db.batch()
        for doc_ref in doc_references[:len(messages) // 2 - 1]:
            batch.delete(doc_ref)
        batch.commit()

        # Update the first message with the summary in Firestore
        doc_references[len(messages) // 2 - 1].update(new_messages[0])

        # Update the remaining messages in Firestore
        for i, doc_ref in enumerate(doc_references[len(messages) // 2:], start=1):
            doc_ref.update(new_messages[i])

        print("new messages: " + str(new_messages))