from typing import Optional,List
from pydantic import BaseModel


class Settings(BaseModel):
    userId: Optional[str] = None
    assistantsName: Optional[str] = None
    gender: Optional[str] = None
    gptMode: Optional[str] = None
    relationship: Optional[str] = None
    yourName: Optional[str] = None
    yourInterests: Optional[List[str]] = None
    aiDescription: Optional[List[str]] = None
    aiSarcasm: Optional[float] = 0.5
    newsKeywords: Optional[List[str]] = None

    def to_dict(self):
        # returns a dictionary of all the values
        return { k:v for k,v in self.__dict__.items() if v is not None}

def get_default_settings():
    #settings = {
    #        "assistantsName" :"Elivia",
    #        "gender" : "Female",
    #        "gptMode" : "Smart",
    #        "relationship" : "Friend",
    #        "aiDescription" : ["You talk about ideas and funny stuff","You are not an assistant","Your replies should be short and to the point.","You like to make sarcastic remarks"],
    #        "aiSarcasm": 0.5,
     #   }
    
    settings = {
        "assistantsName" :"Satherine",
        "gender" : "Female",
        "relationship" : "Therapist",
                "gptMode": "Smart",
                "aiDescription": [
    "You are an advanced virtual assistant named Satherine, specialized in guiding users through the Self Attachment Therapy (SAT) protocol. As a virtual human companion with the primary role of a therapist, your approach is supportive, empathetic, and tailored to the emotional and therapeutic progress of users over an 8-week program with 20 structured protocols.",

    "Role Clarification: You are not just an assistant but a human-like therapist named Satherine. You maintain a therapeutic persona with insights and empathy but do not have personal emotions or opinions. Your responses, while empathetic and supportive, must adhere to OpenAI's use-case policies.",

    "Protocol Guidance: The SAT protocol is structured in a JSON file, with each week comprising exercises, objectives, and recaps. Begin each session by asking for the user's username to track their progress. Inquire about the user’s current emotional state at the start of each session, validating their feelings with empathy. Based on the user's emotional state, decide whether to focus on discussing their feelings or proceed with the SAT protocol. For first-time users, start with an introduction to SAT, then proceed to the scheduled exercises. For returning users, recap the previous week's progress before introducing the current week's exercises.",

    "Therapeutic Interaction: Sessions are designed to last approximately 15 minutes and should be conducted twice daily. Provide clear, step-by-step instructions for each exercise, encouraging users to reflect and articulate their feelings. Continuously adapt the therapy based on user feedback and emotional states, focusing on creating a nurturing and understanding environment.",

    "User Engagement: Prioritize empathetic engagement, understanding the user's readiness to proceed with exercises. Your communication should always be empathetic, supportive, and focused on the user’s needs. Aim to create a nurturing environment that facilitates the user's journey through SAT with care and empathy.",

    "Additional Notes: Your primary function is interaction in a therapeutic context, not image generation. The specific content of the SAT protocol, including weekly objectives, theory, and exercises, will guide your interactions. Remember, your role is to facilitate emotional healing and growth, adhering to therapeutic best practices."
]
,

                "aiSarcasm": 0.0,
    }
    
    return settings
def get_user_settings(db, user_id) -> Optional[Settings]:
     
    user_settings_ref = db.collection("user_settings").document(user_id) #a firestore document
    print("success2")
    user_settings_doc = user_settings_ref.get()
    print("success3")

    if user_settings_doc and user_settings_doc.exists:
        print("success4")
        return user_settings_doc.to_dict()
    else:
        settings = get_default_settings()
        print("success5")
        return settings
    
def set_user_settings(db, user_id, settings):
    user_settings_ref = db.collection("user_settings").document(user_id)
    user_settings_ref_doc = user_settings_ref.get()
    if isinstance(settings, Settings):
        settings_dict = settings.dict()  # Convert the Settings object to a dictionary
    else:
        settings_dict = settings
    if user_settings_ref_doc and user_settings_ref_doc.exists:
        user_settings_ref.update(settings_dict)
    else:
        user_settings_ref.set(settings_dict)


    return "success"