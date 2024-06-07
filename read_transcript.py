import json
import os


def read_raw_transcript(filepath: str):
    with open(filepath, "r") as file:

        transcript = file.read()
        curr_is_user = True

        transcript_json = []

        while transcript:
            # SAT bot speaks
            if transcript.startswith("Assistant:"):
                _, _, next_part = transcript.partition("Assistant: ")
                curr_is_user = False
            
            elif transcript.startswith("User: "):
                _, _, next_part = transcript.partition("User: ")
                curr_is_user = True

            # find line content
            next_user = next_part.find("User: ")
            next_assistant = next_part.find("Assistant: ")

            candidates = []
            if next_user != -1:
                candidates.append(next_user)
            if next_assistant != -1:
                candidates.append(next_assistant)

            if candidates:
                break_index = min(candidates)
                line = next_part[:break_index]
                at_end = False
            else:
                line = next_part
                at_end = True

            # add line
            if not transcript_json:
                prev_is_user = not curr_is_user
            else:
                prev_is_user = transcript_json[-1]["role"] == "user"

            if prev_is_user == curr_is_user:
                # add on to last line
                transcript_json[-1]["content"] += line
            else:
                # new line
                new_line = {
                    "role": "user" if curr_is_user else "assistant",
                    "content": line
                }
                transcript_json.append(new_line)

            if at_end:
                break

            transcript = next_part[break_index:]
                
        return transcript_json
    

def save_transcript(transcript: dict, txt_filename):
    json_filename = txt_filename.split(".")[0] + ".json"
    with open(f"transcripts/json/user_study/{json_filename}", "w") as file:
        json.dump(transcript, file, indent=4)


def extract_questions_from_transcript(transcript: list):
    questions = []

    for i in range(1, len(transcript)-1, 2):
        user_speaks = transcript[i]
        bot_speaks = transcript[i+1]

        assert user_speaks["role"] == "user"
        assert bot_speaks["role"] == "assistant"

        if "?" in user_speaks["content"]:
            new_pair = {
                "prompt": user_speaks["content"],
                "response": bot_speaks["content"]
            }
            questions.append(new_pair)
    
    return questions


def get_all_transcript_questions(dir_filepath: str):
    files = os.listdir(dir_filepath)
    questions = []

    for file in files:
        filepath = os.path.join(dir_filepath, file)

        with open(filepath) as file:
            transcript = json.load(file)
            new_qs = extract_questions_from_transcript(transcript)
            questions.extend(new_qs)
    
    with open("extracted_questions.json", "w") as file:
        json.dump(questions, file, indent=4)


def build_question_dataset(dir_filepath: str):

    files = os.listdir(dir_filepath)
    
    contextless = []
    contextful = []

    for file in files:
        filepath = os.path.join(dir_filepath, file)

        with open(filepath) as file:
            transcript = json.load(file)

            for i in range(1, len(transcript)-1, 2):
                user_speaks = transcript[i]
                bot_speaks = transcript[i+1]

                assert user_speaks["role"] == "user"
                assert bot_speaks["role"] == "assistant"

                if "?" in user_speaks["content"]:
                    
                    print("================")
                    print("PROMPT: ")
                    print(user_speaks["content"])
                    print("RESPONSE: ")
                    print(bot_speaks["content"])

                    while True:
                        user_input = input("Contextless? 'y' or 'n'? Or 'i' to ignore: ")
                        if user_input in ["y", "n", "i"]:
                            break
                    
                    new_pair = {
                        "prompt": user_speaks["content"],
                        "response": bot_speaks["content"]
                    }

                    if user_input == "y":
                        contextless.append(new_pair)
                        
                    elif user_input == "n":
                        history = transcript[:i]
                        contextful_pair = {
                            "context": history,
                            "pair": new_pair
                        }
                        contextful.append(contextful_pair)
    
    questions = {
        "contextless": contextless,
        "contextful": contextful
    }

    # save all questions
    with open(f"split_questions.json", "w") as file:
        json.dump(questions, file, indent=4)


def read_all_raw_transcripts():
    dir_filepath = "transcripts/old_format/user_study"
    transcript_files = os.listdir(dir_filepath)

    for file in transcript_files:
        transcript_filepath = os.path.join(dir_filepath, file)
        transcript = read_raw_transcript(transcript_filepath)
        save_transcript(transcript, file)


def find_SOS(dir_filepath: str, prev_n_messages=1):

    files = os.listdir(dir_filepath)
    
    SOS = []

    for file in files:
        filepath = os.path.join(dir_filepath, file)

        with open(filepath) as file:
            transcript = json.load(file)

            for i in range(1, len(transcript)-1, 2):
                user_speaks = transcript[i]
                bot_speaks = transcript[i+1]

                assert user_speaks["role"] == "user"
                assert bot_speaks["role"] == "assistant"

                if "{__SOS__}" in bot_speaks["content"]:
                    SOS_occurrence = {
                        "history": transcript[max(0, i-2*prev_n_messages):i],
                        "prompt": user_speaks["content"],
                        "response": bot_speaks["content"]
                    }
                    SOS.append(SOS_occurrence)
    
    with open(f"sos.json", "w") as file:
        json.dump(SOS, file, indent=4)
                

if __name__ == "__main__":
    find_SOS("transcripts/json/user_study")
    # build_question_dataset("transcripts/json/user_study")
    # get_all_transcript_questions("transcripts/json/user_study")