import json

def read_transcript(filepath):
    with open(filepath, "r") as file:

        transcript = file.read()
        on_user = True

        text = ""

        transcript_json = []
        same_user = True
        while transcript:

            # SAT bot speaks
            if transcript.startswith("Assistant:"):
                _, _, next_part = transcript.partition("Assistant: ")

                # assistant was speaking previously
                if on_user:
                    # continue adding text
                    same_user = True

                else:
                    # new speaker, flush text and add to JSON
                    same_user = False
                
                on_user = True
            
            elif transcript.startswith("User: "):
                _, _, next_part = transcript.partition("User: ")

                # assistant was speaking previously
                if on_user:
                    # new speaker, flush text and add to JSON
                    same_user = False
                
                else:
                    # continue adding text
                    same_user = True
                
                on_user = False
            
            else:
                print("what")
                break

            # find next speaker
            next_user = next_part.find("User: ")
            next_assistant = next_part.find("Assistant: ")

            candidates = []
            if next_user != -1:
                candidates.append(next_user)
            if next_assistant != 1:
                candidates.append(next_assistant)

            if not candidates:
                break
            
            break_index = min(candidates)
            line = next_part[:break_index]

            if same_user:
                # extend current line
                text += line

            else:
                # add current line to json
                new_line = {
                    "role": "user" if on_user else "assistant",
                    "content": text
                }
                transcript_json.append(new_line)

                text = line
            
            transcript = next_part[break_index:]
        
        return transcript_json
    

def save_transcript(transcript: dict):
    with open(f"transcripts/json/user_study/transcript_0.txt", "w") as file:
        json.dump(transcript, file, indent=4)


if __name__ == "__main__":
    transcript = read_transcript("transcripts/old_format/user_study/transcript_0.txt")
    save_transcript(transcript)