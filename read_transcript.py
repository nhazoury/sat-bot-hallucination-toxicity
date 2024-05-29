import json
import os


def read_raw_transcript(filepath):
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


if __name__ == "__main__":
    dir_filepath = "transcripts/old_format/user_study"
    transcript_files = os.listdir(dir_filepath)

    for file in transcript_files:
        transcript_filepath = os.path.join(dir_filepath, file)
        transcript = read_raw_transcript(transcript_filepath)
        save_transcript(transcript, file)