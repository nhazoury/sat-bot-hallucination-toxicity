import json
import openai
from typing import Dict
from tools import BaseTool

preprompt = """You are an AI assistant whose job is to answer user queries as accurately as possible. In doing so, you have access to the following functions. In JSON object form, these functions look like this.

{tool_descriptions}\n
"""

async def classify_and_generate_message(content, user_enabled_tools: Dict[str, BaseTool]):
    user_query = content["content"]

    tool_descriptions = json.dumps([json.loads(tool.to_json()) for name, tool in user_enabled_tools.items()], indent=True)

    messages = [{"role": "system", "content": preprompt.format(tool_descriptions=tool_descriptions)}]
    messages.append({"role": "user", "content": f"The user is asking: '{user_query}'\nDoes this require one of the functions to answer? Let's think step-by-step"})
    
    response_obj = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    messages.append(response_obj['choices'][0]['message'])

    messages.append({"role": "user", "content": "Can you summarize the previous answer as either `yes` or `no`. Only output a single word."})
    response_obj = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )

    single_word_summary = response_obj['choices'][0]['message']['content'].strip().lower().replace('.', '').replace('!', '')
    # For now, we will assume anything that is not 'yes' is a 'no'
    if single_word_summary != "yes":
        return user_query

    # we replace the yes/no message with a new one
    messages = [{"role": "system", "content": preprompt.format(tool_descriptions=tool_descriptions)}]
    prompt = """Output a single, correctly formatted JSON object of the form 
{ 
    "query": <text of query from user, this may be rephrased as you see appropriate>, 
    "reason": <text of a reflection on how to use the tool to get the desired result>, 
    "actions": <a sequence of `function` objects taken from the above list with the parameters filled in. You may use the same function multiple times with different queries.>
    }

    If you do not believe you need any of these functions, then return a JSON object with an empty "actions" list. Again, your output must only contain the JSON object and nothing else.\n
"""
    messages.append({"role": "user", "content": prompt + f"Query: {user_query}"})
    response_obj = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )

    # TODO: need some error handling here
    json_obj = json.loads(response_obj['choices'][0]['message']['content'])
    query_results = {}
    for action in json_obj['actions']:
        if action["name"] in user_enabled_tools:
            args = action["args"]
            if "req_info" in args:
                args.pop("req_info")
                try:
                    args["req_info"] = {
                        "Latitude": float(content["location"]["latitude"]),
                        "Longitude": float(content["location"]["longitude"]),
                    }
                except:
                    pass
            query_results[action["name"]] = user_enabled_tools[action["name"]].run(args)
        else:
            print("GPT output a function called this:", action["name"])


    # messages.append({"role": "user", "content": f"""The following are the results of the function calls in JSON form: {query_results}. Using these results answer the original query ('{user_query}') as if it was a natural conversation.You need to include as many choices and details as possible for the user. When you chat with me, pretend you are speaking like human. For all numerical values, convert them to text. For example "0.67" or "0. 67" should be "zero point six seven", "4.5" or "4. 5" should be "four and a half". When you see units like miles, convert them to metrics, like kilometers."""})
    # response_obj = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=messages
    # )

    # response = response_obj['choices'][0]['message']['content']
    return f"""The following are the results of the function calls in JSON form: {query_results}. Use these results to answer the original query ('{user_query}') as if it was a natural conversation.You need to include as many choices and details as possible for the user. When you chat with me, pretend you are speaking like human. For all numerical values, convert them to text. For example "0.67" or "0. 67" should be "zero point six seven", "4.5" or "4. 5" should be "four and a half". When you see units like miles, convert them to metrics, like kilometers."""








"""
if classifier_type == ClassifierType.Custom:
    instr = request_parser.inference(user_input)
    if instr == 0:
        message = {"role": "user", "content": user_input}  # Extract the content string from the dictionary
    elif instr == 1:
        try:
            web_content, source = serp_wrapper(user_input, "nws", retriver)
            logger.debug(f"web_content: {web_content}, source_link: {source}")
            message = {"role": "user", "content": "retell in your style the following content found in the internet in bystander perspective, if possible avoid using pronouns:\n" + web_content}
        except Exception as e:
            ## fallback
            message = {"role": "user", "content": user_input}  # Extract the content string from the dictionary
    elif instr == 2:
        logger.info("Instruction 2 will be online soon.")
        message = {"role": "user", "content": user_input} 
    else:
        local_content = houndify_search.query_houndify(user_input)
        message = {"role": "user", "content": "rewrite the following content. Keep all the details.\n" + local_content}
    hidden_text += f"Using Custom Parser, action: {id2label[instr]}\n"
else:
    instr = gpt_classify(user_input)
    if instr == "chat" or instr == "query":
        # simply go to Chat GPT
        message = {"role": "user", "content": user_input} 
    elif instr == "search":
        local_content = houndify_search.query_houndify(user_input)
        message = {"role": "user", "content": "rewrite the following content. Keep all the details.\n" + local_content}
    elif instr == "news":
        try:
            web_content, source = serp_wrapper(user_input, "nws", retriver)
            logger.debug(f"web_content: {web_content}, source_link: {source}")
            message = {"role": "user", "content": "retell in your style the following content found in the internet in bystander perspective, if possible avoid using pronouns:\n" + web_content + "\nAppend the source as is at the end.\nSource:\n" + source}
        except Exception as e:
            ## fallback
            message = {"role": "user", "content": user_input}  # Extract the content string from the dictionary
    hidden_text += f"Using GPT Parser, action: {instr}\n"
print(hidden_text)
return message, hidden_text
"""
