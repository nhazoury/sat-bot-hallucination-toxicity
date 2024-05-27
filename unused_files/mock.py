from main import *

@app.websocket("/ws_mock/{user_id}")
async def websocket_mock_endpoint(websocket: WebSocket, user_id: str):

    if user_id not in user_events:
        user_events[user_id] = asyncio.Event()

    
    messages[user_id] = Messages(db,user_id)

    asyncio.create_task(messages[user_id].initilize_messages())

    await websocket.accept()

    # clear chat history on start up
    ###### Delete #########
    await messages[user_id].clear_messages()
    clear_history_flags[user_id] = False

    i=0
    # try:
    while True:

        data = await websocket.receive_text()

        # Handle clearing history
        if user_id in clear_history_flags and clear_history_flags[user_id] and user_id in messages:
            asyncio.create_task(messages[user_id].clear_messages())
            clear_history_flags[user_id] = False

        content = json.loads(data)['content']
        message = {"role": "user", "content": content}  # Extract the content string from the dictionary
        messages[user_id].append(message)
        asyncio.create_task(messages[user_id].save_message_to_firestore(message))


        # logger.debug(f"Message send to openai:\n" +"\n".join([f"{message['role']}: {message['content']}" for message in message_tracker.messages]))
        
        """
        print("MESSAGES: \n\n")
        for message in messages[user_id].get():
            print(message['role'] + ":  " + message['content'])
        """
        
        all_messages = messages[user_id].get()
        model_name = "Mocked" 
        print(f"{user_id} - model_name: {model_name}")

        retry_count = 0
        max_retries = 5
        timeout = 4
        while retry_count < max_retries:
            try:
                print(f"\t{user_id} - Query OpenAI - \"{all_messages[-1]}\"")
                # response = await asyncio.wait_for(
                #     openai.ChatCompletion.acreate(
                #         model=model_name,
                #         messages=messages[user_id].get(),
                #         temperature=0.9,
                #         stream=True,
                #     ), timeout=timeout + retry_count
                # )
                response = mock_openai()
                print(f"\t\t{user_id} - Recieved Response")

            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                break
            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                break
            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                break
            except asyncio.TimeoutError:
                retry_count += 1
                print(f"Timeout occurred for user {user_id}, retrying attempt {retry_count}...")
                continue

            break
        if retry_count == max_retries:
            print(f"Attempted {max_retries} retries and failed. Skipping this call.")
            logger.error(e)
            await websocket.send_text("0-Sorry, the service that I rely on from OpenAI is currently down. Please try again.")
            await websocket.send_text("0-END")
            continue
                

            break
        if retry_count == max_retries:
            print(f"Attempted {max_retries} retries and failed. Skipping this call.")
            logger.error(e)
            await websocket.send_text("0-Sorry, the service that I rely on from OpenAI is currently down. Please try again.")
            await websocket.send_text("0-END")
            continue
                
        
        content = ''
        full_response = ''
        sentences = []
        num_sentences = 1
    
        try:
            interrupt_flags[user_id] = False
            order = 0
            async for chunk in response:
                # logger.debug(f"chunk payload {chunk}")
                if user_id in interrupt_flags and interrupt_flags[user_id]:
                    interrupt_flags[user_id] = False
                    logger.warning(f"{user_id} INTERUPTED")
                    raise asyncio.CancelledError

                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    content += delta['content']
                    content = filter_emojis(content)
                    full_response += delta['content'] 

                
                sentences_exist, sentence, rest = check_sentence(content) 
                if sentences_exist:
                    sentences.append(sentence)
                    content = rest 

                    
                if len(sentences) == num_sentences: 
                    first_two = str(order) + '-' +  ' '.join(sentences)
                    #print("first: ", first_two) 
                    await websocket.send_text(first_two) 
                    order += 1
       
                    sentences = []
                    # logger.info("SENT: " + first_two)
                    # num_sentences += 1

            
        except asyncio.CancelledError:
            logger.warning("Cancelled")
            pass

        if user_id in interrupt_flags and interrupt_flags[user_id]:
            interrupt_flags[user_id] = False
        

        if len(sentences) > 0:
            #print("rest of sentences: " + ' '.join(sentences))
            await websocket.send_text(str(order) + '-' + ' '.join(sentences))
            order += 1

            logger.info("rest of sentences: " + ' '.join(sentences))
        if len(content) > 3 and sum(c.isalpha() for c in content) >= 3:
            #print("rest of content: " + content)
            await websocket.send_text(str(order) + '-' + content)
            order += 1
  
            logger.info("rest of content: " + content) 

        await websocket.send_text(f'{order-1}-END')
        assistant_message = {"role": "assistant", "content": full_response}
        asyncio.create_task(messages[user_id].save_message_to_firestore(assistant_message))

        messages[user_id].append(assistant_message)
        # await websocket.send_json([assistant_message])
# except Exception as e:
#     logger.error(e)
    i+=1


    
async def mock_openai():
    #messages = ["Ah, hello again!", "Just finished pondering the possibilities of parallel universes.", "Did you know there might be infinite versions of ourselves out there?", "Quite mind-boggling, isn't it?"]
    messages = ["Once upon a time", "in a land far", "far away, there was a curious young girl named Emily ", "She loved exploring and was always seeking out new adventures . ", "One day , while wandering through a dense forest , Emily stumbled upon an old , mysterious book . ", "The book had a strange symbol on its cover and seemed to beckon her . ", "Curiosity getting the better of her . ", "Emily opened the book ", "and in a flash of light ", "she was transported to a magical kingdom . In this kingdom ", "mythical creatures roamed freely ", "and enchantments were common . With each turn of a page ", "Emily discovered a new quest and embarked on thrilling adventures . Along the way ", "she made lifelong friends", "solved perplexing riddles ", "and even found true love . This magical book changed Emily's life forever ", "filling it with wonder , courage , and endless possibilities . And so, her extraordinary journey continued , with the turn of every page . ", "The end . "]
    for message in messages:
        await asyncio.sleep(0.01)
        for chunk in message.split(" "):
            response = {
                "choices": [
                    {
                        "delta": {
                            "content": ' ' + chunk
                        },
                        "finish_reason": None,
                        "index": 0
                    }
                ],
                "created": 1687961281,
                "id": "chatcmpl-7WQ7tTiT4rpXcHUU0Y7AkxGUwAvhP",
                "model": "gpt-3.5-turbo-0613",
                "object": "chat.completion.chunk"
            }
            yield response

    response = {
        "choices": [
            {
            "delta": {},
            "finish_reason": "stop",
            "index": 0
            }
        ],
        "created": 1687961281,
        "id": "chatcmpl-7WQ7tTiT4rpXcHUU0Y7AkxGUwAvhP",
        "model": "gpt-3.5-turbo-0613",
        "object": "chat.completion.chunk"
    }
    yield response

# This causes the app to freeze and I don't know why
"""
@app.get("/mock_openai")
async def mock_openai_endpoint():
    print("mocked openAI called")

    
    data = await request.json()
    model_name = data.get("model_name")
    messages = data.get("messages")

    # Perform any desired mocking or logic based on the inputs
    # You can return a predefined response or simulate the behavior of OpenAI

    response = {
        "choices": [
            {
                "message": "This is a mocked response from OpenAI.",
                "role": "assistant"
            }
        ]
    }

    return response
"""
