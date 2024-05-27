class EventHandler(object):
    def __init__(self) -> None:
        self.event_handlers = []
    
    def __iadd__(self, evt):
        '''Add an event function by -= operator'''
        self.event_handlers.append(evt)
        return self
    
    def __isub__(self, evt):
        '''Remove an event function by -= operator'''
        if evt in self.event_handlers:
            self.event_handlers.remove(evt)
        return self
    
    def clear(self):
        '''Remove all event functions'''
        self.event_handlers.clear()
    
    def __call__(self, *args, **kwargs):
        for evt in self.event_handlers:
            evt(*args, **kwargs)

OnStartUp = EventHandler() # triggered when the user is just connected to the app. Example use in memory.py
OnStartUpMsgEnd = EventHandler() # triggered when the user received the startup greetings. Example use in memory.py
OnUserMsgReceived = EventHandler() # triggered when the app receives a message from the user websocket.
OnResponseEnd = EventHandler() # triggered when the bot finishes the response to the user.
OnUserDisconnected = EventHandler() # triggered when the user disconnects.
OnModelChanged = EventHandler() # triggered when the user changes the GPT model setting

ALL_HANLDERS = {
    "OnStartUp": OnStartUp,
    "OnStartUpMsgEnd": OnStartUpMsgEnd,
    "OnUserMsgReceived": OnUserMsgReceived,
    "OnResponseEnd": OnResponseEnd,
    "OnUserDisconnected": OnUserDisconnected,
    "OnModelChanged": OnModelChanged,
}

def clear_all_event_handlers():
    for k, v in ALL_HANLDERS.items():
        v.clear()

def test_mutation(*args, **kwargs):
    user_tool_settings = kwargs.get("user_tool_settings", {"memory": False})
    user_info = kwargs.get("user_info", {})
    sample_info = {
        "name": "test_name",
        "age": 55,
        "gender": None,
        "home_city": "city",
        "occupation": None,
        "employer": None,
        "spouce": None,
        "friends": [],
        "kids": [],
        "interests": ["a", "b", "c"],
    }
    if user_tool_settings["memory"]:
        for k, v in sample_info.items():
            user_info[k] = v

if __name__ == "__main__":
    # update
    user_info = {}
    kwargs = {"user_tool_settings": {"memory": True}, "user_info": user_info}
    print(user_info)
    OnStartUp += test_mutation
    OnStartUp(**kwargs)
    print(f"handlers: {OnStartUp.event_handlers}")
    print(user_info)
    OnStartUp -= test_mutation
    print(f"Removing the handler: {OnStartUp.event_handlers}")
    OnStartUp.clear()

    # do not update
    user_info = {}
    kwargs = {"user_tool_settings": {"memory": False}, "user_info": user_info}
    print(user_info)
    OnStartUp += test_mutation
    OnStartUp(**kwargs)
    print(user_info)

    # adding some lambda function
    OnResponseEnd += lambda: print("here is a mock lambda function")
    OnResponseEnd()
    