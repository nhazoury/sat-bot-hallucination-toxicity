from pydantic import BaseModel
from typing import Optional,List, Dict
from collections import OrderedDict
import tools

def init_tools(**kwargs) -> Dict[str, tools.BaseTool]:
    available_tools = OrderedDict()
    for tool_name, tool in tools.name_map.items():
        available_tools[tool_name] = tools.__dict__[tool](**kwargs)
    return available_tools

class ToolStatus(BaseModel):
    tool_name: str
    enabled: bool

class UserToolSettings(BaseModel):
    userId: Optional[str] = None
    toolStatus: Optional[Dict[str, bool]] = None

    def to_dict(self):
        # returns a dictionary of all the values
        return { k:v for k,v in self.__dict__.items() if v is not None}
    
def get_user_tools(db, user_id):
    user_tool_settings_ref = db.collection("user_tool_settings").document(user_id)
    user_tool_settings_doc = user_tool_settings_ref.get()
    if user_tool_settings_doc and user_tool_settings_doc.exists:
        # This contains all the available tools
        settings = {
            tn: False for tn in list(tools.name_map.keys())
        }
        existing_settings = user_tool_settings_doc.to_dict()["toolStatus"] # contains settings enabled by user
        for k, v in existing_settings.items():
            if k in settings:
                settings[k] = v 
        return settings
    else:
        settings = {
            tn: False for tn in list(tools.name_map.keys())
        }
        settings[tools.MemoryTool.name] = True #saahi code, set sat tool to true per default
        return settings

def get_user_enabled_tools(db, user_id):
    user_tools = get_user_tools(db, user_id)
    enabled_tools = [key for key, value in user_tools.items() if value]
    return enabled_tools

def set_user_tools(db, user_id, tool_status: ToolStatus):
    user_tool_settings_ref = db.collection("user_tool_settings").document(user_id)
    user_tool_settings_doc = user_tool_settings_ref.get()
    if user_tool_settings_doc and user_tool_settings_doc.exists:
        curr_dict = user_tool_settings_doc.to_dict()
        curr_dict["toolStatus"][tool_status.tool_name] = tool_status.enabled
        user_tool_settings_ref.update(curr_dict)
    else:
        settings = {
            tn: False for tn in list(tools.name_map.keys())
        }
        settings[tool_status.tool_name] = tool_status.enabled
        curr_dict = {
            "userId": user_id,
            "toolStatus": settings
        }
        user_tool_settings_ref.set(curr_dict)