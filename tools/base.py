from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Any
from pydantic import create_model, validate_arguments, BaseModel, Extra
from inspect import signature
import json

# idea from langchain
class SchemaConfig:
    extra = Extra.forbid
    arbitrary_types_allowed = True

def create_subset_model(name: str, model: BaseModel, field_names: dict):
    fields = {
        field_name: (model.__fields__[field_name].type_, model.__fields__[field_name].default) 
        for field_name in field_names 
        if field_name in model.__fields__
    }
    return create_model(name, **fields)

def create_schema_from_function(model_name: str, func: Callable):
    validated= validate_arguments(func, config=SchemaConfig)
    inferred_model = validated.model
    schema = inferred_model.schema()["properties"]
    valid_keys = signature(func).parameters
    valid_args = {k: schema[k] for k in valid_keys}
    return create_subset_model(f"{model_name}_schema", inferred_model, valid_args)

class BaseTool(ABC):
    name: str # the name of the tool that communicates its purpose
    description: str # used to tell the model how/when/why to use the tool. You can provide few-shot examples as a part of the description.
    user_description: str # similar to description, but a concise version, shown to the user
    usable_by_bot: bool = True # whether or not the tool is used by the bot duirng chat
    disable_all_other_tools: bool = False # when using this tool, should all other tools be disabled?
    def __init__(self, func: Callable=None, **kwargs) -> None:
        '''
        Inputs:
            func: the function to call, can be None if we want to use the _run function
            kwargs: in case of any additional information needed. To ensure consistency, this must be appended to the end of all __init__ functions that inherit BaseTool
        '''
        self.args_schema: Optional[Type[BaseModel]] = None # Pydantic model class to validate and parse the tool's input arguments.
        if func is None:
            self.func = self._run
            self.args_schema = create_schema_from_function(self.name, self._run)
        else:
            self.func = func
            self.args_schema = create_schema_from_function(self.name, func)
        self.args = self.args_schema.schema()
    
    @abstractmethod
    def on_enable(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Called when the tool is enabled
        """
    
    @abstractmethod
    def on_disable(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Called when the tool is disabled
        """

    @abstractmethod
    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Use the tool.

        Add run_manager: Optional[CallbackManagerForToolRun] = None
        to child implementations to enable tracing,
        """

    def parse_input(self, tool_input: dict[str, Any]):
        input_args = self.args_schema
        if input_args is not None:
            result = input_args.parse_obj(tool_input)
            return {k : v for k, v in result.dict().items() if k in tool_input}
        return tool_input
    
    def run(self, tool_input: dict[str, Any]):
        parsed_input = self.parse_input(tool_input)
        # use keyword arguments only
        tool_args, tool_kwargs = (), parsed_input
        return self._run(*tool_args, **tool_kwargs)
    
    def to_json(self):
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "parameters": self.args
        }, indent=4)

def test_func(x: int, y: int):
    return x + y

class TestTool(BaseTool):
    def _run(self, x: int, y: int):
        return self.func(x, y)

if __name__ == "__main__":
    test_tool = TestTool("test", "For testing only", test_func)
    print(test_tool.parse_input({"x": 3, "y": 4}))
    print(test_tool.parse_input({"x": 3, "y": 4, "z": 5}))
    print(test_tool.run({"x": 3, "y": 4, "z": 5}))
    print(test_tool.to_json())