from typing import Dict

GPT_MODELS: Dict[str, Dict[str, str]] = {
    "Smart": {
        "model_name": "gpt-4-0613",
        "description": "A stable GPT4 model (gpt-4-0613)",
    },
    "Fast": {
        "model_name": "gpt-3.5-turbo-0613",
        "description": "A stable GPT3.5 model (gpt-3.5-turbo-0613)",
    },
    "GPT4": {
        "model_name": "gpt-4",
        "description": "The latest GPT4 model",
    },
    "GPT3.5": {
        "model_name": "gpt-3.5-turbo",
        "description": "The latest GPT4 model",
    },
    "GPT3.5-16K": {
        "model_name": "gpt-3.5-turbo-16k",
        "description": "The latest GPT3.5 model that supports token size of 16,384 (4 times the context)"
    },
    "GPT3.5-image": {
        "model_name": "ft:gpt-3.5-turbo-0613:extopolis::7vAtD4ud",
        "description": "The GPT3.5 specifically fined tuned for image_generation tool"
    }
}