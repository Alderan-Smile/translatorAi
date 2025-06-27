from typing import List
from ..messageClass import message

class payloadChat:
    model: str
    messages: List[message]

    def __init__(self, model: str = "", messages: List[message] = None):
        if messages is None:
            messages = []
        self.model = model
        self.messages = messages
    
    @classmethod
    def from_dict(cls, data):
        model_local = data.get("model", "")
        messages_local = data.get("messages", [])
        messages = [message.from_dict(msg) for msg in messages_local]
        return cls(model=model_local, messages=messages)