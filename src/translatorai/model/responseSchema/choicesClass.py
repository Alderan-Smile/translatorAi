from ..messageClass import message as messageGet

class choice:
    message: messageGet

    def __init__(self, message: messageGet = None):
        self.message = message

    @classmethod
    def from_dict(cls, data):
        msg = messageGet.from_dict(data.get("message", {}))
        return cls(message=msg)