from typing import List
from .choicesClass import choice
from .choicesComplClass import choicesCompl
from .errorClass import errorcontent

class responseChat:
    id: str
    choices: List[choice]

    def __init__(self, id: str = "", choices: List[choice] = None):
        self.id = id
        self.choices = choices if choices is not None else []

    @classmethod
    def from_dict(cls, data):
        idlocal = data.get("id", "")
        choiceslocal = [choice.from_dict(c) for c in data.get("choices", [])]
        return cls(id=idlocal, choices=choiceslocal)

class responseComplet:
    id: str
    choices: List[choicesCompl]

    def __init__(self, id: str = "", choices: List[choice] = None):
        self.id = id
        self.choices = choices if choices is not None else []

    @classmethod
    def from_dict(cls, data):
        idlocal = data.get("id", "")
        choiceslocal = [choicesCompl.from_dict(c) for c in data.get("choices", [])]
        return cls(id=idlocal, choices=choiceslocal)
    
class responseError:
    error: errorcontent

    def __init__(self, error: errorcontent = None):
        self.error = error if error is not None else errorcontent()
    
    @classmethod
    def from_dict(cls, data):
        errorlocal = errorcontent.from_dict(data.get("error", {}))
        return cls(error=errorlocal)