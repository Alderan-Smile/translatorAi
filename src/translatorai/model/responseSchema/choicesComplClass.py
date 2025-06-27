class choicesCompl:
    text: str
    index: int
    finish_reason: str

    def __init__(self, text: str = "", index: int = 0, finish_reason: str = ""):
        self.text = text
        self.index = index
        self.finish_reason = finish_reason
    
    @classmethod
    def from_dict(cls, data):
        textlocal = data.get("text", "")
        indexlocal = data.get("index", 0)
        finish_reasonlocal = data.get("finish_reason", "")
        return cls(text=textlocal, index=indexlocal, finish_reason=finish_reasonlocal)