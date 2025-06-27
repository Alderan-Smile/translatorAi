class errorcontent:
    code: int
    message: str

    def __init__(self, code: int = 0, message: str = ""):
        self.code = code
        self.message = message
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            code=data.get("code", 0),
            message=data.get("message", "")
        )