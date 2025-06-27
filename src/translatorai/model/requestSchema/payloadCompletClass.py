
class payloadComplet:
    model: str
    prompt: str

    def __init__(self, model: str = "", prompt: str = ""):
        self.model = model
        self.prompt = prompt

    @classmethod
    def from_dict(cls, data):
        model_local = data.get("model", "")
        prompt_local = data.get("prompt", "")
        return cls(model=model_local, prompt=prompt_local)