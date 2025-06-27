class message:
    role: str
    content: str

    def __init__(self, role: str = "", content: str = ""):
        self.role = role
        self.content = content

    @classmethod
    def from_dict(cls, data):
        return cls(
            role=data.get("role", ""),
            content=data.get("content", "")
        )