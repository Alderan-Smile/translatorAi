class header:
    Authorization: str
    Content_Type: str

    def __init__(self, Authorization: str = "", Content_Type: str = ""):
        self.Authorization = Authorization
        self.Content_Type = Content_Type
    
    @classmethod
    def from_dict(cls, data):
        Authorizationlocal = data.get("Authorization", "")
        Content_Typelocal = data.get("Content-Type", "")
        return cls(Authorization=Authorizationlocal, Content_Type=Content_Typelocal)