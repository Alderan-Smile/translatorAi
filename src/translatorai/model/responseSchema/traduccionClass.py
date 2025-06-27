class traduccion:
    texto_original: str
    traduccion: str

    def __init__(self, texto_original: str = "", traduccion: str = ""):
        self.texto_original = texto_original
        self.traduccion = traduccion
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            texto_original=data.get("texto_original", ""),
            traduccion=data.get("traduccion", "")
        )