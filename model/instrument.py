class Instrument:
    def __init__(self, name, label, limit):
        self.name = name
        self.label = label
        self.limit = limit

    def __str__(self):
        return self.name
        
        
INSTRUMENTS = (
    Instrument("bass", 0, 80),
    Instrument("snare", 1, 16),
    Instrument("tenor", 2, 96)
)