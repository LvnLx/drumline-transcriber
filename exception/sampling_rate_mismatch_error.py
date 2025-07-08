from model.instrument import INSTRUMENTS

class SamplingRateMismatchError(Exception):
    def __init__(self, instrument_1, instrument_2, clip_1, clip_2, sr_1, sr_2):
        self.instrument_1 = INSTRUMENTS[instrument_1]
        self.instrument_2 = INSTRUMENTS[instrument_2]
        self.clip_1 = "clip_{}".format(clip_1)
        self.clip_2 = "clip_{}".format(clip_2)
        self.sr_1 = sr_1
        self.sr_2 = sr_2

    def __str__(self):
        return "{} {} and {} {} do not have matching sampling rates ({} and {})".format(self.instrument_1, self.clip_1, self.instrument_2, self.clip_2, self.sr_1, self.sr_2)