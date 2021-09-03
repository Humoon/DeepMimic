class Moni:

    def __init__(self):
        self.data = {}

    def record(self, d):
        for key, value in d.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def result(self, msg_type=None):
        result = self.data.copy()
        if msg_type is not None:
            result['msg_type'] = msg_type
        self.data = {}
        return result
