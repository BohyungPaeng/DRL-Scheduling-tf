class Job(object):
    def __init__(self, idx, type, pt, arrival, due = 0, late=100000):
        self.idx = idx
        self.type = type
        self.pt = pt
        self.due = due
        self.late = late
        self.arrival = arrival
        self.mac = -1
        self.st = 0