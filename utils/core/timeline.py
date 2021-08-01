import heapq
import itertools

class WallTime(object):
    """
    A global time object distributed to all workers
    """
    def __init__(self, bucket_size, window_num=1):
        self.curr_time = 0.0
        self.curr_bucket = 0.0
        self.bucket_size = bucket_size
        self.window_size = bucket_size / window_num
        self.timestep = bucket_size

    def update(self, new_time):
        if self.bucket_size == 0:
            self.timestep = new_time - self.curr_time
        self.curr_time = new_time
        # if self.bucket_size!=0:
        #     self.curr_bucket = int(self.curr_time / self.bucket_size)
    def get_now_bucket(self): return self.curr_bucket * self.bucket_size
    def get_next_bucket(self): return (self.curr_bucket+1)*self.bucket_size

    def update_bucket(self, terminal):
        assert self.bucket_size != 0
        self.curr_bucket += 1
        new_time = self.curr_bucket * self.bucket_size
        if new_time < self.curr_time:
            if not terminal: print("Warning: Bucket was skipped!!!!", new_time, self.curr_time)
            self.curr_bucket = int(self.curr_time / self.bucket_size)
            # self.curr_time = self.curr_bucket * self.bucket_size
            # print("Sett curr_time to ", self.curr_time)
        else:
            self.curr_time = new_time
    def update_window(self, plus=True):
        if plus: self.curr_time += self.window_size
        else: self.curr_time -= self.window_size

    def check_bucket(self, new_time):
        now_bucket = int(new_time / self.bucket_size)
        return True if now_bucket == self.curr_bucket else False

    def reset(self):
        self.curr_time = 0.0
        self.curr_bucket = 0.0

class Timeline(object):
    def __init__(self):
        # priority queue
        self.pq = []
        # tie breaker
        self.counter = itertools.count()

    def __len__(self):
        return len(self.pq)

    def peek(self):
        if len(self.pq) > 0:
            (key, counter, item) = self.pq[0]
            # print("peek", key, counter, item)
            return key, item
        else:
            return None, None

    def push(self, key, item):
        heapq.heappush(self.pq,
            (key, next(self.counter), item))
        # print("push", key, self.counter, item)

    def pop(self):
        if len(self.pq) > 0:
            (key, counter, item) = heapq.heappop(self.pq)
            # print("pop", key, counter, item)
            return key, item
        else:
            return None, None
    def to_dict(self):
        dict = {}
        for i in range(len(self.pq)):
            info = self.pq[i]
            if info[2] is None: continue
            dict.update({info[0]:info[2]})
        return dict

    def reset(self):
        self.pq = []
        self.counter = itertools.count()
