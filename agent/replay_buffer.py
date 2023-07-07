from collections import deque
import random
from model.util_nn import *


class DataBuffer(object):
    def __init__(self):
        self.count = 0
        self.buffer = deque()
    def add(self, s, a, r, t, s2, f_a):
        experience = (s, a, r, t, s2, f_a)
        self.buffer.append(experience)
        self.count += 1
    def get(self):
        batch = self.buffer
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        # a2_batch = np.array([_[5] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch
    def clear(self):
        self.buffer.clear()
        self.count = 0

class Experience(object):
    def __init__(self, state, action, reward, terminal, auxtarget=None, last_action=None, last_reward=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.auxtarget = auxtarget
        self.last_action = last_action
        self.last_reward = last_reward

    def get_auxin(self, action_size):
        """
        Return one hot vectored last action + last reward.
        """
        return Experience.concat_action_and_reward(self.last_action, action_size,
                                                        self.last_reward)

    def get_action_reward(self, action_size):
        """
        Return one hot vectored action + reward.
        """
        return Experience.concat_action_and_reward(self.action, action_size,
                                                        self.reward)

    @staticmethod
    def concat_action_and_reward(action, action_size, reward):
        """
        Return one hot vectored action and reward.
        """
        action_reward = np.zeros([action_size + 1])
        action_reward[action] = 1.0
        action_reward[-1] = float(reward)
        return action_reward


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []#deque()
        self._next_idx = 0
        random.seed(random_seed)

    def add_experience(self, experience):
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            # self.pop()
            self.buffer[self._next_idx] = experience
        self._next_idx = (self._next_idx+1) % self.buffer_size

    def add(self, s, s_, a, r, t, s2, s2_, f_a):
        experience = (s, s_, a, r, t, s2, s2_, f_a)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            # self.pop()
            self.buffer[self._next_idx] = experience
        self._next_idx = (self._next_idx+1) % self.buffer_size

    # def pop(self):
    #     self.buffer.popleft()
    #     self.count -= 1

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        return self.unwrap(batch)

    def sample_sequence(self, recent=True, seq_size=None):
        batch = []
        if recent:
            idx = (self._next_idx-1)%self.buffer_size
            batch.append(self.buffer[idx])
            while self.buffer[idx][4] is False:
                idx = (idx - 1) % self.count
                batch.append(self.buffer[idx])
        else:
            idx = random.randint(0, self.count - 1)
            batch.append(self.buffer[idx])
            while self.buffer[idx][4] is False and (seq_size>1 if seq_size is not None else True):
                if seq_size is not None: seq_size -= 1
                idx = (idx + 1) % self.count # not self.buffer_count
                batch.append(self.buffer[idx])
        return self.unwrap(batch)

    def unwrap(self, batch):
        if type(batch[0])==Experience:
            return batch
        else:
            s_batch = np.array([_[0] for _ in batch])
            xs_batch = np.array([_[1] for _ in batch])
            a_batch = np.array([_[2] for _ in batch])
            r_batch = np.array([_[3] for _ in batch])
            t_batch = np.array([_[4] for _ in batch])
            s2_batch = np.array([_[5] for _ in batch])
            xs2_batch = np.array([_[6] for _ in batch])
            f_a_batch = np.array([_[7] for _ in batch], dtype=object) #feasible_action_list has different length across experiences
            # a2_batch = np.array([_[5] for _ in batch])
            return s_batch, xs_batch, a_batch, r_batch, t_batch, s2_batch, xs2_batch, f_a_batch

    def clear(self):
        self.deque.clear()
        self.count = 0

class FlatteningReplayBuffer(object):
    def __init__(self, buffer_size, buffer_num, random_seed=123):
        self.set = list()
        self.buffer_size = buffer_size
        self.buffer_num = buffer_num
        self.count = 0
        for i in range(buffer_num):
            self.set.append(ReplayBuffer(buffer_size))
    def add(self, s, a, r, t, s2, f_a):
        action = 0
        for i in range(len(a)):
            if a[i] == 1: action = i
        buffer = self.set[action]
        if self.count < self.buffer_size:
            self.count += 1
        else:
            buffer.pop()
            self.count -= 1
        buffer.add(s, a, r, t, s2, f_a)
    def sample_batch(self, batch_size, flattenFlag : bool = True):

        buffer_num = len(self.set)

        per_size = int(batch_size / buffer_num)
        remainder = batch_size % buffer_num
        s_batch, a_batch, r_batch, t_batch, s2_batch, f_a_batch = self.set[0].sample_batch(per_size+remainder)
        for i in range(1, buffer_num):
            buffer = self.set[i]
            t1,t2,t3,t4,t5,t6=buffer.sample_batch(per_size)
            s_batch = np.concatenate((s_batch, t1))
            a_batch = np.concatenate((a_batch, t2))
            r_batch = np.concatenate((r_batch, t3))
            t_batch = np.concatenate((t_batch, t4))
            s2_batch = np.concatenate((s2_batch, t5))
            temp = f_a_batch.tolist()
            temp.extend(t6.tolist())
            f_a_batch = np.array(temp)

        return s_batch, a_batch, r_batch, t_batch, s2_batch, f_a_batch
