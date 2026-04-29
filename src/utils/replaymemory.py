import random
from collections import deque

class ReplayMemory:
    
    def __init__(self, maxlen):
        
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):

        return len(self.buffer)

    def append(self, element):

        self.buffer.append(element)

    def sample(self, n):

        return random.sample(self.buffer, n)

