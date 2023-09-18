# base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, state):
        pass
    
    @abstractmethod
    def learn(self, states, actions, rewards, next_states, dones):
        pass
    
    @abstractmethod
    def save(self, filepath):
        pass
    
    @abstractmethod
    def load(self, filepath):
        pass
