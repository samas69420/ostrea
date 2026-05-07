from abc import ABC, abstractmethod


class BaseAgent(ABC):
    

    @abstractmethod
    def choose_action(self, observation):
        """ 
        takes the current observarion and computes the next action
        to take including exploration during training
        """


    @abstractmethod
    def choose_action_greedy(self, observation):
        """
        takes the current observarion and computes the next action
        to take but without exploration, used for inference and evaluation
        """


    @abstractmethod
    def update(self):
        """
        update the learnable parameters according to the current learning algo
        """


    def save_checkpoint(self, checkpoint_path):
        """
        save the full training checkpoint
        """
        self.checkpoint_handler.save(checkpoint_path, full=True)

    def save_model(self, checkpoint_path):
        """
        save only the inference model 
        """
        self.checkpoint_handler.save(checkpoint_path, full=False)

    def load_checkpoint(self, checkpoint_path, device):
        """
        used for both training checkpoints and inference models
        """
        self.checkpoint_handler.load(checkpoint_path, device)
