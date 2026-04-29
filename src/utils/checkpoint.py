import torch
import os
from utils.replaymemory import ReplayMemory


# helper funcion to write data firstly to a temp file so the damn oom-killer
# doesn't ruin everything again
def safe_save(checkpoint, checkpoint_path):
    
    temp_path = checkpoint_path+".temp"

    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
        
    except Exception as e:
        # if something goes wrong clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


class CheckpointHandler:
    """
    class to properly handle checkpoints

    a cphandler object takes an agent as input, iterates over all of its 
    members to find the right types of objects to save and collect them in
    dictionaries that will be saved by torch

    save function can be used to save a full checkpoint containing all the 
    data needed to resume training (including the replay memory)
    or also small a checkpoint that only include the needed network

    it is assumed that an agent with on a policy based method will have a 
    network called "policy_net" and that an agent with a value based method 
    will have a network called "value net" for inference 
    """
    
    def __init__(self, agent):
        self.agent = agent


    def _get_memory(self):
        return self.agent.memory if 'memory' in self.agent.__dict__ else None
        

    def _get_trainable_params(self):
        """
        iterate over all agent's members and extract all isolated nn.Parameters
        """
        params_dict = {}
        for name, value in self.agent.__dict__.items():
            if isinstance(value, torch.nn.Parameter):
                params_dict[name] = value.data.cpu()
                
        return params_dict
    

    def _get_network_states(self, full = True):
        """
        iterate over all agent's members and return a dict of all nn.Modules 
        objects if it is for a full checkpoint or only the network used for 
        inference 
        """
        networks = {}
        for name, obj in self.agent.__dict__.items():
            if isinstance(obj, torch.nn.Module):
                networks[name] = obj.state_dict()

        if full:
            result = networks
        else:
            result = {"policy_net":networks["policy_net"]} if self.agent.policy_method \
                     else {"value_net":networks["value_net"]}

        return result 
    

    def _get_optimizer_states(self):
        """
        iterate over all agent's members and return a dict of all 
        optim.Optimizers objs
        """
        optimizers = {}
        for name, obj in self.agent.__dict__.items():
            if isinstance(obj, torch.optim.Optimizer):
                optimizers[name] = obj.state_dict()

        return optimizers
    

    def save(self, checkpoint_path, full=True):

        checkpoint = {'networks': self._get_network_states(full)}

        if full:
            checkpoint.update({
                'parameters': self._get_trainable_params(),
                'optimizers': self._get_optimizer_states(),
                'memory': self._get_memory()
            })

        safe_save(checkpoint, checkpoint_path)
        print(f"{'checkpoint' if full else 'model'} saved to {checkpoint_path}")
    

    def load(self, checkpoint_path, device):
        """
        load complete state (networks + learned params + optimizers + memory)
        """

        print(f"loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)
        
        # load networks, this is needed for both testing and training
        for name, state_dict in checkpoint['networks'].items():
            if hasattr(self.agent, name):
                network = getattr(self.agent, name)
                if isinstance(network, torch.nn.Module):
                    network.load_state_dict(state_dict)
                    print(f"loaded network: {name}")

        # load learned parameters (temperature, epsilon, etc.)
        if 'parameters' in checkpoint:
            for name, value in checkpoint['parameters'].items():
                if isinstance(value, torch.Tensor):
                    if hasattr(self.agent, name):
                        param = getattr(self.agent, name)
                        if isinstance(param, torch.nn.Parameter):
                            param.data = value.to(device)
                            print(f"loaded parameter: {name}")
        
        # load optimizer states if available
        if 'optimizers' in checkpoint:
            for name, optim_state in checkpoint['optimizers'].items():
                if hasattr(self.agent, name):
                    optimizer = getattr(self.agent, name)
                    if isinstance(optimizer, torch.optim.Optimizer):
                        optimizer.load_state_dict(optim_state)
                        print(f"loaded optimizer: {name}")

        # load memory
        if 'memory' in checkpoint:
            if checkpoint['memory']:
                self.agent.memory = checkpoint['memory']
                print(f"loaded memory - size: {len(self.agent.memory)}")

