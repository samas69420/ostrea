## TODOs

#### general
    
    * remove unnecessary gradients
    * better logging
    * better comments
    * refine how many dimensions are handled
    * add batch normalization
    * move the call to update inside the agent 
    * record multiple episodes in one file during testing
    * support to other network types other than mlp
    * type hints
    * add support to asymmetric action spaces in bounded algos

#### sac
    
    * use the main q net as second net
    * state indipendent variance
    * optimize update
    * sample multiple next actions to bootstrap

#### ddpg
    
    * better strategies to add noise

#### dql
    
    * choice for decay
    * other types of decay
    * soft update for target nets
    * epsilon in the checkpoint 

#### ppo
    
    * rewrite batching to mix not only timesteps but also environments
    * entropy augmentation 
