## TODOs

#### general
    
    * add support to save and load complex models (multiple networks, other params etc)
    * remove unnecessary gradients
    * better logging
    * better comments
    * refine how many dimensions are handled
    * choice for done signal
    * add batch normalization
    * move the call to update inside the agent 
    * load only policy net for test with policy methods
    * record multiple episodes in one file during testing
    * support to other network types other than mlp
    * type hints

#### sac
    
    * second q net
    * use the main q net as second net
    * state indipendent variance
    * optimize update
    * add a switch for dynamic or fixed temperature
    * add a choice for target entropy computation
    * fix qnet name
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
