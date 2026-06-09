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
    * update readme with stuff about custom environments
    * enable bootstrapping for truncated episodes in on-policy algos
    * better testing, ideally including all possible variants and input/output types
    * continual backprop

#### sac
    
    * use the main q net as second net
    * state indipendent variance
    * optimize update
    * sample multiple next actions to bootstrap
    * add standardization (welford)

#### ddpg
    
    * better strategies to add noise
    * add encoder
    * add standardization/normalization

#### dql
    
    * choice for decay
    * other types of decay
    * soft update for target nets
    * epsilon in the checkpoint 
    * add encoder
    * add standardization/normalization

#### ppo
    
    * rewrite batching to mix not only timesteps but also environments
    * entropy augmentation 
    * add standardization (welford)
    * improve normalization

#### vpg
    
    * squash action
    * add encoder
    * add standardization/normalization
