import torch
from utils.parameters import Params

params = Params(# environment/general training parameters 

                SEED = None,                     # seed used with torch
                MAX_TRAINING_STEPS = 100e6,      # 100M
                BUFFER_SIZE = 100,               # size of episode buffer that triggers the update
                PRINT_FREQ_STEPS = 100,          # after how many steps the logs should be printed during training       
                UPDATE_PLOT_SAVE_FREQ = 100,     # after how many updates the avg return plot should be saved 
                N_EVAL_EPISODES = 10,            # how many episodes should be used for evaluation during training 
                GAMMA = 0.99,
                N_ENV = 32,
                CHECKPOINT_SAVE_FREQ = 100000,   # after how many steps the full checkpoint should be saved during training
                CHECKPOINT_NAME = "ckpt.pt",     # name used to save the full training checkpoint (WARNING: it will also contain the ReplayMemory, make sure you have enough ram)
                MODEL_NAME = "model.pt",         # name used to save the inference model, only saved when a new best score is reached
                DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu"),

                # agent parameters 

                WARMUP = 10000,                  # warmup steps without any update
                MEMORY_MAXLEN = 500_000,
                MEMORY_BATCH_SIZE = 256,
                GRADIENT_STEPS = 100,            # how many gradient steps should be done in the update function, same value as buffer size to have a updates/data ratio close to 1
                NUMERICAL_EPSILON = 1e-6,        # small value for numerical stabilty
                TARGET_H = "auto",               # target entropy, if "auto" it will be set to -|A| for continuous actions and to 0.5*log(|A|) (half the max value) for discrete actions
                ALPHA = "auto",                  # fixed or learnable temperature, if "auto" it will be learned and tuned automatically during training
                VALUE_LR =  3e-4,
                POLICY_LR = 3e-4,
                ALPHA_LR =  1e-3,
                TAU = 0.01,                      # soft update parameter for target nets
                MAX_LOGVAR = 2.,
                MIN_LOGVAR = -20.,
                USE_DOUBLE_Q_NET = True,         # more memory usage since two more networks will be trained (5 nets total) but more accurate Qvalues estimation and usually faster convergence
                POLICY_METHOD = True,
                ALGO_NAME = "sac")
