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
                N_ENV = 64,
                CHECKPOINT_SAVE_FREQ = 100000,   # after how many steps the full checkpoint should be saved during training
                CHECKPOINT_NAME = "ckpt.pt",     # name used to save the full training checkpoint (WARNING: it will also contain the ReplayMemory, make sure you have enough ram)
                MODEL_NAME = "model.pt",         # name used to save the inference model, only saved when a new best score is reached
                DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),

                # agent parameters 

                WARMUP = 3000,                   # warmup steps without any update
                MEMORY_MAXLEN = 500_000,
                MEMORY_BATCH_SIZE = 256,
                GRADIENT_STEPS = 100,            # how many gradient steps should be done in the update function, same value as buffer size to have a updates/data ratio close to 1
                VALUE_LR =  1e-3,
                POLICY_LR = 1e-3,
                NOISE_MAG = 0.2,                 # noise magnitude for exploration
                TAU = 0.005,                     # soft update parameter for target nets
                POLICY_METHOD = True,
                ALGO_NAME = "ddpg")
