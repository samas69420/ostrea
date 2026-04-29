import torch
from utils.parameters import Params

params = Params(# environment/general training parameters 

                SEED = None,                     # seed used with torch
                MAX_TRAINING_STEPS = 100e6,      # 100M
                BUFFER_SIZE = 1000,              # size of episode buffer that triggers the update
                PRINT_FREQ_STEPS = 1000,         # after how many steps the logs should be printed during training
                UPDATE_PLOT_SAVE_FREQ = 10,      # after how many updates the avg return plot should be saved 
                N_EVAL_EPISODES = 10,            # how many episodes should be used for evaluation during training 
                GAMMA = 0.99,
                N_ENV = 64,
                CHECKPOINT_SAVE_FREQ = 50000,    # after how many steps the full checkpoint should be saved during training
                CHECKPOINT_NAME = "ckpt.pt",     # name used to save the full training checkpoint 
                MODEL_NAME = "model.pt",         # name used to save the inference model, only saved when a new best score is reached
                DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),

                # agent parameters 

                PPO_EPS = 5e-2,                  # threshold for the ratio function
                SEPARATE_COV_PARAMS = True,      # if cov matrix should not be computed by the policy net
                DIAGONAL_COV_MATRIX = True,      # learn a diagonal or full cov matrix
                MIN_COV = 1e-2,                  # minimum value allowed for diagonal cov matrix
                VALUE_EPOCHS = 10,
                POLICY_EPOCHS = 10,
                VALUE_BATCH_SIZE = 128,          # for now these batches are made only along the time dimension
                POLICY_BATCH_SIZE = 128,         
                VALUE_LR = 3e-2,
                POLICY_LR = 1e-2,
                NUMERICAL_EPSILON = 1e-7,        # value for numerical stability
                BETA = 5e-3,                     # weight used for entropy
                ADVANTAGE_TYPE = "GAE",          # type of advantages GAE/TD/MC
                GAE_LAMBDA = 0.99,
                POLICY_METHOD = True,
                ALGO_NAME = "ppo")
