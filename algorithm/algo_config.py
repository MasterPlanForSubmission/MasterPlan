# -*- encoding: utf-8 -*-
'''
@File : algo_config.py
@Describe : parameter for RL training
'''
import torch
GAMMA = 0.99
LAMDA = 0.95
LR_ACTOR = 0.0003 # learning rate for actor network
LR_CRITIC = 0.001 # learning rate for critic network
ACTOR_STEP = 1000
CRITIC_STEP = 1000
K_EPOCHS = 10 # update policy for K epochs in one PPO update
EPS_CLIP = 0.2 # clip parameter for PPO
ENTROPY_L = 0.01
INSTANCES = int(2000e5)
USE_CUDA = False
UPDATE_TAR_INTERVAL = 1000
MIN_TRAIN_STEP = 1000 
PRINT_INTERVAL = 1000
NEG_ON_TIME = 300
NEG_OFF_TIME = 300
POS_WEIGHT = 0.3
NEG_WEIGHT = 1 - POS_WEIGHT
LEARNING_START = 0
STATE_DIM = 17  # state dimension per disk group (power state: 4, serving time: 1, request_size: 6, urgency: 6)

## GRU config
BUFFER_SIZE = 32
EMBEDDING_DIM = 32
MAX_TS_EPISODE = 10  # step length in each episode
GRU_HIDDEN_SIZE = 256
GRU_NUM_LAYERS = 2
BUFFER_INTERVAL = 6  # interval of updating episode buffer
################################## set device##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
