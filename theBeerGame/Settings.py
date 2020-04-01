"""
-------------------------------------------------------
This file contains program settings and configurations.
-------------------------------------------------------
Adapted from Tom LaMantia's code
-------------------------------------------------------
"""

"""
-------------------------------------------------------
Beer Game constants are defined here. 
-------------------------------------------------------
"""

STORAGE_COST_PER_UNIT = 0.5
BACKORDER_PENALTY_COST_PER_UNIT = 1

#We can play the full game since no actor is programmed to dump stock near end of game
WEEKS_TO_PLAY = 52

QUEUE_DELAY_WEEKS = 2

INITIAL_STOCK = 12 #TODO better to change it to 4 ???

INITIAL_COST = 0

INITIAL_CURRENT_ORDERS = 0
INITIAL_CURRENT_PIPELINE = 0


CUSTOMER_MINIMUM_ORDERS = 4
CUSTOMER_MAXIMUM_ORDERS = 4 

CUSTOMER_INITIAL_ORDERS = 4
CUSTOMER_SUBSEQUENT_ORDERS = 8 #used in fixed policy

TARGET_STOCK = 12

"""
-------------------------------------------------------
RL constants are defined here. 
-------------------------------------------------------
"""
#MC and DQN Parameters:
NUM_ACTIONS = 30
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
RANDOM_SEED = 42

EPSILON_METHOD = 'exp' # 'exp' or 'quad' 
"""
exp: new_epsilon = INITIAL_EPSILON * (epsilon_decay ** i_episode)
quad: new_epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (i_episode / num_episodes) ** 2

"""
EPSILON_DECAY = 0.9996 #used when EPSILON_METHOD = 'exp'


REWARD_METHOD = 'func1' #'func1' or 'func2' or 'func3'
"""
func1: reward = orders_fulfilled - stock_penalty - backorder_penalty
func2: reward =  - stock_penalty - backorder_penalty
func3: reward = 1 - abs(12 - myWholesaler.CalcEffectiveInventory())
"""


#DQN Parameters Only:
GAMMA=0.99
ALPHA=0.0005
MEM_SIZE=10000
BATCH_SIZE=52

# Create a str from all Parameters:
if EPSILON_METHOD == 'exp':
    DQN_PARAMS = f'G{GAMMA}_A{ALPHA}_MEM{MEM_SIZE}_BS{BATCH_SIZE}_EPS{EPSILON_METHOD}{EPSILON_DECAY}_R{REWARD_METHOD}'
    MC_PARAMS = f'EPS{EPSILON_METHOD}{EPSILON_DECAY}_R{REWARD_METHOD}'
else:
    DQN_PARAMS = f'G{GAMMA}_A{ALPHA}_MEM{MEM_SIZE}_BS{BATCH_SIZE}_EPS{EPSILON_METHOD}_R{REWARD_METHOD}'
    MC_PARAMS = f'EPS{EPSILON_METHOD}_R{REWARD_METHOD}'
