"""
-------------------------------------------------------
This file contains program settings and configurations.
-------------------------------------------------------
Author:  Tom LaMantia
Email:   tom@lamantia.mail.utoronto.ca
Version: February 14th 2016
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

INITIAL_STOCK = 12

INITIAL_COST = 0

INITIAL_CURRENT_ORDERS = 0
INITIAL_CURRENT_PIPELINE = 0


CUSTOMER_INITIAL_ORDERS = 4
CUSTOMER_SUBSEQUENT_ORDERS = 8 #used in fixed policy

CUSTOMER_MINIMUM_ORDERS = 20
CUSTOMER_MAXIMUM_ORDERS = 21

TARGET_STOCK = 12

"""
-------------------------------------------------------
RL constants are defined here. 
-------------------------------------------------------
"""
NUM_EPISODES = 100000
NUM_ACTIONS = CUSTOMER_MAXIMUM_ORDERS - CUSTOMER_MINIMUM_ORDERS + 1
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
RANDOM_SEED = 42
