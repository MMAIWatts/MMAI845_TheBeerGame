"""
-------------------------------------------------------
The main program. The main program handles the simulation
by co-ordinating the game. This involves incrementing all of the
communication queues between the different parts of the system.

The main program also acts as the customer, receiving the
product at the end of the supply chain system.
-------------------------------------------------------
Author:  Tom LaMantia
Email:   tom@lamantia.mail.utoronto.ca
Version: February 14th 2016
-------------------------------------------------------
"""
from theBeerGame.Settings import *
from theBeerGame.Customer import Customer
from theBeerGame.SupplyChainQueue import SupplyChainQueue
from theBeerGame.Retailer import Retailer
from theBeerGame.Wholesaler import Wholesaler
from theBeerGame.Distributor import Distributor
from theBeerGame.Factory import Factory
from theBeerGame.SupplyChainStatistics import SupplyChainStatistics

import SupplyChainAgent
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def roundint(value, base=5):
    return int(value) - int(value) % int(base)

wholesalerRetailerTopQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)
wholesalerRetailerBottomQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)

distributorWholesalerTopQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)
distributorWholesalerBottomQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)

factoryDistributorTopQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)
factoryDistributorBottomQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)

for i in range(0, 2):
    wholesalerRetailerTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    wholesalerRetailerBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    distributorWholesalerTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    distributorWholesalerBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    factoryDistributorTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    factoryDistributorBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)


# Initialize Statistics object
myStats = SupplyChainStatistics()

num_episodes = 500000
num_actions = 20
initial_epsilon = 0.5
final_epsilon = 0.01
agent = SupplyChainAgent.MonteCarloAgent(nA=num_actions, num_episodes=num_episodes, epsilon=initial_epsilon)

costs_incurred = []
epsilon_values = []

for i_episode in tqdm(range(num_episodes)):

    theCustomer = Customer()
    myRetailer = Retailer(None, wholesalerRetailerTopQueue, wholesalerRetailerBottomQueue, None, theCustomer)

    myWholesaler = Wholesaler(wholesalerRetailerTopQueue, distributorWholesalerTopQueue,
                              distributorWholesalerBottomQueue, wholesalerRetailerBottomQueue)

    myDistributor = Distributor(distributorWholesalerTopQueue, factoryDistributorTopQueue,
                                factoryDistributorBottomQueue, distributorWholesalerBottomQueue)

    myFactory = Factory(factoryDistributorTopQueue, None, None, factoryDistributorBottomQueue, QUEUE_DELAY_WEEKS)

    # decrease exploration over time
    new_epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (i_episode / num_episodes) ** 2
    agent.set_epsilon(new_epsilon)
    epsilon_values.append(new_epsilon)

    episode = []
    for thisWeek in range(WEEKS_TO_PLAY):

        # Retailer takes turn, update stats
        myRetailer.TakeTurn(thisWeek)

        # Wholesaler takes turn
        # state is a tuple of (inventory, incoming, outgoing)
        state = roundint(myWholesaler.currentStock, 2), roundint(myWholesaler.currentOrders, 2), \
                roundint(myWholesaler.lastOrderQuantity, 2)
        action = agent.get_next_action(state)
        myWholesaler.TakeTurn(thisWeek, action)
        reward = -myWholesaler.CalcCostForTurn()
        episode.append((state, action, reward))

        # Distributor takes turn
        myDistributor.TakeTurn(thisWeek)

        # Factory takes turn, update stats
        myFactory.TakeTurn(thisWeek)

    # update Q table
    agent.update_Q(episode)
    costs_incurred.append(myWholesaler.GetCostIncurred())

fig, ax1 = plt.subplots()
ax1.set_xlabel('Episode')
ax1.set_ylabel('Cost Incurred', color='b')
ax1.plot(pd.Series(costs_incurred).rolling(50).mean(), color='b', alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel('Epsilon', color='r')
ax2.plot(epsilon_values, color='r')

fig.tight_layout()
plt.show()

