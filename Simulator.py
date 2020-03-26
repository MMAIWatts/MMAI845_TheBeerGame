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

"""
-------------------------------------------------------
Given two SupplyChainActors B <--> A, where
A is higher in the supply chain, let "top queue" denote A's
outgoingOrderQueue/B's incomingOrderQueue. Let "bottom queue"
denote B's outgoingDeliveryQueue/A's incoming delivery queue. 
-------------------------------------------------------
"""
wholesalerRetailerTopQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)
wholesalerRetailerBottomQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)

distributorWholesalerTopQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)
distributorWholesalerBottomQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)

factoryDistributorTopQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)
factoryDistributorBottomQueue = SupplyChainQueue(QUEUE_DELAY_WEEKS)

"""
-------------------------------------------------------
Each queue should have at least 2 orders of size CUSTOMER_INITIAL_ORDER 
-------------------------------------------------------
"""
for i in range(0, 2):
    wholesalerRetailerTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    wholesalerRetailerBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    distributorWholesalerTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    distributorWholesalerBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    factoryDistributorTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
    factoryDistributorBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)

"""
-------------------------------------------------------
Now we initialize our SupplyChainObjects. Passing the correct
queues is essential.
-------------------------------------------------------
"""

# theCustomer = Customer()
# myRetailer = Retailer(None, wholesalerRetailerTopQueue, wholesalerRetailerBottomQueue, None, theCustomer)
#
# myWholesaler = Wholesaler(wholesalerRetailerTopQueue, distributorWholesalerTopQueue,
#                           distributorWholesalerBottomQueue, wholesalerRetailerBottomQueue)
#
# myDistributor = Distributor(distributorWholesalerTopQueue, factoryDistributorTopQueue,
#                             factoryDistributorBottomQueue, distributorWholesalerBottomQueue)
#
# myFactory = Factory(factoryDistributorTopQueue, None, None, factoryDistributorBottomQueue, QUEUE_DELAY_WEEKS)

# Initialize Statistics object
myStats = SupplyChainStatistics()

"""
-------------------------------------------------------
Main game-play!
-------------------------------------------------------
"""
num_episodes = 20000
num_actions = 30
initial_epsilon = 0.9
final_epsilon = 0.01
agent = SupplyChainAgent.MonteCarloAgent(nA=num_actions, num_episodes=num_episodes, epsilon=initial_epsilon)

costs_incurred = []

for i_episode in tqdm(range(num_episodes)):

    theCustomer = Customer()
    myRetailer = Retailer(None, wholesalerRetailerTopQueue, wholesalerRetailerBottomQueue, None, theCustomer)

    myWholesaler = Wholesaler(wholesalerRetailerTopQueue, distributorWholesalerTopQueue,
                              distributorWholesalerBottomQueue, wholesalerRetailerBottomQueue)

    myDistributor = Distributor(distributorWholesalerTopQueue, factoryDistributorTopQueue,
                                factoryDistributorBottomQueue, distributorWholesalerBottomQueue)

    myFactory = Factory(factoryDistributorTopQueue, None, None, factoryDistributorBottomQueue, QUEUE_DELAY_WEEKS)

    # decrease exploration over time
    new_epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (i_episode / num_episodes)
    agent.set_epsilon(new_epsilon)

    episode = []
    for thisWeek in range(WEEKS_TO_PLAY):

        # Retailer takes turn, update stats
        myRetailer.TakeTurn(thisWeek)

        # Wholesaler takes turn
        # state is a tuple of (inventory, incoming, outgoing)
        state = thisWeek, myWholesaler.currentStock, myWholesaler.currentOrders, myWholesaler.lastOrderQuantity
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

plt.plot(costs_incurred)
plt.xlabel('Episode')
plt.ylabel('Cost Incurred')
plt.show()

