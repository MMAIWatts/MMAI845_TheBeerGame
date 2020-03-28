import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

import SupplyChainAgent
from theBeerGame.Customer import Customer
from theBeerGame.Distributor import Distributor
from theBeerGame.Factory import Factory
from theBeerGame.Retailer import Retailer
from theBeerGame.Settings import *
from theBeerGame.SupplyChainQueue import SupplyChainQueue
from theBeerGame.SupplyChainStatistics import SupplyChainStatistics
from theBeerGame.Wholesaler import Wholesaler

# Initialize SupplyChainQueues
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

num_episodes = 5000
num_actions = 30
initial_epsilon = 1.0
final_epsilon = 0.01
# agent = SupplyChainAgent.MonteCarloAgent(nA=num_actions, num_episodes=num_episodes, epsilon=initial_epsilon)
agent = SupplyChainAgent.DQNAgent(gamma=0.99, epsilon=initial_epsilon, alpha=0.0005, input_dims=4,
                                  n_actions=num_actions, mem_size=1000000, batch_size=52)

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
        # state is a list of (week num, inventory, incoming, outgoing)
        state = list((thisWeek, myWholesaler.currentStock, myWholesaler.currentOrders, myWholesaler.lastOrderQuantity))
        action = agent.get_next_action(state)
        myWholesaler.TakeTurn(thisWeek, action)
        state_ = list((thisWeek, myWholesaler.currentStock, myWholesaler.currentOrders, myWholesaler.lastOrderQuantity))
        reward = -myWholesaler.CalcCostForTurn()
        done = 1 if thisWeek == WEEKS_TO_PLAY else 0
        agent.remember(state, action, reward, state_, done)

        # Distributor takes turn
        myDistributor.TakeTurn(thisWeek)

        # Factory takes turn, update stats
        myFactory.TakeTurn(thisWeek)

    # update Q table
    agent.learn()
    costs_incurred.append(myWholesaler.GetCostIncurred())

    if i_episode % 10 == 0 and i_episode > 0:
        agent.save_model()

fig, ax1 = plt.subplots()
ax1.set_xlabel('Episode')
ax1.set_ylabel('Cost Incurred', color='b')
ax1.plot(pd.Series(costs_incurred).rolling(50).mean(), color='b', alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel('Epsilon', color='r')
ax2.plot(epsilon_values, color='r')

fig.tight_layout()
plt.show()

# get best score
best_episode = np.argmin(costs_incurred, axis=0)[0]
best_score = costs_incurred[best_episode][0]
last_200 = np.mean(costs_incurred[-100][0])
print('Best score of {}'.format(best_score))
print('Best episode {}'.format(best_episode))
print('Average of last 200 episodes {}'.format(last_200))
