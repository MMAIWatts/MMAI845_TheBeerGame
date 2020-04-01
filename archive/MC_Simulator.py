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

def safe_division(n, d):
    return n / d if d else 0

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

# Initialize actors
theCustomer = Customer()
myRetailer = Retailer(None, wholesalerRetailerTopQueue, wholesalerRetailerBottomQueue, None, theCustomer)

myWholesaler = Wholesaler(wholesalerRetailerTopQueue, distributorWholesalerTopQueue,
                          distributorWholesalerBottomQueue, wholesalerRetailerBottomQueue)

myDistributor = Distributor(distributorWholesalerTopQueue, factoryDistributorTopQueue,
                            factoryDistributorBottomQueue, distributorWholesalerBottomQueue)

myFactory = Factory(factoryDistributorTopQueue, None, None, factoryDistributorBottomQueue, QUEUE_DELAY_WEEKS)


num_episodes = 10000
num_actions = 30
initial_epsilon = 1.0
final_epsilon = 0.01
agent = SupplyChainAgent.MonteCarloAgent(nA=num_actions, num_episodes=num_episodes, epsilon=initial_epsilon)

costs_incurred = []
epsilon_values = []

for i_episode in tqdm(range(num_episodes)):
    # initialize order queues
    for i in range(0, 2):
        wholesalerRetailerTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        wholesalerRetailerBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        distributorWholesalerTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        distributorWholesalerBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        factoryDistributorTopQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        factoryDistributorBottomQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)

    # reset actor states
    agent.reset()
    theCustomer = Customer()
    myRetailer = Retailer(None, wholesalerRetailerTopQueue, wholesalerRetailerBottomQueue, None, theCustomer)

    myWholesaler = Wholesaler(wholesalerRetailerTopQueue, distributorWholesalerTopQueue,
                              distributorWholesalerBottomQueue, wholesalerRetailerBottomQueue)

    myDistributor = Distributor(distributorWholesalerTopQueue, factoryDistributorTopQueue,
                                factoryDistributorBottomQueue, distributorWholesalerBottomQueue)

    myFactory = Factory(factoryDistributorTopQueue, None, None, factoryDistributorBottomQueue, QUEUE_DELAY_WEEKS)

    # decrease exploration over time and save
    new_epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (i_episode / num_episodes) ** 2
    agent.set_epsilon(new_epsilon)
    epsilon_values.append(new_epsilon)

    # run episode
    for thisWeek in range(WEEKS_TO_PLAY):
        # Retailer takes turn, update stats
        myRetailer.TakeTurn(thisWeek)

        # Wholesaler takes turn
        pre_turn_orders = myWholesaler.currentOrders
        myWholesaler.UpdatePreTurn()

        # Store pre-turn state
        state = list((myWholesaler.CalcEffectiveInventory(), myWholesaler.incomingOrdersQueue.data[0],
                      myWholesaler.currentOrders, myWholesaler.currentPipeline))

        # Detemine which action to take
        action = agent.get_next_action(state)

        # Take action
        myWholesaler.TakeTurn(thisWeek, action)

        # Store post-turn state
        state_ = list((myWholesaler.CalcEffectiveInventory(), myWholesaler.incomingOrdersQueue.data[0],
                       myWholesaler.currentOrders, myWholesaler.currentPipeline))

        # Calculate reward
        orders_fulfilled = myWholesaler.outgoingDeliveriesQueue.data[0] # pre_turn_orders - state_[3]
        stock_penalty = myWholesaler.currentStock * STORAGE_COST_PER_UNIT
        backorder_penalty = myWholesaler.currentOrders * BACKORDER_PENALTY_COST_PER_UNIT
        reward = orders_fulfilled - stock_penalty - backorder_penalty
        # reward = 1 - abs(12 - myWholesaler.CalcEffectiveInventory())


        # Store event
        agent.remember(state, action, reward)

        # Distributor takes turn
        myDistributor.TakeTurn(thisWeek)

        # Factory takes turn, update stats
        myFactory.TakeTurn(thisWeek)

    costs_incurred.append(myWholesaler.GetCostIncurred())
    # Update Q table
    agent.learn()

    # save model every 5000 episodes
    if i_episode % 5000 == 0 and i_episode > 0:
        agent.save_model()

# save final agent
agent.save_model()

fig, ax1 = plt.subplots()
ax1.set_xlabel('Episode')
ax1.set_ylabel('Cost Incurred', color='b')
ax1.plot(pd.Series(costs_incurred).rolling(50).mean(), color='b', alpha=0.6)

ax2 = ax1.twinx()
ax2.set_ylabel('Epsilon', color='r')
ax2.plot(epsilon_values, color='r')

fig.tight_layout()
plt.title('Monte Carlo Learning Curve')
plt.show()
fig.savefig('figures/cost_plot.png')

# get best score
best_episode = np.argmin(costs_incurred, axis=0)
best_score = costs_incurred[best_episode]
last_200 = np.mean(costs_incurred[-100])
print('Best score of {}'.format(best_score))
print('Best episode {}'.format(best_episode))
print('Average of last 100 episodes {}'.format(last_200))
