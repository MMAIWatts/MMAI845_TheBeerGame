import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import time

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
# agent = SupplyChainAgent.MonteCarloAgent(nA=num_actions, num_episodes=num_episodes, epsilon=initial_epsilon)
agent = SupplyChainAgent.DQNAgent(gamma=0.99, epsilon=initial_epsilon, alpha=0.0005, input_dims=5,
                                  n_actions=num_actions, mem_size=100000, batch_size=236)


agent.load_model()
df = pd.DataFrame(columns=['Week', 'currentStock', 'currentOrders', 'lastOrderQuantity', 'currentPipeline',
                           'action', 'reward', 'done'])

weeks = []
currentStock = []
currentOrders = []
lastOrderQuantity = []
currentPipeline = []
actions = []
rewards = []
dones = []

# run episode
for thisWeek in range(WEEKS_TO_PLAY):
    # Retailer takes turn, update stats
    myRetailer.TakeTurn(thisWeek)
    myStats.RecordRetailerCost(myRetailer.GetCostIncurred())
    myStats.RecordRetailerOrders(myRetailer.GetLastOrderQuantity())
    myStats.RecordRetailerEffectiveInventory(myRetailer.CalcEffectiveInventory())
    print("Retailer Complete")

    # Wholesaler takes turn
    # state is a list of (week num, inventory, incoming, outgoing)
    state = list((thisWeek, myWholesaler.currentStock, myWholesaler.currentOrders,
                  myWholesaler.lastOrderQuantity, myWholesaler.currentPipeline))
    action = agent.get_next_action(state)

    # record state
    weeks.append(thisWeek)
    currentStock.append(myWholesaler.currentStock)
    currentOrders.append(myWholesaler.currentOrders)
    lastOrderQuantity.append(myWholesaler.lastOrderQuantity)
    currentPipeline.append(myWholesaler.currentPipeline)

    myWholesaler.TakeTurn(thisWeek, action)
    reward = -myWholesaler.CalcCostForTurn()
    done = 1 if thisWeek == WEEKS_TO_PLAY - 1 else 0

    # record stats
    actions.append(action)
    rewards.append(reward)
    dones.append(done)

    myStats.RecordWholesalerCost(myWholesaler.GetCostIncurred())
    myStats.RecordWholesalerOrders(myWholesaler.GetLastOrderQuantity())
    myStats.RecordWholesalerEffectiveInventory(myWholesaler.CalcEffectiveInventory())
    print("Wholesaler Complete")


    # Distributor takes turn
    myDistributor.TakeTurn(thisWeek)
    myStats.RecordDistributorCost(myDistributor.GetCostIncurred())
    myStats.RecordDistributorOrders(myDistributor.GetLastOrderQuantity())
    myStats.RecordDistributorEffectiveInventory(myDistributor.CalcEffectiveInventory())
    print("Distributor Complete")

    # Factory takes turn, update stats
    myFactory.TakeTurn(thisWeek)
    myStats.RecordFactoryCost(myFactory.GetCostIncurred())
    myStats.RecordFactoryOrders(myFactory.GetLastOrderQuantity())
    myStats.RecordFactoryEffectiveInventory(myFactory.CalcEffectiveInventory())
    print("Factory Complete")

print("--- Final Statistics ----")
print("Beer received by customer: {0}".format(theCustomer.GetBeerReceived()))
myStats.PlotCosts()
myStats.PlotOrders()
myStats.PlotEffectiveInventory()

df['Week'] = weeks
df['currentStock'] = currentStock
df['currentOrders'] = currentOrders
df['lastOrderQuantity'] = lastOrderQuantity
df['currentPipeline'] = currentPipeline
df['action'] = actions
df['reward'] = rewards
df['done'] = dones
df.to_csv('test1.csv')
print(df)
