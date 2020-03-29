import matplotlib.pyplot as plt
import pandas as pd
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

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# set numpy options
# np.random.seed(RANDOM_SEED)

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

BACKORDER_PENALTY_COST_PER_UNIT = 1.0
num_episodes = 10000
num_actions = 30
initial_epsilon = 1.0
final_epsilon = 0.01
agent = SupplyChainAgent.DQNAgent(gamma=0.99, epsilon=initial_epsilon, alpha=0.0005, input_dims=4,
                                  n_actions=num_actions, mem_size=10000, batch_size=52)

agent.load_model()
df = pd.DataFrame(columns=['Week', 'currentStock', 'currentOrders', 'lastOrderQuantity', 'currentPipeline',
                           'action', 'reward', 'cumulativeReward', 'done'])

weeks = []
currentStock = []
currentOrders = []
lastOrderQuantity = []
currentPipeline = []
actions = []
rewards = []
cumulativeReward = []
dones = []

# run episode
for thisWeek in range(WEEKS_TO_PLAY):
    # Retailer takes turn, update stats
    myRetailer.TakeTurn(thisWeek)
    myStats.RecordRetailerCost(myRetailer.GetCostIncurred())
    myStats.RecordRetailerOrders(myRetailer.GetLastOrderQuantity())
    myStats.RecordRetailerEffectiveInventory(myRetailer.CalcEffectiveInventory())

    # Wholesaler takes turn
    myWholesaler.UpdatePreTurn()

    # State is a list of (week num, inventory, incoming, outgoing)
    state = list((thisWeek, myWholesaler.CalcEffectiveInventory(), myWholesaler.incomingOrdersQueue.data[0],
                  myWholesaler.currentOrders, myWholesaler.currentPipeline))

    # Decide which action to take
    action = agent.get_next_action(state)

    # Record state stats
    weeks.append(thisWeek)
    currentStock.append(myWholesaler.CalcEffectiveInventory())
    currentOrders.append(myWholesaler.incomingOrdersQueue.data[0])
    lastOrderQuantity.append(myWholesaler.lastOrderQuantity)
    currentPipeline.append(myWholesaler.currentPipeline)

    # Take action
    myWholesaler.TakeTurn(thisWeek, action)

    # Record post-turn state
    state_ = list((thisWeek, myWholesaler.CalcEffectiveInventory(), myWholesaler.incomingOrdersQueue.data[0],
                  myWholesaler.currentOrders, myWholesaler.currentPipeline))

    # calculate reward
    orders_fulfilled = state[2] - state_[2]
    reward = orders_fulfilled - myWholesaler.CalcCostForTurn()
    done = 1 if thisWeek == WEEKS_TO_PLAY - 1 else 0

    # Record stats
    actions.append(action)
    rewards.append(reward)
    cumulativeReward.append(myWholesaler.GetCostIncurred())
    dones.append(done)

    myStats.RecordWholesalerCost(myWholesaler.GetCostIncurred())
    myStats.RecordWholesalerOrders(myWholesaler.GetLastOrderQuantity())
    myStats.RecordWholesalerEffectiveInventory(myWholesaler.CalcEffectiveInventory())

    # Distributor takes turn
    myDistributor.TakeTurn(thisWeek)
    myStats.RecordDistributorCost(myDistributor.GetCostIncurred())
    myStats.RecordDistributorOrders(myDistributor.GetLastOrderQuantity())
    myStats.RecordDistributorEffectiveInventory(myDistributor.CalcEffectiveInventory())

    # Factory takes turn, update stats
    myFactory.TakeTurn(thisWeek)
    myStats.RecordFactoryCost(myFactory.GetCostIncurred())
    myStats.RecordFactoryOrders(myFactory.GetLastOrderQuantity())
    myStats.RecordFactoryEffectiveInventory(myFactory.CalcEffectiveInventory())

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
df['cumulativeReward'] = cumulativeReward
df['done'] = dones
df.to_csv('test1.csv')
print(df)

