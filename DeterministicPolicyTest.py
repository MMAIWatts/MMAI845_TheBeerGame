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

num_episodes = 10000
num_actions = 30
initial_epsilon = 1.0
final_epsilon = 0.01
agent = SupplyChainAgent.MonteCarloAgent(nA=num_actions, num_episodes=num_episodes, epsilon=initial_epsilon)

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
    print("Retailer Complete")

    # Wholesaler takes turn
    myWholesaler.UpdatePreTurn()

    # Store pre-turn state
    state = list((myWholesaler.CalcEffectiveInventory(), myWholesaler.incomingOrdersQueue.data[0],
                  myWholesaler.currentOrders, myWholesaler.currentPipeline))

    # Detemine which action to take
    target_inventory = 6
    target_pipeline = 2*myWholesaler.incomingOrdersQueue.data[0]
    action = myWholesaler.incomingOrdersQueue.data[0] + (target_inventory - myWholesaler.currentStock + target_pipeline - myWholesaler.currentPipeline)

    # record state
    weeks.append(thisWeek)
    currentStock.append(myWholesaler.CalcEffectiveInventory())
    currentOrders.append(myWholesaler.incomingOrdersQueue.data[0])
    lastOrderQuantity.append(myWholesaler.lastOrderQuantity)
    currentPipeline.append(myWholesaler.currentPipeline)

    # Take action
    pre_turn_orders = myWholesaler.currentOrders
    if action > 0:
        myWholesaler.TakeTurn(thisWeek, action)
    else:
        myWholesaler.TakeTurn(thisWeek, 0)


    # Store post-turn state
    state_ = list((myWholesaler.CalcEffectiveInventory(), myWholesaler.incomingOrdersQueue.data[0],
                   myWholesaler.currentOrders, myWholesaler.currentPipeline))

    # Calculate reward
    orders_fulfilled = state[2] - state_[2]
    stock_penalty = myWholesaler.currentStock * STORAGE_COST_PER_UNIT
    backorder_penalty = myWholesaler.currentOrders * BACKORDER_PENALTY_COST_PER_UNIT
    reward = orders_fulfilled - stock_penalty - backorder_penalty
    done = 1 if thisWeek == WEEKS_TO_PLAY - 1 else 0

    # record stats
    actions.append(action)
    rewards.append(reward)
    cumulativeReward.append(myWholesaler.GetCostIncurred())
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
df['cumulativeReward'] = cumulativeReward
df['done'] = dones
df.to_csv('test1.csv')
print(df)

