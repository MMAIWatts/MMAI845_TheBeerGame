import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


import SupplyChainAgent
from theBeerGame.Customer import Customer
from theBeerGame.Distributor import Distributor
from theBeerGame.Factory import Factory
from theBeerGame.Retailer import Retailer
from theBeerGame.Wholesaler import Wholesaler

#from theBeerGame.Settings import *
from Settings import *
from theBeerGame.SupplyChainQueue import SupplyChainQueue
from theBeerGame.SupplyChainStatistics import SupplyChainStatistics


#some utils functions
def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def safe_division(n, d):
    return n / d if d else 0


def decrease_epsilon(i_episode, num_episodes, method = EPSILON_METHOD, epsilon_decay = EPSILON_DECAY):
    '''
    decrease exploration over time
    '''
    if method == 'quad':
        new_epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (i_episode / num_episodes) ** 2

    elif method == 'exp':

        new_epsilon = INITIAL_EPSILON * (epsilon_decay ** i_episode)

        if new_epsilon < FINAL_EPSILON:
            new_epsilon = FINAL_EPSILON

        #TODO add more methods - like step function divided by 2 every 1000 eps
   
    else: 
        raise ValueError('invaild epsilon method')
    
    return new_epsilon



# Initialize SupplyChainQueues
def Initialize_actors():
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

    customer = Customer()
    retailer = Retailer(None, wholesalerRetailerTopQueue, wholesalerRetailerBottomQueue, None, customer)

    wholesaler = Wholesaler(wholesalerRetailerTopQueue, distributorWholesalerTopQueue,
                            distributorWholesalerBottomQueue, wholesalerRetailerBottomQueue)

    distributor = Distributor(distributorWholesalerTopQueue, factoryDistributorTopQueue,
                                factoryDistributorBottomQueue, distributorWholesalerBottomQueue)

    factory = Factory(factoryDistributorTopQueue, None, None, factoryDistributorBottomQueue, QUEUE_DELAY_WEEKS)

    return customer, retailer, wholesaler, distributor, factory


# Initialize Statistics object
def Initialize_Stats_obj():
    return SupplyChainStatistics()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='either "train" or "test"') 
    parser.add_argument('-a', '--agent', type=str, required=True,
                        help='either "MonteCarlo" or "DQN"')

    #TODO number of episodes -- add in the setting instead of arguments ??? -- easier to include it in file_names
    parser.add_argument('-e', '--episodes', type=int, default= 1,
                        help='number of episodes')

    args = parser.parse_args()

    # config
    #models_folder = f'{args.agent}_models'
    models_folder = 'saved_models'
    results_folder = 'train_results'
    test_folder = 'test_results'
    num_episodes = args.episodes

    maybe_make_dir(models_folder)
    maybe_make_dir(results_folder)
    maybe_make_dir(test_folder)


    #Initialize_game:

    # save stats in test mode
    myStats = Initialize_Stats_obj() 


    #Agent
    if args.agent == "MonteCarlo":
        agent = SupplyChainAgent.MonteCarloAgent(nA=NUM_ACTIONS, num_episodes=args.episodes, epsilon=INITIAL_EPSILON)
        params = MC_PARAMS

    
    if args.agent == "DQN":
        agent = SupplyChainAgent.DQNAgent(gamma=GAMMA, epsilon=INITIAL_EPSILON, alpha=ALPHA, input_dims=5,
                                  n_actions=NUM_ACTIONS, mem_size=MEM_SIZE, batch_size=BATCH_SIZE )
        params = DQN_PARAMS
        
    all_params = f'{args.agent}_{params}' 

    #load a saved model and play one episode
    if args.mode == 'test':
        if args.agent == 'MonteCarlo':
            agent.load_model(path = f'saved_models/{all_params}.npy') 

        elif args.agent == 'DQN':
            agent.load_model(path = f'saved_models/{all_params}.h5')


        df = pd.DataFrame(columns=['Week', 'currentStock', 'currentOrders', 'lastOrderQuantity', 'currentPipeline',
                                'action', 'reward', 'cumulativeCost', 'done'])

        weeks = []
        currentStock = []
        currentOrders = []
        lastOrderQuantity = []
        currentPipeline = []
        actions = []
        rewards = []
        cumulativeCost = []
        dones = []


    
    if args.mode == 'train':
        costs_incurred = []
        epsilon_values = []


    for i_episode in tqdm(range(args.episodes)):
        # initialize order queues
        theCustomer, myRetailer, myWholesaler, myDistributor, myFactory = Initialize_actors()

        if args.agent == "MonteCarlo" and args.mode == 'train':
            # reset actor states
            agent.reset()

        if args.mode == 'train':
        # decrease exploration over time and save 
            new_epsilon = decrease_epsilon(i_episode, num_episodes = args.episodes, method = EPSILON_METHOD)

            agent.set_epsilon(new_epsilon)
            epsilon_values.append(new_epsilon)


        # run one episode
        for thisWeek in range(WEEKS_TO_PLAY):
            # Retailer takes turn, update stats
            myRetailer.TakeTurn(thisWeek) #TODO check retailer script, it doesn't match with Colin result 
            
            if args.mode == 'test':
                myStats.RecordRetailerCost(myRetailer.GetCostIncurred())
                myStats.RecordRetailerOrders(myRetailer.GetLastOrderQuantity())
                myStats.RecordRetailerEffectiveInventory(myRetailer.CalcEffectiveInventory())


            # Store pre-turn state
            state = list((thisWeek,myWholesaler.currentStock, myWholesaler.incomingOrdersQueue.data[0],
                        myWholesaler.currentOrders, myWholesaler.currentPipeline))

            if args.mode == 'test':
                # record state
                weeks.append(thisWeek)
                currentStock.append(myWholesaler.currentStock)
                currentOrders.append(myWholesaler.incomingOrdersQueue.data[0])
                lastOrderQuantity.append(myWholesaler.lastOrderQuantity)
                currentPipeline.append(myWholesaler.currentPipeline)

            # Wholesaler takes turn
            pre_turn_orders = myWholesaler.currentOrders
            myWholesaler.UpdatePreTurn()

            # Detemine which action to take
            action = agent.get_next_action(state)

            # Take action
            myWholesaler.TakeTurn(thisWeek, action)

            # Store post-turn state
            #TODO check state -- MC didn't have thisweek , second element has been changed to currentStock 
            state_ = list((thisWeek, myWholesaler.currentStock, myWholesaler.incomingOrdersQueue.data[0],
                        myWholesaler.currentOrders, myWholesaler.currentPipeline))

            # Calculate reward
            stock_penalty = myWholesaler.currentStock * STORAGE_COST_PER_UNIT
            backorder_penalty = myWholesaler.currentOrders * BACKORDER_PENALTY_COST_PER_UNIT

            if REWARD_METHOD == 'func1':
                #TODO double check orders_fulfilled calculation -- it is different in different files
                orders_fulfilled = myWholesaler.outgoingDeliveriesQueue.data[0] # pre_turn_orders - state_[3]
                reward = orders_fulfilled - stock_penalty - backorder_penalty
            
            elif REWARD_METHOD == 'func2':
                reward =  - stock_penalty - backorder_penalty
            
            elif REWARD_METHOD == 'func3':
                reward = 1 - abs(12 - myWholesaler.CalcEffectiveInventory())
            
            else:
                raise ValueError('invalid reward method')

            
            done = 1 if thisWeek == WEEKS_TO_PLAY - 1 else 0
            # record stats
            if args.mode == 'test':

                actions.append(action)
                rewards.append(reward)
                cumulativeCost.append(myWholesaler.GetCostIncurred())
                dones.append(done)

                myStats.RecordWholesalerCost(myWholesaler.GetCostIncurred())
                myStats.RecordWholesalerOrders(myWholesaler.GetLastOrderQuantity())
                myStats.RecordWholesalerEffectiveInventory(myWholesaler.CalcEffectiveInventory())

            # Store event
            if args.agent == "MonteCarlo":
                #TODO There isn't state_ here ???
                agent.remember(state, action, reward)
            
            if args.agent == "DQN":
                agent.remember(state, action, reward, state_, done)


            # Distributor takes turn
            myDistributor.TakeTurn(thisWeek)

            if args.mode == 'test':
                myStats.RecordDistributorCost(myDistributor.GetCostIncurred())
                myStats.RecordDistributorOrders(myDistributor.GetLastOrderQuantity())
                myStats.RecordDistributorEffectiveInventory(myDistributor.CalcEffectiveInventory())

            # Factory takes turn, update stats
            myFactory.TakeTurn(thisWeek)

            if args.mode == 'test':
                myStats.RecordFactoryCost(myFactory.GetCostIncurred())
                myStats.RecordFactoryOrders(myFactory.GetLastOrderQuantity())
                myStats.RecordFactoryEffectiveInventory(myFactory.CalcEffectiveInventory())

            #------end of the episode------#


        if args.mode == 'train':

            costs_incurred.append(myWholesaler.GetCostIncurred())
            # Update Q table if agent is MC, update Q approximator weights id agent is DQN

            #TODO -- in Colin branch agent will learn each 200 episodes (????) 
            agent.learn()


            # save model every 5000 episodes
            if args.agent == "DQN":
                if i_episode % 5000 == 0 and i_episode > 0:
                    agent.save_model()


    if args.mode == 'train':        
        # save final agent
        if args.agent == 'MonteCarlo':
            agent.save_model(path = f'saved_models/{all_params}.npy')
        elif args.agent == 'DQN':
            agent.save_model(path = f'saved_models/{all_params}.h5')


        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Cost Incurred', color='b')
        ax1.plot(pd.Series(costs_incurred).rolling(50).mean(), color='b', alpha=0.6)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Epsilon', color='r')
        ax2.plot(epsilon_values, color='r')

        fig.tight_layout()
        plt.title(f'{args.agent} Learning Curve')
        plt.show()
        fig.savefig(f'train_results/{all_params}_costplot.png') 

        # get best score
        best_episode = np.argmin(costs_incurred, axis=0)
        best_score = costs_incurred[best_episode]
        last_200 = np.mean(costs_incurred[-100])
        print('Best score of {}'.format(best_score))
        print('Best episode {}'.format(best_episode))
        print('Average of last 100 episodes {}'.format(last_200))

    
    #plots for a test game #TODO: check these plots 
    if args.mode == 'test':

        myStats.PlotCosts(f'test_results/{all_params}_costs.png')
        myStats.PlotOrders(f'test_results/{all_params}_orders.png')
        myStats.PlotEffectiveInventory(f'test_results/{all_params}_ei.png')

        df['Week'] = weeks
        df['currentStock'] = currentStock
        df['currentOrders'] = currentOrders
        df['lastOrderQuantity'] = lastOrderQuantity
        df['currentPipeline'] = currentPipeline
        df['action'] = actions
        df['reward'] = rewards
        df['cumulativeCost'] = cumulativeCost
        df['done'] = dones
        df.to_csv(f'test_results/{all_params}.csv')






