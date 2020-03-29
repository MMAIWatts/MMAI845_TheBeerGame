"""
-------------------------------------------------------
This file contains and defines the Factory class.
-------------------------------------------------------
Author:  Tom LaMantia
Email:   tom@lamantia.mail.utoronto.ca
Version: February 7th 2016
-------------------------------------------------------
"""

from theBeerGame.Settings import *
from theBeerGame.SupplyChainActor import SupplyChainActor
from theBeerGame.SupplyChainQueue import SupplyChainQueue

class Factory(SupplyChainActor):
    
    def __init__(self, incomingOrdersQueue, outgoingOrdersQueue, incomingDeliveriesQueue, outgoingDeliveriesQueue, productionDelayWeeks):
        """
        -------------------------------------------------------
        Constructor for the Factory class.
        -------------------------------------------------------
        Preconditions: incomingOrdersQueue, outgoingOrdersQueue, incomingDeliveriesQueue, outgoingDeliveriesQueue - 
                the supply chain queues. Note: outgoingOrdersQueue and incomingDeliveriesQueue should be NONE.
                productionDelayWeeks - an integer value indicating the nunber of weeks required to make a case of beer.
        Postconditions:
            Initializes the Factory object in its initial state
            by calling parent constructor and setting the
            retailer's customer.
        -------------------------------------------------------
        """
        super().__init__(incomingOrdersQueue, outgoingOrdersQueue, incomingDeliveriesQueue, outgoingDeliveriesQueue)
        self.BeerProductionDelayQueue = SupplyChainQueue(productionDelayWeeks)
        
        #We assume that the factory already has some runs in production. This is in the rules, and ensures initial stability.
        self.BeerProductionDelayQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        self.BeerProductionDelayQueue.PushEnvelope(CUSTOMER_INITIAL_ORDERS)
        return
    
    def ProduceBeer(self, weekNum):
        """
        -------------------------------------------------------
        Calculates the size of this week's production run.
        -------------------------------------------------------
        Preconditions:  weekNum - the current week number.
        Postconditions:
            Calculates the production run using an anchor and maintain
            strategy.
        -------------------------------------------------------
        """

        #basestock policy
        amountToOrder = self.currentOrders - self.currentStock
        if amountToOrder < 0:
            amountToOrder = 0
            
        self.BeerProductionDelayQueue.PushEnvelope(amountToOrder)
        self.lastOrderQuantity = amountToOrder
        
        return
    
    def FinishProduction(self):
        """
        -------------------------------------------------------
        Finishes production by popping the production queue and
        adding this beer to the current stock of the factory.
        -------------------------------------------------------
        Preconditions:  None
        Postconditions: Updates currentStock to reflect the beer
            that the factory just brewed.
        -------------------------------------------------------
        """
        amountProduced = self.BeerProductionDelayQueue.PopEnvelope()
        
        if amountProduced > 0:
            self.currentStock += amountProduced
        
        return
     
    def TakeTurn(self, weekNum):
        
        #The steps for taking a turn are as follows:
        
        #PREVIOUS PRODUCTION RUNS FINISH BREWING.
        self.FinishProduction()
        
        #RECEIVE NEW ORDER FROM DISTRIBUTOR
        self.ReceiveIncomingOrders()     #This also advances the queue!
        
        #PREPARE DELIVERY
        if weekNum <= 4:
            self.PlaceOutgoingDelivery(4)
        else:
            self.PlaceOutgoingDelivery(self.CalcBeerToDeliver())
        
        #PRODUCE BEER
        self.ProduceBeer(weekNum)
        
        #UPDATE COSTS
        self.costsIncurred += self.CalcCostForTurn()
        
        return