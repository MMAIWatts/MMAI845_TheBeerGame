"""
-------------------------------------------------------
This file contains and defines the Customer class.
-------------------------------------------------------
Author:  Tom LaMantia
Email:   tom@lamantia.mail.utoronto.ca
Version: February 14th 2016
-------------------------------------------------------
"""

from Settings import *

class Customer:
    
    def __init__(self):
        """
        -------------------------------------------------------
        Constructor for the Customer class.
        -------------------------------------------------------
        Preconditions: None
        Postconditions:
            Initializes the Customer object in its initial state.
        -------------------------------------------------------
        """
        self.totalBeerReceived = 0
        return
    
    def RecieveFromRetailer(self, amountReceived):
        """
        -------------------------------------------------------
        Receives stock from the retailer.
        -------------------------------------------------------
        Preconditions: amountReceived - the number of cases shipped to the
                    customer by the retailer.
        Postconditions:
            Increments totalBeerReceived accordingly.
        -------------------------------------------------------
        """
        self.totalBeerReceived += amountReceived
        
        return
    
    def CalculateOrder(self, weekNum):
        """
        -------------------------------------------------------
        Calculates the amount of stock to order from the retailer.
        -------------------------------------------------------
        Preconditions: weekNum - the current week of game-play.
        Postconditions:
            The customer orders 4 cases on weeks 1-5, and 8 cases 
            for all other weeks. 
        -------------------------------------------------------
        """
        if weekNum <= 5:
            result = CUSTOMER_INITIAL_ORDERS
        else:
            result = CUSTOMER_SUBSEQUENT_ORDERS
        return result
    
    def GetBeerReceived(self):
        """
        -------------------------------------------------------
        Returns the total beer received by the customer.
        -------------------------------------------------------
        Preconditions: None.
        Postconditions: Returns totalBeerReceived
        -------------------------------------------------------
        """
        return self.totalBeerReceived