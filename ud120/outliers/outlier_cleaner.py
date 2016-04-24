#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    cleaned_data = [ (age, netw, abs(prediction - netw)) for age, netw, prediction in zip(ages, net_worths, predictions) ]
    cleaned_data = sorted(cleaned_data, key = lambda x: x[2])[:int(len(cleaned_data) * 0.9)] # sort by error and remove 10%
    return cleaned_data


