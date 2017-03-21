import collections as c
import cost_function as cf
import numpy as np
"This module calculates gradient descent"
def gradient_descent(feature_set_matrix, output_set, theta, alpha, iterations):
    "Calculates Gradient descent"
    feature_set_length = len(output_set)
    cost_history = np.zeros(iterations)
    for i in range(0, iterations):

        # Resolving the equation h(x) = theta(0) + theta(1)X(1)+ .... theta(n)X(n)
        # for all features in data set
        # The equation is solved via matrix multiplication
        # this vector will contain my guesses
        guesses = theta.dot(feature_set_matrix)

        # calculate the distance of my guesses from reality, the smaller the distance, the better
        guess_differences = guesses - output_set
        
        # create the vector that will be used to update all values of theta at the same time
        # this will work by simply assignation (no budu magic alpha or the like)
        new_theta = np.zeros(2)
        for index, theta_ith in enumerate(np.asarray(theta)):
            # multiplying each and every guess difference 
            # by an actual real value for the feature set
            # and then, adding these calculated values together
            # finally, this will all be divided by the number of training samples
            # and substracted to the previous value
            # and will be multiplied by our budu magic alpha

            ith_feature_set = feature_set_matrix[index, :]
            product_aggregate = guess_differences.dot(ith_feature_set.T)
            new_val = theta_ith - (alpha * np.asscalar(product_aggregate))/feature_set_length
            new_theta[index] = new_val
        theta = new_theta
        cost_history[i] = cf.calculate_cost(feature_set_matrix, output_set, theta)
    Result = c.namedtuple('GradientResult', ['theta', 'j_history'])
    result = Result(theta, cost_history)
    return result