import numpy as np
"This module makes linear regression calculations"
def calculate_cost(x, y, theta):
    "This function calculates the cost"
    m = len(y)
    guesses = theta.dot(x)
    guess_differences = guesses - y
    guess_differeces_squared = np.multiply(guess_differences, guess_differences)
    j = np.sum(guess_differeces_squared)/(2*m)
    return j
