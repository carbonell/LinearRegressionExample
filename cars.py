import numpy as np
import matplotlib.pyplot as plt
import cost_function as cf
import gradient_descent as gd
import pylab
data = pylab.loadtxt('93carsd.txt')
x = data[:, 3]
y = data[:, 0]
ts_length = len(y)

plt.ylabel('Minimum Price')
plt.xlabel('Horse Power')

plt.scatter(x, y)
plt.show()
x = np.matrix([x, np.ones(ts_length)])
theta = np.zeros(x.shape[0])
msg = cf.calculate_cost(x, y, theta)
alpha = 0.00003
iterations = 50
result = gd.gradient_descent(x, y, theta, alpha, iterations)
theta = result.theta
predictionLine = theta.dot(x).flatten()

plt.plot(data[:, 3], predictionLine.T, color='red')
plt.scatter(data[:, 3], y)
plt.show()

plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.plot(range(iterations), result.j_history)
plt.show()

# Printing theta values
print("theta values are {t}".format(t=theta))
# Predicting price for 3 cars whose power is 190, 250, 310
case1 = np.matrix([1, 190])
prediction1 = np.asscalar(((theta.T.dot(case1.T)) * 1000))
case2 = np.matrix([1, 250])
prediction2 = np.asscalar(((theta.T.dot(case2.T)) * 1000))
case3 = np.matrix([1, 310])
prediction3 = np.asscalar(((theta.T.dot(case3.T)) * 1000))

predictionMessage = '''For a car that costs 190, 250 or 310 its minimum
price is predicted to be {pr1}, {pr2} and {pr3}, respectively''' 
print(predictionMessage.format(pr1 = prediction1, pr2 = prediction2, pr3 = prediction3))
