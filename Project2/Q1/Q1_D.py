def main():
	print('START Q1_D\n')
 
	import numpy as np
 
	# Data Pre-processing and fetching the data to clean it
	def data_cleaning(line):
		return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

	def data_fetching(filename):
		with open(filename, 'r') as f:
			data_input = f.readlines()
			cleaning_input = list(map(data_cleaning, data_input))
			f.close()
		return cleaning_input

	# Reading the file after cleaning
	def readFile(dataset_path):
		data_input = data_fetching(dataset_path)
		np_input = np.array(data_input)
		return np_input

	data_training = './datasets/Q1_B_train.txt'
	data_testing = './datasets/Q1_C_test.txt'

	data_training_np = np.array(readFile(data_training), dtype=np.float32)
	data_testing_np = np.array(readFile(data_testing), dtype=np.float32)

	import math
	# Function to generate the basis functions
	def generate(x, d, k):
		l = []                 # Empty list where we will append the basis functions
		for i in range(1, d+1):
			l.append(math.sin(i*k*x)*math.sin(i*k*x))
		return np.array(l)

	# Calculating the Mean Squared Error
	def mse(w, X, y):
		return np.mean((np.dot(X, w)-y)**2)/2

	# Fitting linear regression function
	def fit_LinearRegression(X, y, lr=0.01, eps=0.0001):
		X = np.c_[np.ones((X.shape[0],1)), X]
		m = len(y)
		w = np.ones(X.shape[1])
		costs=[]
		run=True
		costs.append(np.inf)
		i=0
		while run:             # This loop will be repeated until the cost function begins to converge (difference in cost eps)
			error = np.dot(X, w) - y                 # Calculating the error
			cost = 1/(2*m)*np.dot(error.T, error)    # Calculating the cost 
			costs.append(cost)						 # Appending the cost
			w=w-(lr*(1/m)*np.dot(X.T,error))         # Updating the weights
			if abs(costs[i]-costs[i+1])<eps:         # Evaluating whether or not the cost function is converging
				run=False
			i+=1
		costs.pop(0)
		return w, costs

	# X and Y separation(Taking first 20 data points)
	train_X = data_training_np[:20,0]
	train_Y = data_training_np[:20,1]
	test_X = data_testing_np[:,0]
	test_Y = data_testing_np[:,1]
	ln1 = len(train_X)
	ln2 = len(test_X)

	d = 6                            # Adjusting the depth d
	k = 10                           # Adjusting the frequency increment k

	import matplotlib.pyplot as plt

	for i in range(1, k+1):          # Loop over different values of k
		d_legend = []
		plt.figure(figsize=(8,6))
		for j in range(d+1):         # Loop over different depths d
			new_train_X = np.ones((ln1,j))
			for idx in range(ln1):
				new_train_X[idx,:] = generate(train_X[idx], j, i)
			w, _ = fit_LinearRegression(new_train_X, train_Y)         # Fitting the Linear Regression Model
			new_train_X = np.c_[np.ones((new_train_X.shape[0],1)), new_train_X]
			plt.scatter(train_X, np.dot(new_train_X,w))
			plt.title(f'Training data size 20, k={i}')
			d_legend.append(f'd={j}')
		plt.legend(d_legend)
		plt.show()

	lst_n_error = []                  # An empty list to which errors will be appended
	for i in range(1, k+1):          # Loop over different values of k
		print(f'For k={i}')
		for j in range(d+1):         # Loop over different depths d
			new_train_X = np.ones((ln1,j))
			for idx in range(ln1):
				new_train_X[idx,:] = generate(train_X[idx], j, i)
			new_test_X = np.ones((ln2,j))
			for idx in range(ln2):
				new_test_X[idx,:] = generate(test_X[idx], j, i)
			w, _ = fit_LinearRegression(new_train_X, train_Y)         # Fitting the Linear Regression Model
			error = mse(w, np.c_[np.ones((new_test_X.shape[0],1)), new_test_X], test_Y)
			lst_n_error.append([i, j, error])                          # Appending the error to the error_list
			print(f'For d={j} MSE={round(error,4)}')
		print('-------------------')
	lst_n_error = np.array(lst_n_error)
	print('END Q1_D\n')


if __name__ == "__main__":
    main()