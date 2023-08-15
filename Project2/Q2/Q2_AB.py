def main():
    print('START Q2_AB\n')

    import numpy as np
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true-y_pred)**2)
    
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
    
    # Implementing a locally weighted linear regression function
    def weighted_linear_regression(x0, X, Y, gamma = 0.204):
        # add bias term
        x0 = np.r_[1, x0]
        X = np.c_[np.ones(len(X)), X]

        # fit model: normal equations with kernel
        xw = X.T * calculating_weights(x0, X, gamma = 0.204)
        theta = np.linalg.pinv(xw @ X) @ xw @ Y
        # "@" is used to
        # predict value
        return x0 @ theta

    # Function for calculating weight
    def calculating_weights(x0, X, gamma = 0.204):
        return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (gamma **2) ))

    data = './datasets/Q1_B_train.txt'
    np_data = readFile(data)

    # X and Y separation
    X = []
    for i in range(np_data.shape[0]):
        X.append(np.array(np_data[i][:-1], dtype=np.float32))
    X = np.array(X)

    Y = []
    for i in range(np_data.shape[0]):
        Y.append([np_data[i][-1]])
    Y = np.array(Y, dtype=np.float32)

    Y_pred = [weighted_linear_regression(x0, X, Y) for x0 in X]

    import matplotlib.pyplot as plt
    plt.scatter(X, Y_pred)
    plt.title('Locally Weighted Linear Regression')
    plt.show()

    print('END Q2_AB\n')

if __name__ == "__main__":
    main()