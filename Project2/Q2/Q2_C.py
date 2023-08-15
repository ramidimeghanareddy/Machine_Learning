def main():
    print('START Q2_C\n')

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

    data = './datasets/Q1_C_test.txt'
    np_data = readFile(data)

    # test_X and test_Y separation
    test_X = []
    for i in range(np_data.shape[0]):
        test_X.append(np.array(np_data[i][:-1], dtype=np.float32))
    test_X = np.concatenate( test_X, axis=0 )

    test_Y = []
    for i in range(np_data.shape[0]):
        test_Y.append([np_data[i][-1]])
    test_Y = np.array(test_Y, dtype=np.float32)

    test_Y_pred = [weighted_linear_regression(x0, X, Y) for x0 in test_X]
    test_Y_pred = np.concatenate(test_Y_pred, axis=0)

    print(f'data size = {len(Y)}, MSE = {mean_squared_error(test_Y, test_Y_pred)}')
    print('END Q2_C\n')

if __name__ == "__main__":
    main()