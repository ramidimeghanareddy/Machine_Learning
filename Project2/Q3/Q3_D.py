def main():
    print('START Q3_D\n')

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

    data = './datasets/Q3_data.txt'

    np_data = readFile(data)

    # X and Y separation
    X = []
    for i in range(np_data.shape[0]):
        X.append(np.array(np_data[i][:-1], dtype=np.float32))
    X = np.array(X)

    y = []
    for i in range(np_data.shape[0]):
        y.append([np_data[i][-1]])
    y = np.array(y, dtype=object)

    # Encode the target variable
    def encode(y):
        y_en = []
        y_en = [1 if i == 'M' else 0 for i in y]
        return np.array(y_en)

    # Normalize the decision variables
    def normalize(X):
        mean =X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean)/std   
        return X, mean, std

    # Sigmoid function
    def sigmoid(z):
        return 1.0/(1 + np.exp(-z))

    # Initialize function
    def initialize(X):
        w = np.ones((X.shape[1]+1,1))
        X = np.c_[np.ones((X.shape[0],1)),X]
        return w, X

    # Calculate loss
    def calculating_loss(X, y, w):
        z = np.dot(X, w)
        c = -(y.T.dot(np.log(sigmoid(z)))+(1-y).T.dot(np.log(1-sigmoid(z))))/len(y)
        return c

    # Train linear regression
    def fit_LogisticRegression(X, y, epochs, lr=0.01):
        w, X = initialize(X)
        losses = np.zeros(epochs,)
        for i in range(epochs):
            w = w - lr*np.dot(X.T,sigmoid(np.dot(X,w))-np.reshape(y,(len(y),1)))
            losses[i] = calculating_loss(X, y, w)
        return w, losses

    # Make the predictions
    def predict(X, mean, std, w):
        X = (X - mean)/std
        z = np.dot(initialize(X)[1],w)
        preds = []
        preds = [1 if i > 0.5 else 0 for i in z]
        preds_class = []
        preds_class = ['M' if i == 1 else 'W' for i in preds]
        return preds_class[0]
  
    # Leave One Out Evaluation function
    def leave_one_out_evaluation(X, y):
        predictions = []
        for i in range(len(X)):
            test_X = [X[i]]
            train_X = np.delete(X, i, axis=0)
            train_y = np.delete(y, i, axis=0)
            train_X_std, mean, std = normalize(train_X)
            train_y_en = encode(train_y)
            w, _ = fit_LogisticRegression(train_X_std, train_y_en, epochs=20, lr=0.01)
            predictions.append(predict(test_X, mean, std, w))
        predictions = np.array(predictions, dtype=object)

        n = 0
        for i in range(len(y)):
            if y[i] == predictions[i]:
                n+=1

        return n/len(y)
    
    new_X = np.delete(X, 2, axis=1)

    print(f'Height, Weight Only \nFor alpha=0.01, iterations=20 \nLeave one out Accuracy = {leave_one_out_evaluation(new_X, y)*100}%')
    
    print('END Q3_D\n')

if __name__ == "__main__":
    main()