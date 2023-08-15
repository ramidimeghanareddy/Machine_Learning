def main():
    print('START Q3_AB\n')

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
    
    # Preprocess the data
    X_std, mean, std = normalize(X)
    y_en = encode(y)

    # Train the logistic regression with learning rate=0.01
    w, losses = fit_LogisticRegression(X_std, y_en, epochs=20, lr=0.01)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')

    coordinate_x = np.linspace(X.min(axis=0)[0], X.max(axis=0)[0], 100)
    coordinate_y = np.linspace(X.min(axis=0)[1], X.max(axis=0)[1], 100)
    cord_x, cord_y = np.meshgrid(coordinate_x, coordinate_y)
    cord_z = ((0.5 - w[0] - w[1]*(cord_x-mean[0])/std[0] - w[2]*(cord_y-mean[1])/std[1])/w[3])*std[2] + mean[2]


    for i in range(X.shape[0]):
        if w[0] + w[1]*X_std[i,0] + w[2]*X_std[i,1] + w[3]*X_std[i,2] < 0.5:
            ax.scatter3D(X[i,0], X[i,1], X[i,2], color = 'orange')
        else:
            ax.scatter3D(X[i,0], X[i,1], X[i,2], color = 'blue')


    ax.plot_surface(cord_x, cord_y, cord_z, color = 'blue', alpha=0.8)

    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Age')

    ax.view_init(30, 180)
    plt.show()

    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')

    coordinate_x = np.linspace(X.min(axis=0)[0], X.max(axis=0)[0], 100)
    coordinate_y = np.linspace(X.min(axis=0)[1], X.max(axis=0)[1], 100)
    cord_x, cord_y = np.meshgrid(coordinate_x, coordinate_y)
    cord_z = ((0.5 - w[0] - w[1]*(cord_x-mean[0])/std[0] - w[2]*(cord_y-mean[1])/std[1])/w[3])*std[2] + mean[2]


    for i in range(X.shape[0]):
        if w[0] + w[1]*X_std[i,0] + w[2]*X_std[i,1] + w[3]*X_std[i,2] < 0.5:
            ax.scatter3D(X[i,0], X[i,1], X[i,2], color = 'orange')
        else:
            ax.scatter3D(X[i,0], X[i,1], X[i,2], color = 'blue')


    ax.plot_surface(cord_x, cord_y, cord_z, color = 'blue', alpha=0.8)

    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Age')

    ax.view_init(150,30)
    plt.show()

    fig = plt.figure(figsize=(8,6))
    ax = plt.axes(projection='3d')

    coordinate_x = np.linspace(X.min(axis=0)[0], X.max(axis=0)[0], 100)
    coordinate_y = np.linspace(X.min(axis=0)[1], X.max(axis=0)[1], 100)
    cord_x, cord_y = np.meshgrid(coordinate_x, coordinate_y)
    cord_z = ((0.5 - w[0] - w[1]*(cord_x-mean[0])/std[0] - w[2]*(cord_y-mean[1])/std[1])/w[3])*std[2] + mean[2]


    ax.scatter3D(X[:,0], X[:,1], X[:,2], color = 'blue')


    ax.plot_surface(cord_x, cord_y, cord_z, color = 'blue', alpha=0.5)

    ax.set_xlabel('Height')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Age')

    plt.show()

    for epoch in range(20):
        w, _ = fit_LogisticRegression(X_std, y_en, epochs=epoch, lr=0.01)
        preds = []
        for idx in range(len(X)):
            preds.append(predict([X[idx]], mean, std, w))
        preds = np.array(preds, dtype=object)
        
        n = 0
        for i in range(len(y)):
            if y[i] == preds[i]:
                n+=1
                
        print(f'Itr={epoch} accuracy={100*(n/len(y))}')

    print('END Q3_AB\n')


if __name__ == "__main__":
    main()
