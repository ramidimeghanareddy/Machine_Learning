def main():
    print('START Q1_AB\n')
    
    import numpy as np
    
    #data analysis
    def data_cleaning(line):
        return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

    def data_fetching(filename):
        with open(filename, 'r') as f:
            data_input = f.readlines()
            clean_input = list(map(data_cleaning, data_input))
            f.close()
        return clean_input

    def File_reading(dataset_path):
        data_input = data_fetching(dataset_path)
        input_np = np.array(data_input)
        return input_np

    data_train = './datasets/Q1_train.txt'
    data_test = './datasets/Q1_test.txt'

    data_train_np = File_reading(data_train)
    data_test_np = File_reading(data_test)

    # Node class definition
    class Node():
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
            # decision nodes
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.info_gain = info_gain
            # Leaf nodes
            self.value = value

    # creating a Decision Tree Classifier 
    class DecisionTreeClassifier():
        def __init__(self, min_samples_split = 2, max_depth=2):
            # initialize the tree
            self.root = None
            # defining the conditions for pruning
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split

        # function for data splitting
        def split(self, dataset, feature_index, threshold):
            left_database = np.array([row for row in dataset if row[feature_index]<=threshold])
            right_database = np.array([row for row in dataset if row[feature_index]>threshold])
            return left_database, right_database

        # Gini index calculation function
        def gini_index(self, y):
            class_labels = np.unique(y)
            gini=0
            for cls in class_labels:
                p_cls = len(y[y==cls])/len(y)
                gini += p_cls**2
            return 1-gini

        # information gain calculation function
        def information_gain(self, parent, l_child, r_child):
            weight_l = len(l_child)/len(parent)
            weight_r = len(r_child)/len(parent)
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
            return gain

        # function for calculating the leaf values
        def calculating_leaf_values(self, y):
            y = list(y)
            return max(y, key=y.count)

        # function for determining the best split
        def get_optimal_split(self, dataset, num_samples, num_features):
            # the best splitss in a blank dictionary
            optimal_split = {}
            maximum_info_gain = -np.inf
            # loop through each feature
            for feature_index in range(num_features):
                feature_values = dataset[:, feature_index]
                possible_thresholds = np.unique(feature_values)
                # determining the ideal threshold
                for threshold in possible_thresholds:
                    left_database, right_database = self.split(dataset, feature_index, threshold)
                    if len(left_database)>0 and len(right_database)>0:
                        y, left_y, right_y = dataset[:, -1], left_database[:, -1], right_database[:, -1]
                        # Calculate the information gain.
                        current_info_gain = self.information_gain(y, left_y, right_y)
                        # keeping the best splits updated
                        if current_info_gain > maximum_info_gain:
                            optimal_split['feature_index'] = feature_index
                            optimal_split['threshold'] = threshold
                            optimal_split['left_database'] = left_database
                            optimal_split['right_database'] = right_database
                            optimal_split['info_gain'] = current_info_gain
                            maximum_info_gain = current_info_gain
            return optimal_split

        # function for constructing trees
        def constructing_trees(self, dataset, curr_depth=0):
            X, y = dataset[:, :-1], dataset[:, -1]
            num_samples, num_features = np.shape(X)
            # dividing the data until the required conditions for pruning are met
            if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
                # selecting the optimal split
                optimal_split = self.get_optimal_split(dataset, num_samples, num_features)
                if optimal_split['info_gain']>0:
                    subtree_left = self.constructing_trees(optimal_split['left_database'], curr_depth+1)
                    subtree_right = self.constructing_trees(optimal_split['right_database'], curr_depth+1)
                    return Node(optimal_split['feature_index'], optimal_split['threshold'], subtree_left, subtree_right, optimal_split['info_gain'])
            # calculating the leaf node's output
            leaf_value = self.calculating_leaf_values(y)
            return Node(value=leaf_value)

        # the tree training function
        def fit(self, X, y):
            dataset = np.concatenate((X, y), axis=1)
            self.root = self.constructing_trees(dataset)
            
        # function for predicting from a dataset
        def predict(self, X):
            predictions = [self.making_prediction(x, self.root) for x in X]
            return predictions
        
        # function to obtain a datapoint's prediction
        def making_prediction(self, x, tree):
            if tree.value!=None: 
                return tree.value
            feature_val = x[tree.feature_index]
            if feature_val <= tree.threshold:
                return self.making_prediction(x, tree.left)
            else:
                return self.making_prediction(x, tree.right)

    # function for evaluating the model's accuracy
    def accuracy_percentage(true_Y, predict_Y):
        m = len(true_Y)
        n = len(predict_Y)
        if m == n:
            count = 0
            for i in range(m):
                if true_Y[i] == predict_Y[i]:
                    count += 1
            return count/m
        else:
            print('Length mismatch')

    ## Separation of X and y
    # data training
    train_X = []
    for i in range(data_train_np.shape[0]):
        train_X.append(np.array(data_train_np[i][:-1], dtype=np.float32))
    train_X = np.array(train_X)

    train_Y = []
    for i in range(data_train_np.shape[0]):
        train_Y.append([data_train_np[i][-1]])
    train_Y = np.array(train_Y, dtype=object)

    # data testing
    test_X = []
    for i in range(data_test_np.shape[0]):
        test_X.append(np.array(data_test_np[i][:-1], dtype=np.float32))
    test_X = np.array(test_X)

    test_Y = []
    for i in range(data_test_np.shape[0]):
        test_Y.append([data_test_np[i][-1]])
    test_Y = np.array(test_Y, dtype=object)

    for depth in range(1, 6):
        classifier = DecisionTreeClassifier(max_depth=depth)
        classifier.fit(train_X, train_Y)
        train_predicton = classifier.predict(train_X)
        test_predicton = classifier.predict(test_X)
        train_accuracy = accuracy_percentage(train_Y, train_predicton)
        test_accuracy = accuracy_percentage(test_Y, test_predicton)
        print(f"DEPTH = {depth}\nAccuracy | Train = {round(train_accuracy,2)} | Test = {round(test_accuracy,2)}")

    # We can observe overfitting for depths 3, 4 & 5
    print('END Q1_AB\n')
    
if __name__ == "__main__":
    main()