Name: Meghana Ramidi
UTA ID: 1002036880

CSE 6363 - Machine Learning 

Question: 1 - KNN

a) Defined three distance metrics:
	1. Cartesian Distance
	2. Manhattan Distance
	3. Minkowski Distance (of order 3)
   
   Then built a KNN algorithm using the following steps:
	Step 1: Calculate the distance (User can choose any of the above distance metrics)
	Step 2: Get the nearest neighbors (K closest instances) for a new piece of data from the training dataset, as defined by the distance measure.
	Step 3: Make predictions from the most similar neighbors collected from the training dataset by returning the most frequent class among the neighbors.

b) Using the above algorithm make the predictions from the test data with K = 1, 3 and 7

c) Implemented Leave One Out Evaluation on the KNN algorithm (With cartesian distance) and get the performance measure of the algorithm for different values of K.

d) Dropped the 'age' data and again evaluated the performance of the algorithm using Leave One Out Evaluation. I observed that the minimum, average, and maximum accuracies for all values of k utilizing all similarities reduced when the age was removed from the dataset. This brings us to the conclusion that in this situation, age is a crucial component to predict the label. The age is not the most crucial feature, but it is one that definitely helps the predictions because the decrease is not very significant.