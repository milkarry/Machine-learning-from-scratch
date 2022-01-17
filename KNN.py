# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# %%
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)

# %%
class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        prediction = []
        for x in X:
            #Calculate the euclidien_distance of each data
            distance = self._euclidien_distance(x)

            #Get the indices of smallest k distance
            k_indices = np.argsort(distance)[:self.k]

            #Get the labels of k data items with smallest distance 
            k_nearest_label = [self.y[i] for i in k_indices]

            #Classify this object as the most frequent class from the k data items
            prediction.append(max(k_nearest_label, key = k_nearest_label.count))
        
        return prediction


    def _euclidien_distance(self, x):
        return np.sqrt(np.sum((x - self.X)**2, axis = 1))
    
    def evaluate(self, true, pred):
        return np.mean(true == pred)

# %%
my_model = KNN(k=3)
my_model.fit(X_train, y_train)
my_prediction = my_model.predict(X_test)
print('My model score is',str(my_model.evaluate(y_test, my_prediction)))

# %%
#Accuracy score using KNN model from sklearn
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print('SK learn model score is',str(accuracy_score(y_test,pred)))


