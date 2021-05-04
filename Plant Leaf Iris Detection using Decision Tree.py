#importing libraries

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

#loading dataset

dataset = load_iris()

#summarizing dataset

print(dataset.data)
print(dataset.target)
print(dataset.data.shape)

#segregating dataset into X and Y
#X(Input/Independent Variable) and Y(Output/Dependent Variable)

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X
Y = dataset.target
Y

#spilitting dataset into Train and Test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)
print(X_train.shape)
print(X_test.shape)

#finding best max_depth Value

accuracy = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1,10):
    model = DecisionTreeClassifier(max_depth = i, random_state = 0)
    model.fit(X_train,Y_train)
    pred = model.predict(X_test)
    score = accuracy_score(Y_test, pred)
    accuracy.append(score)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Finding best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')
    

#training

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth = 3, random_state = 0)
model.fit(X_train,Y_train)

#prediction for all test data

y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#accuracy

from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(Y_test, y_pred)*100))