#!/usr/bin/env python
# coding: utf-8

# # Modules

# In[42]:


# OS
import os
from os.path import join
# Pandas
import pandas as pd
# Numpy
import numpy as np
# Technical Analaysis Library
import talib
# Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier 


# # Parameters

# In[58]:


dir_data = join( '..', 'data' )
file_path1 = join( dir_data, 'train.csv' )
file_path2 = join( dir_data, 'test.csv' )


# # Load Data

# In[59]:


train = pd.read_csv( file_path1, index_col=0)
test = pd.read_csv( file_path2, index_col=0)


# # Explorations

# In[60]:


print(type(train))
print(train.keys())
print(train.shape)
train.head()

print(type(test))
print(test.keys())
print(test.shape)
test.head()


# In[61]:


train.columns


# In[62]:


print(np.sort(train['type'].unique()))
print(np.sort(train['color'].unique()))
print(np.sort(test['color'].unique()))


# In[63]:


train.info()


# In[64]:


test.info()


# In[65]:


pd.scatter_matrix(train, alpha=0.2, figsize=(10, 10))


# In[66]:


sns.pairplot( train,hue='type')


# # Prediction

# In[71]:


X_train = pd.get_dummies(train.drop("type", axis = 1))
y_train = train["type"]
X_test = pd.get_dummies(test)


# In[72]:


# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train, y_train)

# Predict the labels for the training data X: y_pred
y_pred = knn.predict(X_train)

# Predict and print the label for the new data point X_new
y_prediction = knn.predict(X_test)
print("Prediction: {}".format(y_prediction)) 


# In[73]:


knn.score(X_train,y_train)


# In[75]:


# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_prediction)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



























