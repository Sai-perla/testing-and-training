#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn


# In[2]:


data = pd.read_csv('COS80026_Assignment1_creditcard.csv')


# In[3]:


print(data.columns)


# In[4]:


print(data.shape)


# In[5]:


print(data.describe())


# In[6]:


data = data.sample(frac = 0.1,random_state = 1)
print(data.shape)


# In[7]:


data.hist(figsize = (20,20))
plt.show()


# In[8]:


Fraud = data[data ['Class'] == 1] #anomlydetection
Valid = data[data['Class'] == 0]

outlirer_fraction = len(Fraud) / float(len(Valid))
print(outlirer_fraction)

print('Fraud Cases:{}'.format(len(Fraud)))
print('Valid cases:{}'.format(len(Valid)))


# In[9]:


corrmat = data.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat,vmax = .8, square = True)
plt.show()


# In[10]:


#unsupervised anomolly dection is beining used as such class is removed and used as the target
columns = data.columns.tolist()

#filtering the columns to remove the data we dont want

columns = [c for c in columns if c not in ["Class"]]

#taregt variable for predection

target ="Class"

X = data[columns]
Y = data[target]

#knowing the shape of X and Y
print(X.shape)
print(Y.shape)


# In[11]:


#checking how the co relation matrix has worked
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a rsandom state

state = 1
#define the outlier detection methods
classifiers = {"Isolation Forest":IsolationForest(max_samples=len(X),
                                                 contamination = outlirer_fraction,
                                                 random_state = state),
              "Local Outlier Factor ": LocalOutlierFactor(n_neighbors = 20,
                                                         contamination = outlirer_fraction, novelty=True )
              }


# In[12]:


#fit the model
n_outliers = len(Fraud)

for i,(clf_name, clf) in enumerate(classifiers.items()):
    
    #fit the data and tag outliers
    
    if clf_name == "Local Outlierr Factor":
        y_pred = clf.fit(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    #as this shows -ve for inliers and 1 for outliers the fraud and normal cases so we need to chnage 
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #run classification metrics
    print('{}: {}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))


# In[17]:


from sklearn.model_selection import train_test_split 


# In[21]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




