#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


heart_data=pd.read_csv("E:\Programming with Data Analytics\heart.csv")


# In[7]:


#printing the starting 5 rows of the dataset
heart_data.head()


# In[8]:


#printing the last 5 rows of the dataset
heart_data.tail()


# In[11]:


#Number of rows and columns in the dataset
heart_data.shape


# In[12]:


#getting some information about the data
heart_data.info()


# In[14]:


#checking the missing values
heart_data.isnull().sum()


# In[16]:


#statistical measures of the data
heart_data.describe()


# In[17]:


#checking the distribution of the target variable
heart_data['target'].value_counts()


# In[19]:


#1-----> Heart Disease
#0-----> Healthy Heart


# In[20]:


#Splitting the features and target column
X=heart_data.drop(columns='target',axis=1)
Y=heart_data['target']


# In[23]:


print(X)


# In[22]:


print(Y)


# In[24]:


#Splitting the data into training data and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[25]:


print(X.shape,X_train.shape,X_test.shape)


# In[26]:


#Model Training
model=LogisticRegression()


# In[27]:


#Training the Logistic Regression model with training data
model.fit(X_train,Y_train)


# In[28]:


#Model Evaluation
#Accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[30]:


print("Accuracy on training data: ",training_data_accuracy*100)


# In[35]:


#Accuracy on testing data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[34]:


print("Accuracy on testing data: ",testing_data_accuracy*100)


# In[41]:


import numpy as np
import warnings

# Change the input data into a numpy array
inp_data_arr = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
inp_data_reshaped = inp_data_arr.reshape(1, -1)

# Suppress the warning
warnings.filterwarnings("ignore")

# Make predictions
prediction = model.predict(inp_data_reshaped)

# Revert to the default warning behavior
warnings.filterwarnings("default")

if prediction == 0:
    print("The Person has a Healthy Heart")
else:
    print("The person has Heart Disease")


# In[42]:


#Saving the Model


# In[46]:


import pickle

filename = 'trained_model.sav'

# Use a context manager to ensure the file is properly closed
with open(filename, 'wb') as file:
    pickle.dump(model, file)


# In[48]:


#Loading the saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))


# In[49]:


inp_data_arr = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
inp_data_reshaped = inp_data_arr.reshape(1, -1)

# Suppress the warning
warnings.filterwarnings("ignore")

# Make predictions
prediction = loaded_model.predict(inp_data_reshaped)

# Revert to the default warning behavior
warnings.filterwarnings("default")

if prediction == 0:
    print("The Person has a Healthy Heart")
else:
    print("The person has Heart Disease")


# In[ ]:




