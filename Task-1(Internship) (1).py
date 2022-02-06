#!/usr/bin/env python
# coding: utf-8

# # LGMVIP December 2021 Data Science Internship task -1
# 
# --Sampada Gajendra Gaikwad
# 
# Iris Flower Classification ML Project
# 
# 

# #  Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# # Importing Dataset

# In[3]:


df=pd.read_csv("C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)\iris.data")


# In[4]:


#head--top 5 values
df.head()


# In[6]:


#tail--last 5 values
df.tail()


# In[7]:


#shape--gives no. of rows & columns
df.shape


# In[21]:


# Giving names to the columns
columns=['sepal_length','sepal_width','petal_length','petal_width','S_Class']


# In[22]:


df.columns=columns
df.head()


# In[23]:


# Detect missing values for an array-like object.
df.isnull()


# In[24]:


# returns sum of missing values
df.isnull().sum()


# In[25]:


# describe() is used to view some basic statistical details like percentile, mean, std etc. of a data frame
df.describe()


# In[26]:


#nunique--Return Series with number of distinct elements
df.nunique()


# In[27]:


df.S_Class .nunique()


# In[28]:


df.S_Class.value_counts()


# In[29]:


# returns max value
df.max()


# In[30]:


#returns minimum value
df.min()


# #  Data Visualisation

# Graph for each feature vs species
# 

# In[36]:


# sepal_length vs S_Class
plt.bar(df['S_Class'],df['sepal_length'],width=0.5)
plt.title('Sepal length vs Species')
plt.show()

# sepal_width vs S_Class
plt.bar(df['S_Class'],df['sepal_width'],width=0.5)
plt.title('Sepal Width vs Species')
plt.show()

# petal_length vs S_Class
plt.bar(df['S_Class'],df['petal_length'],width=0.5)
plt.title('Petal length vs Species')
plt.show()

# petal_width vs S_Class
plt.bar(df['S_Class'],df['petal_width'],width=0.5)
plt.title('Petal Width vs Species')
plt.show()

#


# # Data Preparation

# In[37]:


from sklearn import preprocessing


# In[38]:


X=df.iloc[:,0:4]
X.head()


# In[43]:


Y=df['S_Class']
Y=Y.values
Y[0:5]

Splitting dataset into Train and Tests sets
# In[45]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
print("x_train :"+str(x_train.shape))
print("x_test :"+str(x_test.shape))
print("y_train :"+str(y_train.shape))
print("y_test :"+str(y_test.shape))


# In[49]:


corr=df.corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr,annot=True , vmin=-1.0,cmap='mako')
plt.title('Correlation Matrix')
plt.show()


# In[50]:


sns.pairplot(data=df,hue='S_Class')


# In[54]:


from sklearn.cluster import KMeans
list=[]
x=df[['sepal_length','sepal_width','petal_length','petal_width',]].to_numpy()
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    list.append(kmeans.inertia_)
plt.plot(range(1,11),list)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('LIST')
plt.show()


# In[56]:


kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)


# In[58]:


# Visualising the clusters- On the first two columns
plt.scatter(x[y_kmeans==0, 0],x[y_kmeans==0, 1],s=100,c='red',label='Iris-sesota')

plt.scatter(x[y_kmeans==1, 0],x[y_kmeans==1, 1],s=100,c='blue',label='Iris-versicolour')

plt.scatter(x[y_kmeans==2, 0],x[y_kmeans==2, 1],s=100,c='green',label='Iris-virginica')


plt.legend()


# Thank You!!
