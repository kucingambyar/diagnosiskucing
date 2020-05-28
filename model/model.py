#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import GaussianNB
import pickle
import pandas


# In[2]:


file = "../data/dataset.csv"
data = pandas.read_csv(file)
data.head()


# In[3]:


x = data[data.columns[:-1].tolist()]
x


# In[4]:


y = data["Class"]
y


# In[5]:


model = GaussianNB()
model.fit(x, y)


# In[6]:


confidence = model.score(x, y)
confidence


# In[7]:


with open('model.pkl','wb') as m:
    pickle.dump(model,m)


# In[8]:


file = "../data/disease.csv"
labels = pandas.read_csv(file)
labels.head()


# In[9]:


with open('model.pkl', 'rb') as m:
    model2 = pickle.load(m)


# In[10]:


predicted= model2.predict([[1,1,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1]])
predicted_labels = predicted.tolist()
for label in predicted_labels:
    label_name = labels.loc[labels['DiseaseId'] == label]['DiseaseName'][0]
    print(label_name)


# In[11]:


from sklearn.metrics import confusion_matrix
y_prediction = model.predict(x)
cm = confusion_matrix(y, y_prediction)
cm


# In[ ]:




