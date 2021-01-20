#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("../Data/train_F3WbcTw.csv")


# In[3]:


df.head()


# In[6]:


df.sentiment.value_counts()


# In[9]:


df[["text",'drug',"sentiment"]].head()


# In[13]:


df.iloc[4][1]


# In[20]:


len(df.drug.unique())


# In[36]:


df.shape


# In[35]:


len(df.unique_hash.unique())


# In[29]:


df[df["drug"]=="gilenya"]["sentiment"].value_counts()


# In[31]:


df[df["drug"]=="gilenya"]["sentiment"].value_counts().plot(kind="bar")


# In[32]:


df["sentiment"].value_counts().plot(kind="bar")


# In[33]:


df["sentiment"].value_counts()


# ###  noticed that the dataset is imbalanced. Hence need to balance the dataset before bulding model

# In[39]:


df[["text",'drug',"sentiment"]].head()


# In[ ]:




