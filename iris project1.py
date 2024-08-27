#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
  
# Reading the CSV file 
df = pd.read_csv("Iris.csv") 
  
# Printing top 5 rows 
df.head()


# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


data = df.drop_duplicates(subset ="Species",) 
data


# In[7]:


df.value_counts("Species")


# In[8]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 
  
  
sns.countplot(x='Species', data=df, ) 
plt.show()


# In[9]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', 
				hue='Species', data=df, ) 

# Placing Legend outside the Figure 
plt.legend(bbox_to_anchor=(1, 1), loc=2) 

plt.show()


# In[10]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', 
				hue='Species', data=df, ) 

# Placing Legend outside the Figure 
plt.legend(bbox_to_anchor=(1, 1), loc=2) 

plt.show()


# In[11]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.pairplot(df.drop(['Id'], axis = 1), 
			hue='Species', height=2)


# In[12]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


fig, axes = plt.subplots(2, 2, figsize=(10,10)) 

axes[0,0].set_title("Sepal Length") 
axes[0,0].hist(df['SepalLengthCm'], bins=7) 

axes[0,1].set_title("Sepal Width") 
axes[0,1].hist(df['SepalWidthCm'], bins=5); 

axes[1,0].set_title("Petal Length") 
axes[1,0].hist(df['PetalLengthCm'], bins=6); 

axes[1,1].set_title("Petal Width") 
axes[1,1].hist(df['PetalWidthCm'], bins=6);


# In[13]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "SepalLengthCm").add_legend() 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "SepalWidthCm").add_legend() 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "PetalLengthCm").add_legend() 

plot = sns.FacetGrid(df, hue="Species") 
plot.map(sns.distplot, "PetalWidthCm").add_legend() 

plt.show()


# In[14]:


data.corr(method='pearson')


# In[15]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 


sns.heatmap(df.corr(method='pearson').drop( 
['Id'], axis=1).drop(['Id'], axis=0), 
			annot = True); 

plt.show()


# In[16]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 

def graph(y): 
	sns.boxplot(x="Species", y=y, data=df) 

plt.figure(figsize=(10,10)) 
	
# Adding the subplot at the specified 
# grid position 
plt.subplot(221) 
graph('SepalLengthCm') 

plt.subplot(222) 
graph('SepalWidthCm') 

plt.subplot(223) 
graph('PetalLengthCm') 

plt.subplot(224) 
graph('PetalWidthCm') 

plt.show()


# In[17]:


# importing packages 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Load the dataset 
df = pd.read_csv('Iris.csv') 

sns.boxplot(x='SepalWidthCm', data=df)


# In[ ]:






# In[ ]:





# In[ ]:




