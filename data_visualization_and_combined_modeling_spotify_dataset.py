#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('dataset.csv')
# index= false
# rename unnamed column


# In[3]


data.shape




# We have 1,14,000 rows and 21 columns


# In[5]:


data.head(5)




# In[6]:

# ## Finding Null values in the dataset
data.isnull().sum()


# In[7]:


data = data.dropna()


# In[8]:


data.isnull().sum()




# In[9]:
# ## Exploratory Data Analysis.

# ## 1. Univariate Analysis

sns.boxplot(data['popularity'])


# In[10]:


# This shows us that the median popularity is around 30 which we have taken for analysis.


# In[11]:


data['album_name']=[i for i in data['album_name']]
data['album_name'].value_counts()


# In[12]:


# We have majority of Alternative christmas 2022 and feliz cumpleanos con perreo albums in the dataset


# In[13]:


data['track_genre']=[i for i in data['track_genre']]
data['track_genre'].value_counts()


# In[14]:


# We can see all genres are equally divided in the dataset 


# In[15]:


sns.boxplot(data['danceability'])


# In[16]:


# Majority of songs danceable. 


# In[17]:


sns.boxplot(data['energy'])


# In[18]:


# Majority of songs are high energy tracks. 


# In[19]:


sns.countplot(x="key", data=data)
data["key"].value_counts()


# In[20]:


# Majority of songs have key 7 i.e, G pitch and 0 i.e, C pitch 


# In[21]:


sns.countplot(x="mode", data=data)
data["mode"].value_counts()


# In[22]:


# From the above graph we can say that majority of songs have major modality of track. 


# In[23]:


filt = data['speechiness'] > 0.66
print("Tracks that are probably made entirely of spoken words are: ",filt.sum())


# In[24]:


filt2 = (data['speechiness'] > 0.33) & (data['speechiness'] < 0.66)
print("Tracks that may contain both music and speech: ", filt2.sum())


# In[25]:


filt3 = data['speechiness'] < 0.33
print("Tracks that are most likely represent music and other non-speech-like music: ",filt3.sum())


# In[26]:


# Acousticness measure
sns.boxplot(data['acousticness'])


# In[27]:


# Majority of tracks have low acousticness


# In[28]:


# Valence measure, describing the musical positiveness conveyed by a track
sns.boxplot(data['valence'])


# In[29]:


# Average of all tracks are likely to sound more negative.


# # 2. Bivariate Analysis

# In[30]:


plt.figure(figsize=(20, 10))
#correlation plot of all the variables. 
data_corr = data.corr()
sns.heatmap(data_corr, 
            xticklabels = data_corr.columns.values,
            yticklabels = data_corr.columns.values,
            annot = True);


# In[31]:


# From the graph we can see that, 
# Danceability and valence are positively related
# Energy and loudness are highly postively related 
# Energy and acousticness are higly negatively related
# Loudness and acousticness are highly negatively related 
# Loudness and instrumentalness are negatively related 
# Valence and instrumentalness are negatively related 


# ## From the correlation graph we can make following conclusions.
# ## 1. As danceability of the song increases, the positiveness conveyed by the track also increases. 
# ## 2. As energy of the song increases, loudness of the song also increases. 
# ## 3. As energy of the song increases, acousticness of the song increases. 
# ## 4. As acousticness increases, loudness of the song decreases therefore the energy of the song also decreases. 
# ## 5. As loudness of the track increases, instrumentalness used in the song decreases hence the energy also decreases. 
# ## 6. As instrumentalness increases, the positiveness conveyed by the track decreases hence the danceability also decreases. 

# In[35]:


from statsmodels.formula.api import ols
model = ols(formula='popularity ~ danceability + energy + speechiness + acousticness + C(mode) + liveness + loudness',data=data).fit()
model.summary()


# # Modeling for traditional and algorithmic variables combined

# In[36]:


X = data[['energy', 'instrumentalness','acousticness', 'liveness','loudness', 'tempo', 'valence']]
#X = data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
y = data['danceability']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =1234, test_size = 0.3)


# In[37]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)*100


# In[38]:



from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors = 5)
model.fit(X_train, y_train)


# In[39]:


model.score(X_test, y_test)


# In[40]:


import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import explained_variance_score
# Instantiation
xgb_r = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123)

# Fitting the model
xgb_r.fit(X_train, y_train)

# Predict the model
pred = xgb_r.predict(X_test)

# RMSE Computation
rmse = np.sqrt(MSE(y_test, pred))
print("RMSE : % f" %(rmse))
print("Accuracy:", xgb_r.score(X_test, y_test)*100)


# In[41]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred, label="predicted")
plt.title("Danceability test and predicted data")
plt.legend()
plt.show()


# In[42]:


from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_r, max_num_features=8, ax=ax)
plt.show();


# In[43]:


## Where here the F score is a measure "...based on the number of times a variable is selected for splitting, 
## weighted by the squared improvement to the model as a result of each split, and averaged over all trees."


# In[ ]:





# In[ ]:





# In[ ]:




