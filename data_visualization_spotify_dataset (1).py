#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[69]:


data=pd.read_csv('dataset.csv')
# index= false
# rename unnamed column


# In[70]:


data.shape


# In[71]:


# We have 1,14,000 rows and 21 columns


# In[72]:


data.head(5)


# ## Finding Null values in the dataset

# In[73]:


data.isnull().sum()


# In[74]:


data = data.dropna()


# In[75]:


data.isnull().sum()


# ## Exploratory Data Analysis.

# ## 1. Univariate Analysis

# In[268]:


sns.boxplot(data['popularity'])


# In[77]:


# This shows us that the median popularity is around 30 which we have taken for analysis.


# In[78]:


data['album_name']=[i for i in data['album_name']]
data['album_name'].value_counts()


# In[79]:


# We have majority of Alternative christmas 2022 and feliz cumpleanos con perreo albums in the dataset


# In[80]:


data['track_genre']=[i for i in data['track_genre']]
data['track_genre'].value_counts()


# In[81]:


# We can see all genres are equally divided in the dataset 


# In[82]:


sns.boxplot(data['danceability'])


# In[83]:


# Majority of songs danceable. 


# In[84]:


sns.boxplot(data['energy'])


# In[85]:


# Majority of songs are high energy tracks. 


# In[86]:


sns.countplot(x="key", data=data)
data["key"].value_counts()


# In[87]:


# Majority of songs have key 7 i.e, G pitch and 0 i.e, C pitch 


# In[88]:


sns.countplot(x="mode", data=data)
data["mode"].value_counts()


# In[89]:


# From the above graph we can say that majority of songs have major modality of track. 


# In[90]:


filt = data['speechiness'] > 0.66
print("Tracks that are probably made entirely of spoken words are: ",filt.sum())


# In[91]:


filt2 = (data['speechiness'] > 0.33) & (data['speechiness'] < 0.66)
print("Tracks that may contain both music and speech: ", filt2.sum())


# In[92]:


filt3 = data['speechiness'] < 0.33
print("Tracks that are most likely represent music and other non-speech-like music: ",filt3.sum())


# In[93]:


# Acousticness measure
sns.boxplot(data['acousticness'])


# In[94]:


# Majority of tracks have low acousticness


# In[95]:


# Valence measure, describing the musical positiveness conveyed by a track
sns.boxplot(data['valence'])


# In[96]:


# Average of all tracks are likely to sound more negative.


# # 2. Bivariate Analysis

# In[134]:


plt.figure(figsize=(20, 10))
#correlation plot of all the variables. 
data_corr = data.corr()
sns.heatmap(data_corr, 
            xticklabels = data_corr.columns.values,
            yticklabels = data_corr.columns.values,
            annot = True);


# In[104]:


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

# # ANOVA test

# In[148]:


import scipy.stats as stats

unique_genre = data['track_genre'].unique()
for genre in unique_genre:
    stats.probplot(data[data['track_genre'] == genre]['popularity'], dist="norm", plot=plt)
    plt.title("Probability Plot - " +  genre)
    plt.show()
    


# In[149]:


## track_genre, bins, barplot on correlated variables, link that to danceability. 


# In[150]:


data.head()


# In[245]:


X = data[['energy', 'instrumentalness','acousticness', 'liveness','loudness', 'tempo', 'valence']]
y = data['danceability']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =1234, test_size = 0.3)


# In[236]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)*100


# In[197]:


from statsmodels.formula.api import ols
model = ols(formula='popularity ~ danceability + energy + speechiness + acousticness + C(mode) + liveness + loudness',data=data).fit()
model.summary()


# In[189]:


data['duration_ms'] = data['duration_ms'] / 6000


# In[210]:


data['track_genre'].unique()


# In[214]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['track_genre'] = le.fit_transform(data['track_genre'])


# In[215]:


data.head()


# In[246]:



from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors = 5)
model.fit(X_train, y_train)


# In[247]:


model.score(X_test, y_test)


# In[266]:


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


# In[259]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred, label="predicted")
plt.title("Danceability test and predicted data")
plt.legend()
plt.show()


# In[263]:


from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(12,6))
plot_importance(xgb_r, max_num_features=8, ax=ax)
plt.show();


# In[264]:


## Where here the F score is a measure "...based on the number of times a variable is selected for splitting, 
## weighted by the squared improvement to the model as a result of each split, and averaged over all trees."


# In[ ]:




