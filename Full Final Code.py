import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


full_spot_data = pd.read_csv('dataset.csv')
# index= false
# rename unnamed column


# In[3]:


full_spot_data.shape


# In[4]:


# We have 1,14,000 rows and 21 columns


# In[5]:


full_spot_data.head(5)


# ## Finding Null values in the dataset

# In[6]:


full_spot_data.isnull().sum()


# In[7]:


full_spot_data = full_spot_data.dropna()


# In[8]:


full_spot_data.isnull().sum()


# ## Exploratory Data Analysis.

# ## 1. Univariate Analysis

# In[9]:


sns.boxplot(full_spot_data['popularity'])


# In[10]:


# This shows us that the median popularity is around 30 which we have taken for analysis.


# In[11]:


full_spot_data['album_name']=[i for i in full_spot_data['album_name']]
full_spot_data['album_name'].value_counts()


# In[12]:


# We have majority of Alternative christmas 2022 and feliz cumpleanos con perreo albums in the dataset


# In[13]:


full_spot_data['track_genre']=[i for i in full_spot_data['track_genre']]
full_spot_data['track_genre'].value_counts()


# In[14]:


# We can see all genres are equally divided in the dataset 


# In[15]:


sns.boxplot(full_spot_data['danceability'])


# In[16]:


# Majority of songs danceable. 


# In[17]:


sns.boxplot(full_spot_data['energy'])


# In[18]:


# Majority of songs are high energy tracks. 


# In[19]:


sns.countplot(x="key", data=full_spot_data)
full_spot_data["key"].value_counts()


# In[20]:


# Majority of songs have key 7 i.e, G pitch and 0 i.e, C pitch 


# In[21]:


sns.countplot(x="mode", data=full_spot_data)
full_spot_data["mode"].value_counts()


# In[22]:


# From the above graph we can say that majority of songs have major modality of track. 


# In[23]:


filt = full_spot_data['speechiness'] > 0.66
print("Tracks that are probably made entirely of spoken words are: ",filt.sum())


# In[24]:


filt2 = (full_spot_data['speechiness'] > 0.33) & (full_spot_data['speechiness'] < 0.66)
print("Tracks that may contain both music and speech: ", filt2.sum())


# In[25]:


filt3 = full_spot_data['speechiness'] < 0.33
print("Tracks that are most likely represent music and other non-speech-like music: ",filt3.sum())


# In[26]:


# Acousticness measure
sns.boxplot(full_spot_data['acousticness'])


# In[27]:


# Majority of tracks have low acousticness


# In[28]:


# Valence measure, describing the musical positiveness conveyed by a track
sns.boxplot(full_spot_data['valence'])


# In[29]:


# Average of all tracks are likely to sound more negative.


# # 2. Bivariate Analysis

# In[30]:


plt.figure(figsize=(20, 10))
#correlation plot of all the variables. 
data_corr = full_spot_data.corr()
sns.heatmap(data_corr, 
            xticklabels = data_corr.columns.values,
            yticklabels = data_corr.columns.values,
            annot = True);

f = alt.Chart(req_spot_data).mark_point().encode(x = alt.X('popularity',
                                                       axis= alt.Axis(title='Popularity(%)')),
                                             y = alt.Y('duration_ms',
                                                       axis = alt.Axis(title = 'Duration(in seconds)')),
                                             color = alt.Color('Emotional_outcome',scale = alt.Scale(scheme = 'turbo'),
                                                               legend=alt.Legend(title="Emotional Response")),
                                             shape = 'Emotional_outcome').properties(
                                             title='Popularity vs. Duration based on Emotional outcome')
print(f)

h = alt.Chart(req_spot_data).mark_point().encode(x = alt.X('popularity',
                                                       axis= alt.Axis(title='Popularity(%)')),
                                             y = alt.Y('danceability',
                                                       axis = alt.Axis(title = 'Danceability')),
                                             color = alt.Color('Emotional_outcome',scale = alt.Scale(scheme = 'turbo'),
                                                               legend=alt.Legend(title="Track Type"))).properties(
                                             title='Popularity vs. Dancabiltiy based on Emotional outcome & Track Type')
print(h)

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
model = ols(formula='popularity ~ danceability + energy + speechiness + acousticness + C(mode) + liveness + loudness',data=full_spot_data).fit()
model.summary()


# # Modeling for traditional and algorithmic variables combined


# <h3>Model Building</h3>

# In[21]:


# Linear regression to predict popularity using traditional terms.
x1 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 'explicit']]
Y = full_spot_data[['popularity']]
LR1 = LinearRegression()
LR1.fit(x1, Y)
x1_train, x1_test, Y1_train, Y1_test = train_test_split(x1, Y,random_state = 0,test_size=0.25)
Y1_pred = LR1.predict(x1_train)
residuals1 = Y1_train-Y1_pred
mean_residuals1 = np.mean(residuals1)
print("Mean of Residuals {}".format(mean_residuals1))
#Close enough to 0 for our purposes
print(LR1.score(x1, Y))
print(LR1.coef_)


# In[6]:


# Linear regression to predict popularity using algo terms.
x2 = full_spot_data[['danceability','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y = full_spot_data[['popularity']]
LR2 = LinearRegression()
LR2.fit(x2, Y)
x2_train, x2_test, Y2_train, Y2_test = train_test_split(x2, Y,random_state = 0,test_size=0.25)
Y2_pred = LR2.predict(x2_train)
residuals2 = Y2_train-Y2_pred
mean_residuals2 = np.mean(residuals2)
#print("Mean of Residuals {}".format(mean_residuals2))
#Close enough to 0 for our purposes
print(LR2.score(x2, Y))
print(LR2.coef_)


# In[7]:


# Using Polynomial Regression to predict popularity using traditional terms
x1 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 'explicit']]
Y = full_spot_data[['popularity']]

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(x1)
X_train, X_test, y_train, y_test = train_test_split(poly_features, Y, test_size=0.3, random_state=42)

poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train, y_train)

poly_reg_y_predicted = poly_reg_model.predict(X_test)
from sklearn.metrics import mean_squared_error
poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
poly_reg_rmse

print(poly_reg_model.score(poly_features, Y))
print(poly_reg_model.coef_)


# In[8]:


# Polynomial regression to predict popularity using algo terms.
x2 = full_spot_data[['danceability','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y = full_spot_data[['popularity']]
poly_features2 = poly.fit_transform(x2)
X_train, X_test, y_train, y_test = train_test_split(poly_features2, Y, test_size=0.3, random_state=42)

poly_reg_model2 = LinearRegression()
poly_reg_model2.fit(X_train, y_train)

poly_reg_y_predicted2 = poly_reg_model2.predict(X_test)
from sklearn.metrics import mean_squared_error
poly_reg_rmse2 = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
poly_reg_rmse2

print(poly_reg_model2.score(poly_features2, Y))
print(poly_reg_model2.coef_)


# In[9]:


# Predictiong dancability by using algo terms using linear regression
X3 = full_spot_data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y1 = full_spot_data[['danceability']]
LR3 = LinearRegression()
LR3.fit(X3,Y1)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3,Y1,random_state = 0,test_size = 0.25)
Y3_pred = LR3.predict(X3_train)
residuals3 = Y3_train-Y3_pred
mean_residuals3 = np.mean(residuals3)
print(LR3.score(X3, Y1))
print(LR3.coef_)


# In[10]:


# Linear regression by using traditional terms for predicting dancability
X4 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 'explicit']]
Y1 = full_spot_data[['danceability']]
LR4 = LinearRegression()
LR4.fit(X4,Y1)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4,Y1,random_state = 0,test_size = 0.25)
Y4_pred = LR4.predict(X4_train)
residuals4 = Y4_train-Y4_pred
mean_residuals4 = np.mean(residuals4)
print(LR4.score(X4, Y1))
print(LR4.coef_)


# In[11]:


# Polynomial regression by using algo terms for dancability
X3 = full_spot_data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y1 = full_spot_data[['danceability']]
poly_features3 = poly.fit_transform(X3)
X3a_train, X3a_test, Y3a_train, Y3a_test = train_test_split(poly_features3, Y1, test_size=0.3, 
                                                            random_state=42)

poly_reg_model3 = LinearRegression()
poly_reg_model3.fit(X3a_train, Y3a_train)

poly_reg_y_predicted3 = poly_reg_model3.predict(X3a_test)
from sklearn.metrics import mean_squared_error
poly_reg_rmse3 = np.sqrt(mean_squared_error(Y3a_test, poly_reg_y_predicted3))
poly_reg_rmse3

print(poly_reg_model3.score(poly_features3, Y1))
print(poly_reg_model3.coef_)


# In[12]:


# Polynomial regression by using traditional terms for predicting dancability.
X4 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 'explicit']]
Y1 = full_spot_data[['danceability']]
poly_features4 = poly.fit_transform(X4)
X4a_train, X4a_test, Y4a_train, Y4a_test = train_test_split(poly_features4, Y1, test_size=0.3, random_state=42)

poly_reg_model4 = LinearRegression()
poly_reg_model4.fit(X4a_train, Y4a_train)

poly_reg_y_predicted4 = poly_reg_model4.predict(X4a_test)
poly_reg_rmse4 = np.sqrt(mean_squared_error(Y4a_test, poly_reg_y_predicted4))
poly_reg_rmse4

print(poly_reg_model4.score(poly_features4, Y1))
print(poly_reg_model4.coef_)


# In[13]:


# Linear regression using algo terms for determining duration of track.
X5 = full_spot_data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y2 = full_spot_data[['duration_ms']]
LR5 = LinearRegression()
LR5.fit(X5,Y2)
X5_train, X5_test, Y5_train, Y5_test = train_test_split(X5,Y2,random_state = 0,test_size = 0.25)
Y5_pred = LR5.predict(X5_train)
residuals5 = Y5_train-Y5_pred
mean_residuals5 = np.mean(residuals5)
print(LR5.score(X5, Y2))
print(LR5.coef_)


# In[14]:


# Linear regression by using traditional terms for predicting duration of track.
X6 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'explicit']]
Y2 = full_spot_data[['duration_ms']]
LR6 = LinearRegression()
LR6.fit(X6,Y2)
X6_train, X6_test, Y6_train, Y6_test = train_test_split(X6,Y2,random_state = 0,test_size = 0.25)
Y6_pred = LR6.predict(X6_train)
residuals6 = Y6_train-Y6_pred
mean_residuals6 = np.mean(residuals6)
print(LR6.score(X6, Y2))
print(LR6.coef_)


# In[15]:


# Using Polynomial regression to predict duration of track using algo terms
X5 = full_spot_data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y2 = full_spot_data[['duration_ms']]
poly_features5 = poly.fit_transform(X5)
X5a_train, X5a_test, Y5a_train, Y5a_test = train_test_split(poly_features5, Y2, test_size=0.3, random_state=42)

poly_reg_model5 = LinearRegression()
poly_reg_model5.fit(X5a_train, Y5a_train)

poly_reg_y_predicted5 = poly_reg_model5.predict(X5a_test)
poly_reg_rmse5 = np.sqrt(mean_squared_error(Y5a_test, poly_reg_y_predicted5))
poly_reg_rmse5

print(poly_reg_model5.score(poly_features5, Y2))
print(poly_reg_model5.coef_)


# In[16]:


# Using Polynomial regression to predict duration of track using traditional terms
X6 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'explicit']]
Y2 = full_spot_data[['duration_ms']]
poly_features6 = poly.fit_transform(X6)
X6a_train, X6a_test, Y6a_train, Y6a_test = train_test_split(poly_features6, Y2, test_size=0.3, random_state=42)

poly_reg_model6 = LinearRegression()
poly_reg_model6.fit(X6a_train, Y6a_train)

poly_reg_y_predicted6 = poly_reg_model6.predict(X6a_test)
poly_reg_rmse6 = np.sqrt(mean_squared_error(Y6a_test, poly_reg_y_predicted6))
poly_reg_rmse6

print(poly_reg_model6.score(poly_features6, Y2))
print(poly_reg_model6.coef_)


# <h3>Using KNN Regression</h3>

# In[23]:


X7 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 
                    'explicit']]
Y3 = full_spot_data[['danceability']]

from sklearn.neighbors import KNeighborsRegressor

X7a_train, X7a_test, Y7a_train, Y7a_test = train_test_split(X4, Y1, test_size=0.25,
                                                            random_state=42)
knn = KNeighborsRegressor(n_neighbors=5)
print(knn)
knn.fit(X7a_train,Y7a_train)
Y7a_pred_test = knn.predict(X7a_test)
print(knn.score(X4, Y1))
#print(knn.coef_)


# In[33]:


X8 = full_spot_data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y3 = full_spot_data[['danceability']]

X8a_train, X8a_test, Y8a_train, Y8a_test = train_test_split(X8, Y3, test_size=0.25,
                                                            random_state=42)
knn1 = KNeighborsRegressor(n_neighbors=5)
print(knn)
knn1.fit(X8a_train,Y8a_train)
Y8a_pred_test = knn1.predict(X8a_test)
print(knn1.score(X8, Y3))


# <p>Let us try using KNN on time duration.</p>

# In[19]:


# Using traditional variables
X9 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature','explicit']]
Y4 = full_spot_data[['duration_ms']]

X9a_train, X9a_test, Y9a_train, Y9a_test = train_test_split(X9, Y4, test_size=0.25,
                                                            random_state=42)
knn2 = KNeighborsRegressor(n_neighbors=5)
print(knn2)
knn2.fit(X9a_train,Y9a_train)
Y9a_pred_test = knn2.predict(X9a_test)
print(knn2.score(X9, Y4))


# In[20]:


X10 = full_spot_data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y4 = full_spot_data[['duration_ms']]

X10a_train, X10a_test, Y10a_train, Y10a_test = train_test_split(X10, Y4, test_size=0.25,
                                                            random_state=42)
knn3 = KNeighborsRegressor(n_neighbors=5)
print(knn3)
knn3.fit(X10a_train,Y10a_train)
Y10a_pred_test = knn3.predict(X10a_test)
print(knn3.score(X10, Y4))


# In[26]:


X11 = full_spot_data[['key','mode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 'explicit']]
Y5 = full_spot_data[['popularity']]

X11a_train, X11a_test, Y11a_train, Y11a_test = train_test_split(X11, Y5, test_size=0.25,
                                                            random_state=42)
knn4 = KNeighborsRegressor(n_neighbors=5)
print(knn4)
knn4.fit(X11a_train,Y11a_train)
Y11a_pred_test = knn4.predict(X11a_test)
print(knn4.score(X11, Y5))


# In[28]:


X12 = full_spot_data[['danceability','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y5 = full_spot_data[['popularity']]

X12a_train, X12a_test, Y12a_train, Y12a_test = train_test_split(X12, Y5, test_size=0.25,
                                                            random_state=42)
knn5 = KNeighborsRegressor(n_neighbors=5)
print(knn5)
knn5.fit(X12a_train,Y12a_train)
Y12a_pred_test = knn5.predict(X12a_test)
print(knn5.score(X12, Y5))
# ------------------------------------------------------------------------------------------------------------------------------------------------|


X = full_spot_data[['energy', 'instrumentalness','acousticness', 'liveness','loudness', 'tempo', 'valence']]
#X = data[['popularity','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
y = full_spot_data['danceability']
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