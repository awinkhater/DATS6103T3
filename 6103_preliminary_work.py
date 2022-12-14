#Group 3 Final
# Alex Khater, Vaishnavi Nagraj, Aditya Nayak, Pooja Chandrashekara 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sms
from sklearn.linear_model import LinearRegression
#I added researchpy and scipy.stats for the t-tests
import researchpy as rp
import scipy.stats as stats
#%%
ds1=  pd.read_csv("spotify_dataset.csv") # use this instead of hitting the server if csv is on local
# %%
'''First let's get a feel for our dataset within our code'''
ds1.head()
# %%
'''Now We Should Start Cleaning'''
print(ds1.isnull().sum().sum())
'''With only 3 null values, It should be inconsequential to drop their respective rows'''
ds= ds1.dropna()
#%%
'''For ease of coding, we are going to rename one column:'''
ds.rename(columns={'mode': 'Mmode'}, inplace= True)
# %%
'''Now lets get some info on the dataset after cleaning'''
ds.info()
# %%
'''Because we are focused on (semi) traditional music theory metrics,
lets look at the following variables: key, loudness, tempo, 
time_signature, track_genre, popularity, mode, energy, and valence.'''
plt.hist([ds["key"]], alpha=0.5, edgecolor="black", label=['Key Plot (C=0, B-Flat=11) '])
plt.legend(loc='upper right')
plt.show()
print("The Key of a song is to represent what sacle of the 13 tones it is (mainly) native to, C=0")

plt.hist([ds["key"]], alpha=0.5, edgecolor="black", label=['Key Plot (C=0, B-Flat=11) '])
plt.legend(loc='upper right')
plt.show()
print("The Key of a song is to represent what sacle of the 13 tones it is (mainly) native to, C=0")

# %%
plt.hist([ds["loudness"]], alpha=0.5, edgecolor="black", label=['Loudness Plot (in db)'])
plt.legend(loc='upper right')
plt.show()
print("This is simply the loudness of a song in decibles")
# %%
plt.hist([ds["tempo"]], alpha=0.5, edgecolor="black", label=['Tempo Plot (in bpm)'])
plt.legend(loc='upper right')
plt.show()
print("This column measures the beats per minute (bpm) of a song. The higher the bpm, the faster the song")
# %%
plt.hist([ds["time_signature"]], alpha=0.5, edgecolor="black", label=['Tempo Plot (in Beats per Measure)'])
plt.legend(loc='upper right')
plt.show()
print("This measures how many beats per measure (the rhytmic standard subdivisions of a song). it ranges from 1-7.")
# %%
plt.hist([ds["popularity"]], alpha=0.5, edgecolor="black", label=['Popularity Plot (in Alg. Rating)'])
plt.legend(loc='upper right')
plt.show()
print("This is an algorithmically generated popularity score for a song from 1 - 100, with 100 being the most popular.")

# %%
plt.hist([ds["valence"]], alpha=0.5, edgecolor="black", label=['Valence Plot (Alg Rating)'])
plt.legend(loc='upper right')
plt.show()
print("This is an algorthmically generated value from 0-1 measuring a song's postiveness with 1 being most positve")
#%%
plt.hist([ds["energy"]], alpha=0.5, edgecolor="black", label=['Energy Plot (Alg Rating)'])
plt.legend(loc='upper right')
plt.show()
print("This is an algorthmically generated value from 0-1 measuring a song's energy with 1 being most energetic")
# %%
GEN= [ "Minor Key", "Major Key" ]
plt.pie(ds.Mmode.value_counts(), labels=GEN)
plt.axis('equal')
plt.title('Moduality Distribution ')
plt.show()
print("This is a binary variable showing whether the song is in a minor (0) or major (1) key")
print("GRAPH NOTE: THERE ARE LABELS, BUT THEY ARE NOT VISIBLE IN DARK MODE")

# %%
#Unit 2: Correlation Plots TBD
ds.corr()
#%%
#Unit 3: Modeling

# %%
#linearity Check
ds.plot.scatter(x='tempo', y='popularity')
plt.xlabel('tempo', fontsize=18)
plt.ylabel('popularity', fontsize=18)
plt.show()
#Not very linear
ds.plot.scatter(x='valence', y='popularity')
plt.xlabel('Valence', fontsize=18)
plt.ylabel('popularity', fontsize=18)
plt.show()
# Not really linear
ds.plot.scatter(x='loudness', y='popularity')
plt.xlabel('loudness', fontsize=18)
plt.ylabel('popularity', fontsize=18)
plt.show()
#Sort of linear
ds.plot.scatter(x='energy', y='popularity')
plt.xlabel('energy', fontsize=18)
plt.ylabel('popularity', fontsize=18)
plt.show()
#High-end Linear
#Not Graphing Key, Mmode, time_signature because they are all non-continuous and wont scatterplot well
#None are very strongly correlated to popularity

# %%
#Normality Check
sms.qqplot(ds['popularity'])
plt.show()
#Approximately Normal
# %%
#Mean Residuals

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


  
#Maximal Model with All Variables
x1=ds[['key','Mmode', 'loudness', 'tempo', 'time_signature', 'duration_ms', 'explicit']]
Y=ds[['popularity']]
LR1 = LinearRegression()
LR1.fit(x1, Y)
x1_train, x1_test, Y1_train, Y1_test = train_test_split(x1, Y,random_state = 0,test_size=0.25)
Y1_pred = LR1.predict(x1_train)
residuals1 = Y1_train-Y1_pred
mean_residuals1 = np.mean(residuals1)
print("Mean of Residuals {}".format(mean_residuals1))
#Close enough to 0 for our purposes



# %%
print(LR1.score(x1, Y))
# %%
x2=ds[['danceability','energy', 'speechiness', 'acousticness', 'instrumentalness','liveness']]
Y=ds[['popularity']]
LR2 = LinearRegression()
LR2.fit(x2, Y)
x2_train, x2_test, Y2_train, Y2_test = train_test_split(x2, Y,random_state = 0,test_size=0.25)
Y2_pred = LR2.predict(x2_train)
residuals2 = Y2_train-Y2_pred
mean_residuals2 = np.mean(residuals2)
#print("Mean of Residuals {}".format(mean_residuals2))
#Close enough to 0 for our purposes
print(LR2.score(x2, Y))
#%%
#polymodel
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
# %%
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features2 = poly.fit_transform(x2)
X_train, X_test, y_train, y_test = train_test_split(poly_features2, Y, test_size=0.3, random_state=42)

poly_reg_model2 = LinearRegression()
poly_reg_model2.fit(X_train, y_train)

poly_reg_y_predicted = poly_reg_model2.predict(X_test)
from sklearn.metrics import mean_squared_error
poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
poly_reg_rmse

print(poly_reg_model2.score(poly_features2, Y))
# %%
