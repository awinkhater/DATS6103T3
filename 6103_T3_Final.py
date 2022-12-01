#Group 3 Final
# Alex Khater, Vaishnavi Nagraj, Aditya Nayak, Pooja Chandrashekara 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
