#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
import geopandas as gpd
import geopy.distance
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from pandas.core.common import flatten


# In[2]:


def load_plt_files(user):
    user_data = []
    
    # Run over all files associated with the current user
    for filepath in glob('data/{}/Trajectory/*plt'.format(user)):
        # Get the trajectory id from the filename
        trajectory = int(filepath.split(os.sep)[-1].split('.')[0])
        
        # Read all the lines, except the first 6. These do not matter
        lines = open(filepath).readlines()[6:]

        # Strip each line, and convert them into an array
        data  = list(map(lambda l: l.strip().split(","), lines))

        # Create a function to turn each data array into a dictionary for ease of use
        # Some of the columns do not matter for our usecase
        group = lambda item: {"user": int(user),
                              "lat": float(item[0]), 
                              "lon": float(item[1]),
                              "trajectory": trajectory,
                              "altitude": float(item[3]) if item[3] != -707 else None,
                              "datetime": datetime.datetime.strptime("{} {}".format(item[5], item[6]), '%Y-%m-%d %H:%M:%S')}

        # Convert the data into a list of dictionaries
        data = list(map(group, data))
        user_data += data
    
    return user_data


# In[3]:


# Find all avaliable users
all_users = list(map(lambda s: s.split(os.sep)[-1], glob('data/*')))
print('There are ' + str(len(all_users)) + ' users')


# In[4]:


# Create a DataFrame from my data set
print("Loading all data. This might take a 5-10 minutes depending on your computer")
all_data = []
for data in map(load_plt_files, all_users):
    all_data += data

full_df = pd.DataFrame(all_data)


# In[6]:


# Check my full_df DataFrame.

print(full_df.head())


# In[7]:


print('The shape of our DataFrame is: ')
print(full_df.shape)


# In[8]:


# I will check if the values of the lat column lie within [-90,90] and 
# the Lon column lies within [-180, 180].

print(full_df[full_df['lat']>=90])
print(full_df[full_df['lat']<=-90])
print(full_df[full_df['lon']>=180])
print(full_df[full_df['lon']<=-180])

# I will delete the row with index
# index       user         lat        lon      
# 5067886    20       400.166667  116.21539  

full_df=full_df.drop(labels=5067886, axis=0)


# In[9]:


print(full_df.describe())


# In[10]:


# I will check my data set for missing values and duplicates.

# Missing values
print(full_df.isnull().sum())


# In[11]:


# Duplicates
print('There are ' + str(full_df.duplicated().sum()) + " duplicate values")
full_df.drop_duplicates(inplace = True)
print('The shape of the new data set is')
print(full_df.shape)


# In[12]:


# Now I will convert my pandas DataFrame to a pandas GeoDataFrame
print("Converting my full_df DataFrame to a pandas GeoDataFrame. This might take around 10 minutes")
gdf = gpd.GeoDataFrame(full_df, geometry=gpd.points_from_xy(full_df.lon, full_df.lat))


# In[14]:


print(gdf.head())


# In[15]:


# I will create a plot with my whole data set.
# I used the world geopandas set to visualize my GPS points on the map of China.


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
fig, ax = plt.subplots()
ax = world[world.name == 'China'].plot(color='#C9F8F2', edgecolor='black')
ax.plot(gdf["lon"], gdf["lat"], marker='o', color='red', markersize=2, linestyle="None")
ax.set_xlabel("Longitute")
ax.set_ylabel("Latitute")
ax.set_title('GPS points for all the users')
plt.xlim(70, 140)
plt.ylim(15, 55)

plt.savefig('output_2.png')
plt.show()


# In[16]:


# I will group the dataframe and make a visualization for every user.

grouped = gdf.groupby('user')

gdf_list=[]

for name, group in grouped:
    gdf_list.append(group)

    
def plot_function(df):
    # fig, ax = plt.subplots()
    ax = world[world.name == 'China'].plot(color='#C9F8F2', edgecolor='black')
    ax.plot(df["lon"], df["lat"], marker='o', color='red', markersize=2, linestyle="None", label='User {}'.format(df.iloc[0, 0]))
    ax.set_xlabel("Longitute")
    ax.set_ylabel("Latitute")
    ax.legend()
    ax.set_title('GPS points for the user {}'.format(df.iloc[0, 0]))
    plt.xlim(70, 140)
    plt.ylim(15, 55)
    plt.savefig('output{}.png'.format(df.iloc[0, 0]))
    return plt.show()
    

    
for ele in gdf_list:
    print(plot_function(ele), ele.shape)


# In[17]:


# Now I will create two Distribution plots to see where the majority of the GPS points 


g=sns.distplot(gdf["lat"])
plt.xlim(38, 41)
g.set(xlabel="Latitude", title="Kernel Density Estimate of Latitute")
fig.set_size_inches([12, 7])
plt.savefig('KDE_Lat.png')
plt.show()


# In[18]:


g=sns.distplot(gdf["lon"])
plt.xlim(115, 120)
g.set(xlabel="Longitude", title="Kernel Density Estimate of Longitude")
fig.set_size_inches([12, 7])
plt.savefig('KDE_Lon.png')
plt.show()


# In[19]:


# I want to create a new table with columns tmin, tmax distance.
# I will reindex my grouped DataFrames in the grouped list.
# I need it because otherwise I have problem with the indexes at the following function.

for ele in gdf_list:
    ele.index=range(len(ele))

def distance_coord(df):
    """This function computes the total distance covered for a user.
    The possible dfs are the DataFrames for each individual via Groupby"""
    temp_list_distance=[]
    list_distance=[]
    for i in range(len(df)-1):
        coord1 = (df['lat'][i], df['lon'][i])
        coord2 = (df['lat'][i+1], df['lon'][i+1])
        dist = geopy.distance.geodesic(coord1, coord2).km
        temp_list_distance.append(dist)
    list_distance.append(sum(temp_list_distance))   
    return(list_distance)   


# In[20]:


# I will run the distance_coord function for every element (DataFrame) in the gdf_list.

total_distance=[]

for item in gdf_list:
    total_distance.append(distance_coord(item))


# In[21]:


# I will flatten the total_distance list of lists to a single flat list.

total_distance = list(flatten(total_distance))


# In[22]:


# I will create a function to compute the tmin, tmax and timedelta for every user.

def compute_tmin_tmax_timedelta(df):
    'This function computes the min, max timestamp and the timedelta for evry user'
    tmin = df['datetime'].min()
    tmax = df['datetime'].max()
    timedelta = tmax-tmin
    return tmin, tmax, timedelta
    


# In[23]:


# I will run the compute_tmin_tmax_timedelta function for every element (DataFrame) in the gdf_list.

min_time = []
max_time = []
timedelta = []

for item in gdf_list:
    min_time.append(compute_tmin_tmax_timedelta(item)[0])
    max_time.append(compute_tmin_tmax_timedelta(item)[1])
    timedelta.append(compute_tmin_tmax_timedelta(item)[2])


# In[24]:


# I will crreate a new data frame with the lists of users, min_time , max_time, timedelta, and total_distance.

new_df_per_user = pd.DataFrame(list(zip(all_users, min_time, max_time, timedelta, total_distance)), 
                               columns = ['user', 'tmin', 'tmax', 'timedelta', 'distance (km)'])


# In[25]:


print(new_df_per_user.head(2))
print(new_df_per_user.dtypes)


# In[26]:


# Now I will create two Distribution plots to see where the majority of the GPS points 

plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.plot(new_df_per_user['user'], new_df_per_user['distance (km)'], marker="o", linestyle="--", c='b')
ax.set_xticklabels(new_df_per_user.index, rotation=90)
ax.set_xlabel("User")
ax.set_ylabel("Total Distance in Km")
ax.set_title("Distance covered per user")
fig.set_size_inches([12, 7])
plt.savefig('Distance covered per user1.png')
plt.show()


# In[27]:


plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.bar(new_df_per_user['user'], new_df_per_user['distance (km)'], color='b')
ax.set_xticklabels(new_df_per_user.index, rotation=90)
ax.set_xlabel("User")
ax.set_ylabel("Total Distance in Km")
ax.set_title("Distance covered per user")
fig.set_size_inches([12, 7])
plt.savefig('Distance covered per user1.png')
plt.show()


# In[28]:


# I will create a function to count how many users have been tracked for:
# 1. 1 week,
# 2. 1week ~ 1month
# 3. 1month - 1year
# 4. 1year - more.


def class_period(df):
    s1 = sum(df['timedelta'] < datetime.timedelta(days = 7))
    s2 = sum(df['timedelta'] <= datetime.timedelta(days = 30))-s1
    s3 = sum(df['timedelta'] <= datetime.timedelta(days = 365))-s2
    s4 = sum(df['timedelta'] > datetime.timedelta(days = 365))
    return s1, s2, s3, s4
   


# In[29]:


# Convert the tuple class_period(new_df_per_user) to list.

classif_period = list(class_period(new_df_per_user))


# In[30]:


# I will create a small new DataFrame with first column classif_period and second column the time deltas.

timedeltas_list = ['< 1week', '1week ~ 1month', '1month ~ 1year', '≥1year']
new_df_classif_period = pd.DataFrame(list(zip(classif_period, timedeltas_list)), columns = ['Sum of users', 'duration'])


# In[31]:


print(new_df_classif_period)


# In[32]:


plt.style.use("ggplot")
fig, ax = plt.subplots()
ax.bar(new_df_classif_period['duration'], new_df_classif_period['Sum of users'], color='b')
ax.set_xlabel("Classes")
ax.set_ylabel("Users")
ax.set_title("Distribution of users by data collection period")
fig.set_size_inches([12, 7])
plt.savefig('Distribution of users by data collection period.png')
plt.show()


# In[33]:


plt.figure(figsize=(16,8))
# plot chart
ax1 = plt.subplot(121, aspect='equal')
new_df_classif_period.plot(kind='pie', y = 'Sum of users', autopct='%1.1f%%', ax=ax1,
 startangle=90, shadow=False, labels=new_df_classif_period['duration'], legend = True, fontsize=14)
plt.savefig('PieChart.png')


# In[ ]:




