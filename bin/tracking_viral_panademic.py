#!/usr/bin/env python
# coding: utf-8

# # Installing the required libraries

# In[16]:


#!pip install geonamescache


# # Importing the Libraries

# In[17]:


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from math import asin
from unidecode import unidecode
from geonamescache import GeonamesCache

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

gc = GeonamesCache()


# # Loading the headlines data

# In[18]:


headline_file = open('../datasets/headlines.txt','r')
headlines = [line.strip() for line in headline_file.readlines()]
print(f'{len(headlines)} headlines has been loaded from directory')


# In[19]:


def name_to_regex(name):
    decoded_name = unidecode(name)
    if name != decoded_name:
        regex = fr'\b({name}|{decoded_name})\b'
    else:
        regex = fr'\b{name}\b'
    return re.compile(regex, flags=re.IGNORECASE)


# In[20]:


countries = [country['name'] for country in gc.get_countries().values()]
country_to_name = {name_to_regex(name): name for name in countries}

cities = [city['name'] for city in gc.get_cities().values()]
city_to_name = {name_to_regex(name): name for name in cities}


# # Finding location in text

# In[21]:


def get_name_in_text(text, dictionary):
    for regex, name in sorted(dictionary.items(), key = lambda x: x[1]):
        if regex.search(text):
            return name
    return None


# In[22]:


matched_countries = [get_name_in_text(headline, country_to_name) for headline in headlines]
matched_cities = [get_name_in_text(headline, city_to_name) for headline in headlines]

data = {
    'Headline': headlines,
    'City': matched_cities,
    'Country': matched_countries
}

df = pd.DataFrame(data)
df.head()


# In[23]:


df[['City','Country']].describe()


# In[24]:


of_cities = df[df.City == 'Of'][['City','Headline']]
print(of_cities.head(10).to_string(index = False))


# In[25]:


def get_cities_in_headline(headline):
    cities_in_headline = set()
    for regex, name in city_to_name.items():
        match = regex.search(headline)
        if match:
            if headline[match.start()].isupper():
                cities_in_headline.add(name)
    return list(cities_in_headline)

df['Cities'] = df['Headline'].apply(get_cities_in_headline)
df['Num_cities'] = df['Cities'].apply(len)
df_multiple_cities = df[df.Num_cities > 1]
num_rows, _ = df_multiple_cities.shape
print(f"{num_rows} headlines match multiple cities")


# In[26]:


df.head()


# ## Sampling multi-cities headlines

# In[27]:


ten_cities = df_multiple_cities[['Cities', 'Headline']].head(10)
print(ten_cities.to_string(index = False))


# Short, invalid city names are getting matched to the headlines along with longer, more correct location names. One solution is simply to assign the longest city-name as the representative location if more than one matched city is found.

# In[28]:


def get_longest_city(cities):
    if cities:
        return max(cities, key = len)
    return None
df['City'] = df['Cities'].apply(get_longest_city)


# In[29]:


df_countries = df[df.Country.notnull()][['City','Country','Headline']]
print(df_countries.to_string(index = False))


# In[30]:


df.drop('Country', axis = 1, inplace = True)


# # Exploring the unmatched headlines

# In[31]:


df_unmatched = df[df.City.isnull()]
num_unmatched = len(df_unmatched)
print(f'{num_unmatched} has no city matches')
print(df_unmatched.head(10)[['Headline']].values)


# In[32]:


df = df[~df.City.isnull()][['City','Headline']]


# # Visulizing and Clustering the Extracted Location Data

# In[33]:


latitide, longitude = [],[]
for city_name in df.City.values:
    city = max(gc.get_cities_by_name(city_name), key = lambda x: list(x.values())[0]['population'])
    city = list(city.values())[0]
    latitide.append(city['latitude'])
    longitude.append(city['longitude'])
df = df.assign(Latitude = latitide, Longitude = longitude)


# In[34]:


coordinates = df[['Latitude', 'Longitude']].values
k_values = range(1,10)
inertia_values = []

for k in k_values:
    inertia_values.append(KMeans(k).fit(coordinates).inertia_)

plt.figure(figsize=(18,7))
plt.plot(range(1,10), inertia_values)
plt.title('Elbow Method', fontsize = 18)
plt.xlabel('Value of K', fontsize = 18)
plt.ylabel('Inertia', fontsize = 18)
plt.show()


# ### The "elbow" within our Elbow plot points to a K of 3 That K-value is very low, limiting our scope to at-most 3 different geographic territories

# In[35]:


def plot_clusters(clusters, longitude, latitude):
    fig = px.scatter_geo(lat = latitide,lon = longitude, color = clusters, symbol = clusters)
    fig.update_layout(title = 'World map', title_x=0.5)
    fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

df['Cluster'] = KMeans(3).fit_predict(coordinates)
plot_clusters(df.Cluster, df.Longitude, df.Latitude)


# In[36]:


df['Cluster'] = KMeans(6).fit_predict(coordinates)
plot_clusters(df.Cluster, df.Longitude, df.Latitude)


# # Using DBSCAN algorithm

# In[37]:


def great_circle_distance(coord1, coord2, radius=3956):
    if np.array_equal(coord1, coord2):
        return 0.0 

    coord1, coord2 = np.radians(coord1), np.radians(coord2)
    delta_x, delta_y = coord2 - coord1
    haversin = np.sin(delta_x / 2) ** 2 + np.product([np.cos(coord1[0]),
                                                   np.cos(coord2[0]), 
                                                   np.sin(delta_y / 2) ** 2])
    return  2 * radius * asin(haversin ** 0.5)


# In[38]:


metric = great_circle_distance
dbscan = DBSCAN(eps = 250, min_samples = 3, metric = metric)
df['Cluster'] = dbscan.fit_predict(coordinates)


# In[39]:


no_outliers = df[df.Cluster > -1]

fig = px.scatter_geo(lat = no_outliers.Latitude ,lon = no_outliers.Longitude, color = no_outliers.Cluster, symbol = no_outliers.Cluster)
fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# # Assigning country codes to cities

# In[40]:


def get_country_code(city_name):
    city = max(gc.get_cities_by_name(city_name), 
               key=lambda x: list(x.values())[0]['population'])
    return list(city.values())[0]['countrycode']

df['Country_code'] = df.City.apply(get_country_code)


# # Seperating the cities and global cities

# In[41]:


df_us = df[df.Country_code == 'US']
df_not_us = df[df.Country_code != 'US']


# # Re-Clustering the data

# In[42]:


def re_cluster(input_df, eps):
    input_coord = input_df[['Latitude','Longitude']].values
    dbscan = DBSCAN(eps = eps, min_samples = 3, metric = great_circle_distance)
    clusters = dbscan.fit_predict(input_coord)
    input_df = input_df.assign(Cluster = clusters)
    return input_df[input_df.Cluster >-1]

df_not_us = re_cluster(df_not_us, 250)
df_us = re_cluster(df_us, 125)


# In[43]:


df_not_us


# In[44]:


df_us


# # Extracting the insights from clusters

# In[45]:


# Grouping cities by clusters 
groups = df_not_us.groupby('Cluster')
print(f'{len(groups)} Non-US clusters have been detected')


# In[46]:


sorted_groups = sorted(groups, key = lambda x: len(x[1]), reverse=True)
group_id, largest_group = sorted_groups[0]

group_size = len(largest_group)
print(f'Largest clusters contains {group_size} headlines')


# # Computing the cluster centrality

# In[47]:


def compute_centrality(group):
    group_coords = group[['Latitude', 'Longitude']].values
    center = group_coords.mean(axis = 0)
    distance_to_center = [great_circle_distance(center, coord) for coord in group_coords]
    group['Distance_to_center'] = distance_to_center  


# In[48]:


def sort_by_centrality(group):
    compute_centrality(group)
    return group.sort_values(by = ['Distance_to_center'], ascending = True)

largest_group = sort_by_centrality(largest_group)
for headline in largest_group.Headline.values[:5]:
    print(headline)


# # Finding the top three countries in the largest cluster

# In[49]:


from collections import Counter

def top_countries(group):
    countries = [gc.get_countries()[country_code]['name'] for country_code in group.Country_code.values]
    return Counter(countries).most_common(3)

print(top_countries(largest_group))


# In[50]:


for _, group in sorted_groups[1:5]:
    sorted_group = sort_by_centrality(group)
    print(top_countries(sorted_group))
    for headline in sorted_group.Headline.values[:5]:
        print(headline)
    print('\n')


# # Plotting DBSCAN cluster

# In[51]:


fig = px.scatter_geo(lat = df_us.Latitude,lon = df_us.Longitude, color = df_us.Cluster , symbol = df_us.Cluster, scope = 'usa')
fig.update_layout(title = 'World map', title_x=0.5)
fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# # Summarizing content within the largest US clusters

# In[52]:


us_groups = df_us.groupby('Cluster')
us_sorted_groups = sorted(us_groups, key = lambda x: len(x[1]), reverse = True)

for _, group in us_sorted_groups[:5]:
    sorted_group = sort_by_centrality(group)
    for headline in sorted_group.Headline.values[:5]:
        print(headline)
    print('\n')


# # Plotting Zika clusters

# In[53]:


def count_zika_mention(headlines):
    zika_regex = re.compile(r'\bzika\b', flags = re.IGNORECASE)
    zika_count = 0
    
    for headline in headlines:
        if zika_regex.search(headline):
            zika_count += 1
    return zika_count


# In[54]:


zika_longitude = []
zika_latitude = []
zika_cluster = []
zika_headline = []

for _, group in sorted_groups + us_sorted_groups:
    headlines = group.Headline.values
    zika_count = count_zika_mention(headlines)
    if float(zika_count) / len(headlines) > 0.5:
        zika_longitude.append(list(group.Longitude.values))
        zika_latitude.append(list(group.Latitude.values))
        zika_cluster.append(list(group.Cluster.values))
        zika_headline.append(list(group.Headline.values))
        


# In[55]:


def flatten(l):
    return [item for sublist in l for item in sublist]


# In[56]:


zika_longitude = flatten(zika_longitude)
zika_latitude = flatten(zika_latitude)
zika_cluster = flatten(zika_cluster)


# In[57]:


zika_df = pd.DataFrame({
    'Longitude': zika_longitude,
    'Latitude': zika_latitude,
    'Cluster': zika_cluster
})

zika_df.head()


# In[58]:


fig = px.scatter_geo(lat = zika_df.Latitude,lon = zika_df.Longitude, color = zika_df.Cluster , symbol = zika_df.Cluster)
fig.update_layout(title = 'World map', title_x=0.5)
fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()