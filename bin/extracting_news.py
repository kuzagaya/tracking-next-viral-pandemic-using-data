#!/usr/bin/env python
# coding: utf-8

# # Using NewsApi to download news

# In[42]:


API_KEY = 'API_KEY'
from newsapi import NewsApiClient


api = NewsApiClient(api_key = API_KEY)
news = api.get_top_headlines(q= 'health', category = 'health')


# In[43]:


articles = []

for i in range(len(news['articles'])):
    articles.append(news['articles'][i]['title'].split('|')[0].split('-')[0])


# In[44]:


articles

