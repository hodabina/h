#!/usr/bin/env python
# coding: utf-8

# # Modules

# In[1]:


# OS
import os
from os.path import join
# Pandas
import pandas as pd
# Numpy
import numpy as np
import glob
import re


# # Date and Concatenate

# In[2]:


os.chdir("C:\\Users\\alibi\\OneDrive\\Desktop\\hoawi\\Task")

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#print(all_filenames)


# In[3]:


for f in all_filenames:
    df = pd.read_csv(f)
    df['Date'] = re.findall(r'\d.\d.\-\d.\-\d.\_\d.\-\d\d', f )[0]
    df.to_csv(f)


# In[4]:


#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False)


# In[5]:


foo = lambda x: pd.Series([i for i in reversed(x.split(';'))])
rev = combined_csv['Color;"Some Event";"Some Name";"Count"'].apply(foo)


# In[6]:


rev.columns=['color','event','name','count']
rev['Date'] = combined_csv['Date']
rev.to_csv('combined_csv.csv')


# In[7]:


df = pd.read_csv('C:\\Users\\alibi\\OneDrive\\Desktop\\hoawi\\Task\\combined_csv.csv')
df_reorder = df[['Date','color','event','name','count']] # rearrange column here
df_reorder.to_csv('C:\\Users\\alibi\\OneDrive\\Desktop\\hoawi\\Task\\combined_csv.csv', index=False)


# In[ ]:





# In[ ]:




