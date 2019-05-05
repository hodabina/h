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


# # Concatenate

# In[14]:


os.chdir("C:\\Users\\alibi\\OneDrive\\Desktop\\hoawi\\Task")

#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#print(all_filenames)

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False)


# In[15]:


#find all csv files in the folder
#use glob pattern matching -> extension = 'csv'
#save result in list -> all_filenames
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]


# In[16]:


#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False)


# In[41]:


combined_csv


# In[ ]:





# In[ ]:




