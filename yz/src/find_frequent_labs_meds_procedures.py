
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from IPython.display import display

pd.options.display.max_columns = 100
pd.options.display.max_rows = 20


# In[8]:


disease_code = 309

SD_patient = pd.read_csv('{}_Patient.csv'.format(disease_code))
SD_med = pd.read_csv('short_{}_Med.csv'.format(disease_code))
SD_lab = pd.read_csv('short_{}_Lab.csv'.format(disease_code))
SD_procedure = pd.read_csv('short_{}_Procedure.csv'.format(disease_code))


# In[9]:


def shortest_name(name_lst):
    return min(name_lst, key=len).lower()

patient_num = SD_patient.shape[0]
frequent_threshold = 0.05*patient_num
print 'threshold is {}'.format(frequent_threshold)

grouped = SD_lab[SD_lab.lab_val.notnull()].groupby('mrd_lab_id')
lab_count_by_id = grouped.agg({'lab_nm': shortest_name}).assign(count=grouped.size())

lab_count_by_nm = lab_count_by_id.groupby('lab_nm').sum()
ignore_lab_nm = {'final report', 'color', 'additional comments', 'comment', 
                 'unit number', 'unit tag comment', 'units ordered'}

lab_count_by_nm = lab_count_by_nm[(lab_count_by_nm['count'] > frequent_threshold).values
                                  & ~lab_count_by_nm.index.isin(ignore_lab_nm)]
lab_count_by_nm.to_csv('frequent_labs.csv')
lab_count_by_nm


# In[10]:


grouped = SD_med.groupby('mrd_med_id')
med_count_by_id = grouped.agg({'generic_nm': shortest_name}).assign(count=grouped.size())
med_count_by_nm = med_count_by_id.groupby('generic_nm').sum()
med_count_by_nm = med_count_by_nm[med_count_by_nm['count'] > frequent_threshold]

med_count_by_nm.to_csv('frequent_meds.csv')

med_count_by_nm


# In[11]:


grouped = SD_procedure[SD_procedure['order_type_desc'] != 'LAB'].groupby('order_cpt_cd')
procedure_count_by_id = grouped.agg({'order_nm': shortest_name}).assign(count=grouped.size())
procedure_count_by_nm = procedure_count_by_id.groupby('order_nm').sum()
procedure_count_by_nm = procedure_count_by_nm[procedure_count_by_nm['count'] > frequent_threshold]
procedure_count_by_nm.to_csv('frequent_procedures.csv')
procedure_count_by_nm


# In[ ]:




