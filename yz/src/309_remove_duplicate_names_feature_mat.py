
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import pickle

pd.options.display.max_columns = 100
pd.options.display.max_rows = 1000


# In[3]:

disease_code = 309

lab_case = pd.read_csv('/home/data/sensitive_disease/{}_Lab.csv'.format(disease_code))
med_case = pd.read_csv('/home/data/sensitive_disease/{}_Med.csv'.format(disease_code))
procedure_case = pd.read_csv('/home/data/sensitive_disease/{}_Procedure.csv'.format(disease_code))
patient_case = pd.read_csv('/home/data/sensitive_disease/{}_Patient.csv'.format(disease_code))

print lab_case.shape
print med_case.shape
print procedure_case.shape
print patient_case.shape

lab_control = pd.read_csv('/home/data/sensitive_disease/nonSD5pct_Lab.csv')
med_control = pd.read_csv('/home/data/sensitive_disease/nonSD5pct_Med.csv')
procedure_control = pd.read_csv('/home/data/sensitive_disease/nonSD5pct_Procedure.csv')
patient_control = pd.read_csv('/home/data/sensitive_disease/nonSD5pct_Patient.csv')

print lab_control.shape
print med_control.shape
print procedure_control.shape
print patient_control.shape


# In[5]:

patient_case_person_id = set(med_case['person_id']) | set(lab_case['person_id']) | set(procedure_case['person_id'])

import random
random.seed(1)
patient_case_sample = random.sample(patient_case_person_id, 5000)

med_case_sample = med_case[med_case.person_id.isin(patient_case_sample)]
lab_case_sample = lab_case[lab_case.person_id.isin(patient_case_sample)]
procedure_case_sample = procedure_case[procedure_case.person_id.isin(patient_case_sample)]


# ## lab features

# In[8]:

name_mapping = pd.read_csv('name_mapping/lab_name_mapping.csv')
name_mapping = dict(zip(name_mapping.irregular_name.str.lower(), 
                        name_mapping.regular_name.str.lower()))

def normalize_lab_nm(lab_name):
    norm_name = lab_name.strip().lower().replace(', ', ' ').replace(' - ', ' ')
    if norm_name in name_mapping:
        return name_mapping[norm_name]
    else:
        return norm_name

def select_rows(df):
    return (  (df.lab_val < 9999999.0).values
            & (df.lab_ref_high_val_numeric < 9999999.0).values
            & (df.lab_ref_low_val_numeric < 9999999.0).values
           )

lab_case_sample = lab_case_sample.assign(
    lab_nm_lower_case=lab_case_sample['lab_nm'].apply(normalize_lab_nm),
    lab_ref_high_val_numeric=lab_case_sample['lab_ref_high_val'].apply(pd.to_numeric, errors='ignore'),
    lab_ref_low_val_numeric=lab_case_sample['lab_ref_low_val'].apply(pd.to_numeric, errors='ignore')
)

lab_count_by_nm = pd.DataFrame(lab_309_sample.loc[select_rows].groupby('lab_nm_lower_case').size(), 
                               columns=['count'])
lab_case_frequent_nm = set(lab_count_by_nm[lab_count_by_nm['count'] > 50].index)

print 'lab {} features: '.format(disease_code)
print len(lab_309_frequent_nm)


lab_nonSD['lab_nm_lower_case'] = lab_nonSD['lab_nm'].apply(normalize_lab_nm)
lab_count_by_nm = pd.DataFrame(lab_nonSD.loc[select_rows].groupby('lab_nm_lower_case').size(), 
                               columns=['count'])
lab_nonSD_frequent_nm = set(lab_count_by_nm[lab_count_by_nm['count'] > 50].index)

print 'lab non SD features: '
print len(lab_nonSD_frequent_nm)

print 'lab features: '
lab_frequent_nm = lab_nonSD_frequent_nm | lab_309_frequent_nm
print len(lab_frequent_nm)


# ## medication features

# In[13]:

from collections import OrderedDict

name_mapping = pd.read_csv('name_mapping/med_name_mapping.csv')
name_mapping = dict(zip(name_mapping.irregular_name.str.lower(), 
                        name_mapping.regular_name.str.lower()))

def normalize_med_nm(name):
    norm_name = name.strip().lower().replace(', ', ' ').replace(' - ', ' ').replace('/', '-')                     .replace(' topical', '').replace(' ophthalmic', '').replace(' (product)', '')                    .replace(' (substance)', '').replace(' product', '').replace(' [chemical-ingredient]', '')

    norm_name = '-'.join(OrderedDict.fromkeys(norm_name.split('-')))
    
    if norm_name in name_mapping:
        return name_mapping[norm_name]
    else:
        return norm_name

med_case_sample = med_case_sample.assign(
    generic_nm_lower_case=med_case_sample['generic_nm'].apply(normalize_med_nm)
)
med_count_by_nm = pd.DataFrame(med_case_sample.groupby('generic_nm_lower_case').size(), 
                               columns=['count'])
med_case_frequent_nm = set(med_count_by_nm[med_count_by_nm['count'] > 50].index)

print 'med {} features: '.format(disease_code)
print len(med_case_frequent_nm)

    
med_control = med_control.assign(
    generic_nm_lower_case=med_control['generic_nm'].apply(normalize_med_nm)
)
med_count_by_nm = pd.DataFrame(med_control.groupby('generic_nm_lower_case').size(), 
                               columns=['count'])
med_control_frequent_nm = set(med_count_by_nm[med_count_by_nm['count'] > 50].index)

print 'non SD features: '
print len(med_control_frequent_nm)

print 'total features:'
med_frequent_nm = med_case_frequent_nm | med_control_frequent_nm
print len(med_frequent_nm)


# ## Procedure features

# In[6]:

name_mapping = pd.read_csv('procedure_309_name_mapping.csv')
name_mapping = dict(zip(name_mapping.irregular_name.str.lower(), 
                        name_mapping.regular_name.str.lower()))

def normalize_order_nm(name):
    norm_name = name.strip().lower().replace(', ', ' ').replace(' - ', ' ')
    if norm_name.endswith(' nmh'):
        norm_name = norm_name[:-4]
    elif norm_name.endswith(' nmff'):
        norm_name = norm_name[:-5]
        
    if norm_name in name_mapping:
        return name_mapping[norm_name]
    else:
        return norm_name

def select_rows(df):
    return (  (df.order_type_desc != 'LAB').values
            & (df.order_type_desc != 'VASCULAR LAB').values
           )

procedure_309_sample['order_nm_lower_case'] = procedure_309_sample['order_nm'].apply(normalize_order_nm)
procedure_count_by_nm = pd.DataFrame(procedure_309_sample.loc[select_rows].groupby('order_nm_lower_case').size(), 
                                     columns=['count'])
procedure_309_frequent_nm = set(procedure_count_by_nm[procedure_count_by_nm['count'] > 50].index) 

print 'procedure 309 features: '
print len(procedure_309_frequent_nm)


procedure_nonSD['order_nm_lower_case'] = procedure_nonSD['order_nm'].apply(normalize_order_nm)
procedure_count_by_nm = pd.DataFrame(procedure_nonSD.loc[select_rows].groupby('order_nm_lower_case').size(), 
                                     columns=['count'])
procedure_nonSD_frequent_nm = set(procedure_count_by_nm[procedure_count_by_nm['count'] > 50].index) 

print 'procedure non SD features: '
print len(procedure_nonSD_frequent_nm)

print 'procedure features: '
procedure_frequent_nm = procedure_309_frequent_nm | procedure_nonSD_frequent_nm
print len(procedure_frequent_nm)


# ### find common names in case and control

# In[9]:

get_ipython().system(u'ls *.txt -l')


# In[10]:

lab_common_nm = '\n'.join(sorted(lab_nonSD_frequent_nm & lab_309_frequent_nm))

with open('309_labs_common.txt', 'w') as fp:
    fp.write(lab_common_nm)

get_ipython().system(u'cat 309_labs_common.txt')


# In[11]:

med_common_nm = '\n'.join(sorted(med_nonSD_frequent_nm & med_309_frequent_nm))

with open('309_meds_common.txt', 'w') as fp:
    fp.write(med_common_nm)

get_ipython().system(u'cat 309_meds_common.txt')


# In[12]:

procedure_common_nm = '\n'.join(sorted(procedure_nonSD_frequent_nm & procedure_309_frequent_nm))

with open('309_procedures_common.txt', 'w') as fp:
    fp.write(procedure_common_nm)

get_ipython().system(u'cat 309_procedures_common.txt')


# ### names unique to case

# In[13]:

case = '\n'.join(sorted(lab_309_frequent_nm - lab_nonSD_frequent_nm))

with open('309_labs_case.txt', 'w') as fp:
    fp.write(case)

get_ipython().system(u'cat 309_labs_case.txt')


# In[14]:

case = '\n'.join(sorted(med_309_frequent_nm - med_nonSD_frequent_nm))

with open('309_meds_case.txt', 'w') as fp:
    fp.write(case)

get_ipython().system(u'cat 309_meds_case.txt')


# In[15]:

case = '\n'.join(sorted(procedure_309_frequent_nm - procedure_nonSD_frequent_nm))

with open('309_procedures_case.txt', 'w') as fp:
    fp.write(case)

get_ipython().system(u'cat 309_procedures_case.txt')


# ### names unique to control

# In[16]:

control = '\n'.join(sorted(lab_nonSD_frequent_nm - lab_309_frequent_nm))

with open('309_labs_control.txt', 'w') as fp:
    fp.write(control)

get_ipython().system(u'cat 309_labs_control.txt')


# In[17]:

control = '\n'.join(sorted(med_nonSD_frequent_nm - med_309_frequent_nm))

with open('309_meds_control.txt', 'w') as fp:
    fp.write(control)

get_ipython().system(u'cat 309_meds_control.txt')


# In[18]:

control = '\n'.join(sorted(procedure_nonSD_frequent_nm - procedure_309_frequent_nm))

with open('309_procedures_control.txt', 'w') as fp:
    fp.write(control)

get_ipython().system(u'cat 309_procedures_control.txt')


# ## lab feature matrix

# In[19]:

def select_rows(df):
    return (  (df.lab_val < 9999999.0).values
            & (df.lab_ref_high_val_numeric < 9999999.0).values
            & (df.lab_ref_low_val_numeric < 9999999.0).values
            & (df.lab_ref_low_val_numeric < 9999999.0).values
            & (df.lab_nm_lower_case.isin(lab_frequent_nm)).values
           )
# lab_nonSD_frequent_nm | lab_309_frequent_nm
# df = pd.concat([lab_309_sample[lab_309_sample.lab_nm_lower_case.isin(lab_309_frequent_nm)], 
#                 lab_nonSD[lab_nonSD_frequent_nm]])

# set(lab_309_sample[select_rows].columns) - 


df = pd.concat([lab_309_sample, lab_nonSD])
df = df[select_rows]
high_flag = df['lab_val'] > df['lab_ref_high_val_numeric']
low_flag = df['lab_val'] < df['lab_ref_low_val_numeric'] 
df = df.assign(Lab_high=high_flag, Lab_low=low_flag)

lab_feature_mat = df[['person_id', 'lab_nm_lower_case', 'Lab_high', 'Lab_low']].                   groupby(['person_id', 'lab_nm_lower_case']).max().unstack().fillna(False)
lab_feature_mat.columns = ['_'.join(col_name) for col_name in lab_feature_mat.columns]

print '\n lab feature:'
print lab_feature_mat.shape
display(lab_feature_mat.head(10))

pickle.dump(lab_feature_mat, open('lab_feature_mat.p', 'wb'))


# ## med feature matrix

# In[1]:

med_309_sample


# In[20]:

def select_rows(df):
    return df.generic_nm_lower_case.isin(med_frequent_nm)
           
df = pd.concat([med_309_sample, med_nonSD])
df = df[select_rows]

med_feature_mat = pd.DataFrame(df[['person_id', 'generic_nm_lower_case', 'patient_medications_id']]                                .groupby(['person_id', 'generic_nm_lower_case']).size(), 
                               columns=['Med'])
med_feature_mat = med_feature_mat.applymap(lambda x: 1 if x > 0 else 0).unstack(fill_value=0)
med_feature_mat.columns = ['_'.join(col_name) for col_name in med_feature_mat.columns]

print '\n med feature:'
print med_feature_mat.shape
display(med_feature_mat.head(10))

pickle.dump(med_feature_mat, open('med_feature_mat.p', 'wb'))


# ## procedure feature matrix

# In[21]:

def select_rows(df):
    return (  (df.order_type_desc != 'LAB').values
            & (df.order_type_desc != 'VASCULAR LAB').values
            & (df.order_nm_lower_case.isin(procedure_frequent_nm)).values
           )
           
df = pd.concat([procedure_309_sample, procedure_nonSD])[select_rows]

procedure_feature_mat = pd.DataFrame(df.groupby(['person_id', 'order_nm_lower_case']).size(), 
                                     columns=['Procedure']).applymap(lambda x: 1 if x > 0 else 0).unstack(fill_value=0)
procedure_feature_mat.columns = ['_'.join(col_name) for col_name in procedure_feature_mat.columns]

print '\n procedure feature:'
print procedure_feature_mat.shape
display(procedure_feature_mat.head(10))

pickle.dump(procedure_feature_mat, open('procedure_feature_mat.p', 'wb'))


# In[ ]:



