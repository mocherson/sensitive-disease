
# coding: utf-8

# In[3]:


patient309 = pd.read_csv('../data/sensitive_disease/309_Patient.csv')
med309 = pd.read_csv('../data/sensitive_disease/309_Med.csv')
lab309 = pd.read_csv('../data/sensitive_disease/309_Lab.csv')
procedure309 = pd.read_csv('../data/sensitive_disease/309_Procedure.csv')


# In[18]:


df = pd.DataFrame(lab309.groupby(['mrd_lab_id', 'lab_nm']).size(), columns=['count'])
df.to_csv('309_Lab_mrd_lab_id_vs_lab_nm.csv')

get_ipython().system(u'head 309_Lab_mrd_lab_id_vs_lab_nm.csv')


# In[19]:


df = pd.DataFrame(med309.groupby(['mrd_med_id', 'generic_nm']).size(), columns=['count'])
df.to_csv('309_Med_mrd_med_id_vs_generic_nm.csv')

get_ipython().system(u'head 309_Med_mrd_med_id_vs_generic_nm.csv')


# In[21]:


df = procedure309[procedure309['order_type_desc'] != 'LAB']

df = pd.DataFrame(df.groupby(['order_cpt_cd', 'order_nm']).size(), columns=['count'])
df.to_csv('309_Procedure_order_cpt_cd_vs_order_nm.csv')

get_ipython().system(u'head 309_Procedure_order_cpt_cd_vs_order_nm.csv')


# In[5]:


get_ipython().system(u'ls *.csv')


# In[4]:


get_ipython().system(u'ls ../data/sensitive_disease/yz/')


# In[3]:


get_ipython().system(u'mv *.csv ../data/sensitive_disease/yz/')

