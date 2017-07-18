
import pandas as pd
import numpy as np
import cPickle as pk
from collections import Counter

disease=311

patient_file='/home/data/sensitive_disease/%s_Patient.csv' % (disease)
lab_file='/home/data/sensitive_disease/%s_Lab.csv' % (disease)
procedure_file='/home/data/sensitive_disease/%s_Procedure.csv' % (disease)
med_file='/home/data/sensitive_disease/%s_Med.csv' % (disease)
pt_data=pd.read_csv(patient_file)
lab_data=pd.read_csv(lab_file)
pr_data=pd.read_csv(procedure_file).query("order_type_desc!='LAB'")
med_data=pd.read_csv(med_file)

pid=pt_data.loc[:,'person_id'].unique()
print "numeber of patients:",len(pid)
print "lab data size:",lab_data.shape
print "procedure data size:",pr_data.shape
print "med data size:",med_data.shape

lab_fea=[]
[lab_fea.extend(['lab_{}_min'.format(x),'lab_{}_max'.format(x),'lab_{}_median'.format(x)]) for x in lab_data.mrd_lab_id.unique()]
pr_fea=['pr_{}'.format(x) for x in pr_data.order_cpt_cd.unique()]
med_fea=['pr_{}'.format(x) for x in med_data.mrd_med_id.unique()]
fea=lab_fea+pr_fea+med_fea
print "length of feathers:",len(fea),". lab features:",len(lab_fea),". prcedure features:",len(pr_fea),". med features:",len(med_fea)

fea_table=pd.DataFrame(index=pid,columns=fea)

for i,p in enumerate(pid[0:100]):
    print 'pid=',p,i,'/',len(pid)
    # lab feature
    data=lab_data.loc[lab_data.person_id==p,['person_id','mrd_lab_id','lab_val']]
    ulab=data.mrd_lab_id.unique()
    for x in ulab:
        v=data.loc[data.mrd_lab_id==x,'lab_val']
        fea_table.loc[p,'lab_{}_min'.format(x)]=v.min()
        fea_table.loc[p,'lab_{}_max'.format(x)]=v.max()
        fea_table.loc[p,'lab_{}_median'.format(x)]=v.median()
           
    # procedure feature   
    data=pr_data.loc[pr_data.person_id==p,['person_id','order_cpt_cd']]
    upr=Counter(data.order_cpt_cd)
    for x in upr.keys():       
        fea_table.loc[p,'pr_{}'.format(x)]=upr[x]
    
    # med feature
    data=med_data.loc[med_data.person_id==p,['person_id','mrd_med_id']]
    umed=Counter(data.mrd_med_id)
    for x in umed:
        fea_table.loc[p,'med_{}'.format(x)]=umed[x]
        
        
fea_table.to_csv('/home/data/sensitive_disease/csm/fea_{}_sample_100.csv'.format(disease))