import pandas as pd
import numpy as np
import cPickle as pk
from collections import Counter

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pk.load(f)

disease=311

patient_file='/home/data/sensitive_disease/%s_Patient.csv' % (disease)
lab_file='/home/data/sensitive_disease/%s_Lab.csv' % (disease)
procedure_file='/home/data/sensitive_disease/%s_Procedure.csv' % (disease)
med_file='/home/data/sensitive_disease/%s_Med.csv' % (disease)
pt_data=pd.read_csv(patient_file)
lab_data=pd.read_csv(lab_file)
pr_data=pd.read_csv(procedure_file)
med_data=pd.read_csv(med_file)

pid=pt_data.loc[:,'person_id'].unique()
print "numeber of patients:",len(pid)
print "lab data size:",lab_data.shape
print "procedure data size:",pr_data.shape
print "med data size:",med_data.shape

# for lab data
print "mrd_lab_id unique number:",lab_data.mrd_lab_id.unique().size
print "lab_nm unique number:",lab_data.lab_nm.unique().size
key=zip(lab_data.mrd_lab_id,lab_data.lab_nm)
pair=Counter(key)
save_obj(pair,'/home/data/sensitvie_disease/csm/lab{}_idnm'.format(disease))
item=[(x[0][0],x[0][1],x[1]) for x in pair.items()]
df=pd.DataFrame.from_records(item,columns=['mrd_lab_id','lab_nm','counts'])
df.to_csv('/home/data/sensitive_disease/csm/lab{}_idnm.csv'.format(disease),index=False)

# for procedure_data
print "order_cpt_cd unique number:",pr_data.order_cpt_cd.unique().size
print "order_nm unique number:",pr_data.order_nm.unique().size
print "mrd_order_id unique number:",pr_data.mrd_order_id.unique().size
key=zip(pr_data.order_cpt_cd,pr_data.order_nm)
pair=Counter(key)
save_obj(pair,'/home/data/sensitvie_disease/csm/pr{}_cptnm'.format(disease))
item=[(x[0][0],x[0][1],x[1]) for x in pair.items()]
df=pd.DataFrame.from_records(item,columns=['order_cpt_cd','order_nm','counts'])
df.to_csv('/home/data/sensitive_disease/csm/pr{}_cptnm.csv'.format(disease),index=False)

# for med_data
print "mrd_med_id unique number:",med_data.mrd_med_id.unique().size
print "generic_nm unique number:",med_data.generic_nm.unique().size
#print "mrd_order_id unique number:",pr_data.mrd_order_id.unique().size
key=zip(med_data.mrd_med_id,med_data.generic_nm)
pair=Counter(key)
save_obj(pair,'/home/data/sensitvie_disease/csm/med{}_cptnm'.format(disease))
item=[(x[0][0],x[0][1],x[1]) for x in pair.items()]
df=pd.DataFrame.from_records(item,columns=['mrd_med_id','generic_nm','counts'])
df.to_csv('/home/data/sensitive_disease/csm/med{}_idnm.csv'.format(disease),index=False)