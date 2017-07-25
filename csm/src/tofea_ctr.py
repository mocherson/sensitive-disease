

import pandas as pd
import numpy as np
import cPickle as pk
from collections import Counter
import re
from random import sample

def stdstr(s):
    r=s.lower().replace(' - ',' ').replace(',',' ').strip()    #  ignore case, ignore plural
    r=re.sub('\s+', ' ', r)   # handling the white space
    return r

def to_num(s):
    if isinstance(s,float):
        return s
    if isinstance(s,str):
        s=s.lstrip('><=')
    try:        
        return float(s)
    except ValueError:
        return np.nan


disease='nonSD5pct'
case=311
case_thr=50

patient_file='/home/data/sensitive_disease/%s_Patient.csv' % (disease)
lab_file='/home/data/sensitive_disease/%s_Lab.csv' % (disease)
procedure_file='/home/data/sensitive_disease/%s_Procedure.csv' % (disease)
med_file='/home/data/sensitive_disease/%s_Med.csv' % (disease)

pt_data=pd.read_csv(patient_file,usecols=['person_id'])
lab_data=pd.read_csv(lab_file,usecols=['person_id','mrd_lab_id','lab_nm','lab_val','lab_ref_low_val','lab_ref_high_val','result_flag_desc'])
pr_data=pd.read_csv(procedure_file,usecols=['person_id','order_cpt_cd','order_nm','order_type_desc']).query("order_type_desc!='LAB'")
med_data=pd.read_csv(med_file,usecols=['person_id','mrd_med_id','generic_nm'])

pid=pt_data.person_id.unique()
print "numeber of patients:",len(pid)
print "lab data size:",lab_data.shape
print "procedure data size:",pr_data.shape
print "med data size:",med_data.shape

thr=50

val_not_nan=lab_data.lab_val<999999
low_not_nan=lab_data.lab_ref_low_val.notnull()
high_not_nan=lab_data.lab_ref_high_val.notnull()
lab_data=lab_data.loc[val_not_nan&(low_not_nan|high_not_nan),:]
lab_data['low']=lab_data['lab_val']<lab_data['lab_ref_low_val'].apply(to_num) 
lab_data['high']=lab_data['lab_val']>lab_data['lab_ref_high_val'].apply(to_num)

pid_lab=set(lab_data.loc[:,'person_id'].unique())
pid_pr=set(pr_data.loc[:,'person_id'].unique())
pid_med=set(med_data.loc[:,'person_id'].unique())
pid=set.union(pid_lab,pid_pr,pid_med)
#pid=sample(pid,psnum)      # randomly select a number of patient

lab_data=lab_data.loc[lab_data.person_id.isin(pid),:]
pr_data=pr_data.loc[pr_data.person_id.isin(pid),:]
med_data=med_data.loc[med_data.person_id.isin(pid),:]

### for lab_data, lab to use 
if lab_data.mrd_lab_id.isnull().any():
    id_isnan=lab_data.mrd_lab_id.isnull()
    lab_data.loc[id_isnan,['mrd_lab_id']]=['nan_'+str(x) for x in range(np.count_nonzero(id_isnan))]
    
id_group=lab_data[['mrd_lab_id','lab_nm','person_id','low','high']].groupby(['mrd_lab_id'])
id_agg=id_group.agg({'lab_nm':lambda x: min(x,key=len),'person_id':lambda x:set(x),'low':any,'high':any})
idnm_dict=id_agg.lab_nm.to_dict()
lab_data['lab_nm_alt']=lab_data['mrd_lab_id'].apply(lambda x: stdstr(idnm_dict[x]))

id_agg=id_agg.reset_index()
nm_agg=id_agg.groupby(id_agg.lab_nm.apply(stdstr)).agg({'mrd_lab_id':min,'person_id':lambda x: set.union(*x)}).reset_index()
idnm_agg_thr=nm_agg.loc[nm_agg.person_id.apply(len)>thr,:]
idnm_agg_thr.to_csv('/home/data/sensitive_disease/csm/lab{}_idnm_agg_{}.csv'.format(disease,thr))


### for pr_data, procedure to use
if pr_data.order_cpt_cd.isnull().any():
    cpt_isnan=pr_data.order_cpt_cd.isnull()
    pr_data.loc[cpt_isnan,['order_cpt_cd']]=['nan_'+str(x) for x in range(np.count_nonzero(cpt_isnan))]
    
id_group=pr_data.groupby(['order_cpt_cd'])
id_agg=id_group.agg({'order_nm':lambda x: min(x,key=len),'person_id':lambda x: set(x)})
idnm_dict=id_agg.order_nm.to_dict()
pr_data['order_nm_alt']=pr_data['order_cpt_cd'].apply(lambda x: stdstr(idnm_dict[x]))

id_agg=id_agg.reset_index()
nm_agg=id_agg.groupby(id_agg.order_nm.apply(stdstr)).agg({'order_cpt_cd':min,'person_id': lambda x: set.union(*x)}).reset_index()
idnm_agg_thr=nm_agg.loc[nm_agg.person_id.apply(len)>thr,:]
idnm_agg_thr.to_csv('/home/data/sensitive_disease/csm/pr{}_idnm_agg_{}.csv'.format(disease,thr))


### for med_data, med to use
if med_data.mrd_med_id.isnull().any():
    print "null found in med_data.mrd_med_id for disease {}".format(disease)
    id_isnan=med_data.mrd_med_id.isnull()
    med_data.loc[id_isnan,['mrd_med_id']]=['nan_'+str(x) for x in range(np.count_nonzero(id_isnan))]
id_group=med_data.groupby(['mrd_med_id'])
id_agg=id_group.agg({'generic_nm':lambda x: min(x,key=len),'person_id':lambda x: set(x)})
idnm_dict=id_agg.generic_nm.to_dict()
med_data['generic_nm_alt']=med_data['mrd_med_id'].apply(lambda x: stdstr(idnm_dict[x]))

id_agg=id_agg.reset_index()
nm_agg=id_agg.groupby(id_agg.generic_nm.apply(stdstr)).agg({'mrd_med_id':min,'person_id': lambda x: set.union(*x)}).reset_index()
idnm_agg_thr=nm_agg.loc[nm_agg.person_id.apply(len)>thr,:]
idnm_agg_thr.to_csv('/home/data/sensitive_disease/csm/med{}_idnm_agg_{}.csv'.format(disease,thr))


## generate features
# lab feature
lab_idnm_ctr=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/lab{}_idnm_agg_{}.csv'.format(disease,thr))
lab_idnm_case=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/lab{}_idnm_agg_{}.csv'.format(case,case_thr))
lab_is_use=lab_data.lab_nm_alt.isin(set(lab_idnm_ctr.lab_nm).union(set(lab_idnm_case.lab_nm)))
lab_use=lab_data.loc[lab_is_use,['person_id','lab_nm_alt','low','high']]
lab_fea=lab_use.groupby(['person_id','lab_nm_alt']).any().unstack(fill_value=0).apply(pd.to_numeric)
lab_fea.to_csv('/home/data/sensitive_disease/csm/fea_mat/lab{}_fea_{}_{}.csv'.format(disease,case,thr))
lab_fea.to_pickle('/home/data/sensitive_disease/csm/fea_mat/lab{}_fea_{}_{}.pkl'.format(disease,case,thr))

#pr feature
pr_idnm_ctr=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/pr{}_idnm_agg_{}.csv'.format(disease,thr))
pr_idnm_case=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/pr{}_idnm_agg_{}.csv'.format(case,case_thr))
pr_is_use=pr_data.order_nm_alt.isin(set(pr_idnm_ctr.order_nm).union(set(pr_idnm_case.order_nm)))
pr_use=pr_data.loc[pr_is_use,['person_id','order_nm_alt']]
pr_fea=pr_use.groupby(['person_id','order_nm_alt']).agg(lambda x: 1).unstack(fill_value=0)
pr_fea.to_csv('/home/data/sensitive_disease/csm/fea_mat/pr{}_fea_{}_{}.csv'.format(disease,case,thr))
pr_fea.to_pickle('/home/data/sensitive_disease/csm/fea_mat/pr{}_fea_{}_{}.pkl'.format(disease,case,thr))

#med feature
med_idnm_ctr=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/med{}_idnm_agg_{}.csv'.format(disease,thr))
med_idnm_case=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/med{}_idnm_agg_{}.csv'.format(case,case_thr))
med_is_use=med_data.generic_nm_alt.isin(set(med_idnm_ctr.generic_nm).union(set(med_idnm_case.generic_nm)))
med_use=med_data.loc[med_is_use,['person_id','generic_nm_alt']]
med_fea=med_use.groupby(['person_id','generic_nm_alt']).agg(lambda x: 1).unstack(fill_value=0)
med_fea.to_csv('/home/data/sensitive_disease/csm/fea_mat/med{}_fea_{}_{}.csv'.format(disease,case,thr))
med_fea.to_pickle('/home/data/sensitive_disease/csm/fea_mat/med{}_fea_{}_{}.pkl'.format(disease,case,thr))


#lab_fea=pd.read_pickle('/home/data/sensitive_disease/csm/fea_mat/lab{}_fea_{}_{}.pkl'.format(disease,case,thr))
#pr_fea=pd.read_pickle('/home/data/sensitive_disease/csm/fea_mat/pr{}_fea_{}_{}.pkl'.format(disease,case,thr))
#med_fea=pd.read_pickle('/home/data/sensitive_disease/csm/fea_mat/med{}_fea_{}_{}.pkl'.format(disease,case,thr))

lab_fea.columns = ['_'.join(col).strip() for col in lab_fea.columns.values]
lab_fea.rename(columns=lambda x: 'Lab_'+str(x),inplace=True)
pr_fea.rename(columns=lambda x: 'Procedure_'+str(x),inplace=True)
med_fea.rename(columns=lambda x: 'Med_'+str(x),inplace=True)

fea = pd.concat([lab_fea, pr_fea, med_fea], axis=1).fillna(value=0)
fea.to_csv('/home/data/sensitive_disease/csm/fea_mat/fea{}_{}_{}.csv'.format(disease,case,thr))
fea.to_pickle('/home/data/sensitive_disease/csm/fea_mat/fea{}_{}_{}.pkl'.format(disease,case,thr))




