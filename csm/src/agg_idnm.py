

import pandas as pd
import numpy as np
import cPickle as pk
from collections import Counter

disease=311

patient_file='/home/data/sensitive_disease/%s_Patient.csv' % (disease)
lab_file='/home/data/sensitive_disease/%s_Lab.csv' % (disease)
procedure_file='/home/data/sensitive_disease/%s_Procedure.csv' % (disease)
med_file='/home/data/sensitive_disease/%s_Med.csv' % (disease)

pt_data=pd.read_csv(patient_file,usecols=['person_id'])
lab_data=pd.read_csv(lab_file,usecols=['person_id','mrd_lab_id','lab_nm','lab_val'])
pr_data=pd.read_csv(procedure_file,usecols=['person_id','order_cpt_cd','order_nm'])
med_data=pd.read_csv(med_file,usecols=['person_id','mrd_med_id','generic_nm'])

pid=pt_data.loc[:,'person_id'].unique()
print "numeber of patients:",len(pid)
print "lab data size:",lab_data.shape
print "procedure data size:",pr_data.shape
print "med data size:",med_data.shape

# for lab_data
idnm=pd.read_csv('/home/data/sensitive_disease/csm/lab{}_idnm.csv'.format(disease))
idagg=idnm.loc[:,['mrd_lab_id','lab_nm','counts']].groupby(['mrd_lab_id']).agg({'lab_nm':lambda s :min(s,key=len),'counts':np.sum})
idagg=idagg.reset_index()
idnm_agg=idagg.groupby(['lab_nm']).agg({'counts':np.sum,'mrd_lab_id':np.min})
idnm_agg.to_csv('/home/data/sensitive_disease/csm/lab{}_idnm_agg.csv'.format(disease))

# for pr_data
idnm=pd.read_csv('/home/data/sensitive_disease/csm/pr{}_cptnm.csv'.format(disease))
idagg=idnm.loc[:,['order_cpt_cd','order_nm','counts']].groupby(['order_cpt_cd']).agg({'order_nm':lambda s :min(s,key=len),'counts':np.sum})
idagg=idagg.reset_index()
idnm_agg=idagg.groupby(['order_nm']).agg({'counts':np.sum,'order_cpt_cd':np.min})
idnm_agg.to_csv('/home/data/sensitive_disease/csm/pr{}_cptnm_agg.csv'.format(disease))

# for med_data
idnm=pd.read_csv('/home/data/sensitive_disease/csm/med{}_idnm.csv'.format(disease))
idagg=idnm.loc[:,['mrd_med_id','generic_nm','counts']].groupby(['mrd_med_id']).agg({'generic_nm':lambda s :min(s,key=len),'counts':np.sum})
idagg=idagg.reset_index()
idnm_agg=idagg.groupby(['generic_nm']).agg({'counts':np.sum,'mrd_med_id':np.min})
idnm_agg.to_csv('/home/data/sensitive_disease/csm/med{}_idnm_agg.csv'.format(disease))


#extract frequent lab
idnm=lab_data.loc[lab_data.lab_val<999999,['mrd_lab_id','lab_nm','person_id']].groupby(['mrd_lab_id','lab_nm']).agg(lambda x: x.unique().size)
idnm=idnm.reset_index()
idagg=idnm.groupby(['mrd_lab_id']).agg({'lab_nm':lambda s :min(s,key=len),'person_id':np.sum})
idagg=idagg.reset_index()
idnm_agg=idagg.groupby(['lab_nm']).agg({'person_id':np.sum,'mrd_lab_id':np.min})
idnm_agg_005=idnm_agg.loc[idnm_agg.person_id>len(pid)*0.05,:].rename(columns={'person_id':'persons'})
idnm_agg_005.to_csv('/home/data/sensitive_disease/csm/lab{}_idnm_agg_005.csv'.format(disease))

#extract frequent procedure
idnm=pr_data[['order_cpt_cd','order_nm','person_id']].groupby(['order_cpt_cd','order_nm']).agg(lambda x: x.unique().size)
idnm=idnm.reset_index()
idagg=idnm.groupby(['order_cpt_cd']).agg({'order_nm':lambda s :min(s,key=len),'person_id':np.sum})
idagg=idagg.reset_index()
idnm_agg=idagg.groupby(['order_nm']).agg({'person_id':np.sum,'order_cpt_cd':np.min})
idnm_agg_005=idnm_agg.loc[idnm_agg.person_id>len(pid)*0.05,:]
idnm_agg_005.to_csv('/home/data/sensitive_disease/csm/pr{}_cptnm_agg_005.csv'.format(disease))


#extract frequent med
idnm=med_data[['mrd_med_id','generic_nm','person_id']].groupby(['mrd_med_id','generic_nm']).agg(lambda x: x.unique().size)
idnm=idnm.reset_index()
idagg=idnm.groupby(['mrd_med_id']).agg({'generic_nm':lambda s :min(s,key=len),'person_id':np.sum})
idagg=idagg.reset_index()
idnm_agg=idagg.groupby(['generic_nm']).agg({'person_id':np.sum,'mrd_med_id':np.min})
idnm_agg_005=idnm_agg.loc[idnm_agg.person_id>len(pid)*0.05,:]
idnm_agg_005.to_csv('/home/data/sensitive_disease/csm/med{}_idnm_agg_005.csv'.format(disease))