import pandas as pd
import numpy as np
import cPickle as pk
from collections import Counter,OrderedDict
import re
from random import sample, seed
from os.path import join
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

def to_num(s):
    if isinstance(s,float):
        return s
    if isinstance(s,str):
        s=s.lstrip('><=')
    try:        
        return float(s)
    except ValueError:
        return np.nan
    
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pk.load(f)
    
def get_nm_set(sd):
    sd.lab_nm_set=set(pd.Series.from_csv(join(sd.path,'lab{}_nmset_{}.csv'.format(sd.disease,sd.thr))).index)
    sd.pr_nm_set=set(pd.Series.from_csv(join(sd.path,'pr{}_nmset_{}.csv'.format(sd.disease,sd.thr))).index)
    sd.med_nm_set=set(pd.Series.from_csv(join(sd.path,'med{}_nmset_{}.csv'.format(sd.disease,sd.thr))).index)

class Sens_disease:
    def __init__(self,disease,ctr='nonSD5pct',ctr_thr=50,thr=50,ifload=True,savepath='/home/data/sensitive_disease/csm/'):
        self.disease=disease
        self.ctr=ctr
        self.thr=thr
        self.ctr_thr=ctr_thr
        self.path=savepath
        if ifload:
            self.load()
        
    def load(self):
        disease=self.disease
        patient_file='/home/data/sensitive_disease/%s_Patient.csv' % (disease)
        lab_file='/home/data/sensitive_disease/%s_Lab.csv' % (disease)
        procedure_file='/home/data/sensitive_disease/%s_Procedure.csv' % (disease)
        med_file='/home/data/sensitive_disease/%s_Med.csv' % (disease)

        self.pt_data_org=pd.read_csv(patient_file)
        self.lab_data_org=pd.read_csv(lab_file)
        self.pr_data_org=pd.read_csv(procedure_file)
        self.med_data_org=pd.read_csv(med_file)
        
        pid=self.pt_data_org.person_id.unique()
        print "numeber of original patients:",len(pid)
        print "original lab data size:",self.lab_data_org.shape
        print "original procedure data size:",self.pr_data_org.shape
        print "original med data size:",self.med_data_org.shape
        
        self.lab_data=self.lab_data_org.dropna(subset=['person_id','mrd_lab_id', 'lab_nm','lab_val'])[ \
                ['person_id','mrd_lab_id', 'lab_nm','lab_val', 'lab_ref_low_val','lab_ref_high_val']]
        self.lab_data.loc[:,'lab_ref_low_val']=self.lab_data.lab_ref_low_val.apply(to_num)
        self.lab_data.loc[:,'lab_ref_high_val']=self.lab_data.lab_ref_high_val.apply(to_num)
        val_not_nan=self.lab_data.lab_val<999999
        low_not_nan=self.lab_data.lab_ref_low_val.notnull()
        high_not_nan=self.lab_data.lab_ref_high_val.notnull()
        self.lab_data=self.lab_data.loc[val_not_nan&(low_not_nan|high_not_nan)]
        self.lab_data.loc[:,'low']=self.lab_data['lab_val']<self.lab_data['lab_ref_low_val']
        self.lab_data.loc[:,'high']=self.lab_data['lab_val']>self.lab_data['lab_ref_high_val']

        self.pr_data=self.pr_data_org.query("order_type_desc not in ['LAB','VASCULAR LAB']")[[  \
                        'person_id', 'order_cpt_cd','order_nm']].dropna()
        self.pr_data=self.pr_data.loc[self.pr_data.order_nm.apply(    \
                        lambda x: not ('report' in x.lower() or x.lower().endswith('note') ))]
        self.med_data=self.med_data_org.loc[:,['person_id','mrd_med_id','generic_nm']].dropna()
        
    def get_persons(self,psnum=5000,rs=0):
        pid_lab=set(self.lab_data.person_id)
        pid_pr=set(self.pr_data.person_id)
        pid_med=set(self.med_data.person_id)
        self.pid=set.union(pid_lab,pid_pr,pid_med)
        if psnum != 'all' and psnum>0:
            seed(rs)
            self.pid=sample(self.pid,psnum)      # randomly select a number of patient

        self.lab_data=self.lab_data.query('person_id in @self.pid')
        self.pr_data=self.pr_data.query('person_id in @self.pid')
        self.med_data=self.med_data.query('person_id in @self.pid')

    
    def normalize_lab_nm(self,name):
        name=re.sub('\s+', ' ', name)
        name = name.lower().replace(', ', ' ').replace(' - ', '-').strip().rstrip(',.#% ')
        norm_name=re.sub(' level$| lvl$|-nwanal$|^abs |^absolute |\(manual diff\)', '', name).rstrip(',.#% ').strip()
        if norm_name in self.lab_nm_map:
            return self.lab_nm_map[norm_name]
        else:
            return norm_name
        
    def normalize_pr_nm(self,name):
        name=re.sub('\s+', ' ', name)
        name = name.lower().replace(', ', ' ').replace(' - ', '-').strip()
        norm_name=re.sub('-nmff$|-nmh$| nmff card$| nlfh$| note$', '', name).strip()

        if norm_name in self.pr_nm_map:
            return self.pr_nm_map[norm_name]
        else:
            return norm_name
        
    def normalize_med_nm(self,name):
        name=re.sub('\s+', ' ', name)
        name = name.lower().replace(', ', ' ').replace(' - ', '-').replace('/', '-').strip()
        name=re.sub(' topical$| ophthalmic$|\(product\)$|\(substance\)$| product$|\[chemical-ingredient\]$', '', name).strip()

        norm_name = '-'.join(OrderedDict.fromkeys(name.split('-')))

        if norm_name in self.med_nm_map:
            return self.med_nm_map[norm_name]
        else:
            return norm_name

    def generate_nm(self,ps=5000,seed=0,isctr=False):
        disease=self.disease
        self.get_persons(psnum=ps,rs=seed)
        
        case_lab_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}lab_id_mapping.csv'.format(self.disease)))
        ctr_lab_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}lab_id_mapping.csv'.format(self.ctr)))
        case_pr_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}procedure_id_mapping.csv'.format(self.disease)))
        ctr_pr_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}procedure_id_mapping.csv'.format(self.ctr)))
        case_med_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}med_id_mapping.csv'.format(self.disease)))
        ctr_med_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}med_id_mapping.csv'.format(self.ctr)))
        if isctr:
            self.lab_id_map=pd.concat([ctr_lab_id_mapping,case_lab_id_mapping]).to_dict()
            self.pr_id_map=pd.concat([ctr_pr_id_mapping,case_pr_id_mapping]).to_dict()
            self.med_id_map=pd.concat([ctr_med_id_mapping,case_med_id_mapping]).to_dict()

        else:
            self.lab_id_map=pd.concat([case_lab_id_mapping,ctr_lab_id_mapping]).to_dict()
            self.pr_id_map=pd.concat([case_pr_id_mapping,ctr_pr_id_mapping]).to_dict()
            self.med_id_map=pd.concat([case_med_id_mapping,ctr_med_id_mapping]).to_dict()

        self.lab_nm_map=pd.Series.from_csv(join(self.path,'mapping/lab_name_mapping.csv')).to_dict()
        self.pr_nm_map=pd.Series.from_csv('procedure_name_mapping.csv').to_dict()
        self.med_nm_map=pd.Series.from_csv('med_name_mapping.csv').to_dict()
        
        ### for lab_data, lab to use 
#        id_agg=self.lab_data[['mrd_lab_id','lab_nm',]].groupby(['mrd_lab_id']).agg({'lab_nm':lambda x: min(x,key=len)})
#        self.lab_idnm_mapping=id_agg.lab_nm.to_dict()
        self.lab_data.loc[:,'lab_nm_alt']=self.lab_data['mrd_lab_id'].apply(  \
                    lambda x: self.normalize_lab_nm(self.lab_id_map[x]))
        self.lab_ups=self.lab_data.groupby('lab_nm_alt')['person_id'].nunique()
        self.lab_nm_set=set(self.lab_ups.index[self.lab_ups>self.thr])
        self.lab_ups[self.lab_ups>self.thr].to_csv(join(self.path,'lab{}_nmset_{}.csv'.format(self.disease,self.thr)))

        ### for pr_data, procedure to use
#        if self.pr_data.order_cpt_cd.isnull().any():
#            cpt_isnan=self.pr_data.order_cpt_cd.isnull()
#            self.pr_data.loc[cpt_isnan,['order_cpt_cd']]=['nan_'+str(x) for x in range(np.count_nonzero(cpt_isnan))]

#        id_agg=self.pr_data.groupby(['order_cpt_cd']).agg({'order_nm':lambda x: min(x,key=len)})
#        idnm_dict=id_agg.order_nm.to_dict()
        self.pr_data.loc[:,'order_nm_alt']=self.pr_data['order_cpt_cd'].apply(  \
                    lambda x: self.normalize_pr_nm(self.pr_id_map[x]))
        self.pr_ups=self.pr_data.groupby('order_nm_alt')['person_id'].nunique()
        self.pr_nm_set=set(self.pr_ups.index[self.pr_ups>self.thr])
        self.pr_ups[self.pr_ups>self.thr].to_csv( join(self.path,'pr{}_nmset_{}.csv'.format(self.disease,self.thr)))

        ### for med_data, med to use
#        if self.med_data.mrd_med_id.isnull().any():
#            print "null found in med_data.mrd_med_id for disease {}".format(disease)
#            id_isnan=self.med_data.mrd_med_id.isnull()
#            self.med_data.loc[id_isnan,['mrd_med_id']]=[self.med_data['mrd_med_id'].max()+1+x for x in \
#                                                        range(np.count_nonzero(id_isnan))]
            
#        id_agg=self.med_data.groupby(['mrd_med_id']).agg({'generic_nm':lambda x: min(x,key=len)})
#        idnm_dict=id_agg.generic_nm.to_dict()
        self.med_data.loc[:,'generic_nm_alt']=self.med_data['mrd_med_id'].apply(  \
                lambda x:self.normalize_med_nm(self.med_id_map[x]))
        self.med_ups=self.med_data.groupby('generic_nm_alt')['person_id'].nunique()
        self.med_nm_set=set(self.med_ups.index[self.med_ups>self.thr])
        self.med_ups[self.med_ups>self.thr].to_csv( join(self.path,'med{}_nmset_{}.csv'.format(self.disease,self.thr)))
        
    def get_nm_set(self):
        self.lab_nm_set=set(pd.Series.from_csv(join(self.path,'lab{}_nmset_{}.csv'.format(self.disease,self.thr))).index)
        self.pr_nm_set=set(pd.Series.from_csv(join(self.path,'pr{}_nmset_{}.csv'.format(self.disease,self.thr))).index)
        self.med_nm_set=set(pd.Series.from_csv(join(self.path,'med{}_nmset_{}.csv'.format(self.disease,self.thr))).index)
        
    def generate_feamat(self):
        # lab feature
        lab_idnm_ctr=pd.Series.from_csv(join(self.path,'lab{}_nmset_{}.csv'.format(self.ctr,self.ctr_thr)))
        lab_is_use=self.lab_data.lab_nm_alt.isin(self.lab_nm_set.union(set(lab_idnm_ctr.index)))
        lab_use=self.lab_data.loc[lab_is_use,['person_id','lab_nm_alt','low','high']]
        self.lab_fea=lab_use.groupby(['person_id','lab_nm_alt']).any().unstack(fill_value=0).apply(pd.to_numeric)
        self.lab_fea.to_csv(join(self.path,'fea_mat/lab{}_fea_{}_{}.csv'.format(self.disease,self.ctr,self.thr)))
        self.lab_fea.to_pickle(join(self.path,'fea_mat/lab{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))

        #pr feature
        pr_idnm_ctr=pd.Series.from_csv(join(self.path,'pr{}_nmset_{}.csv'.format(self.ctr,self.ctr_thr)))
        pr_is_use=self.pr_data.order_nm_alt.isin(self.pr_nm_set.union(set(pr_idnm_ctr.index)))
        pr_use=self.pr_data.loc[pr_is_use,['person_id','order_nm_alt']]
        self.pr_fea=pr_use.groupby(['person_id','order_nm_alt']).agg(lambda x: 1).unstack(fill_value=0)
        self.pr_fea.to_csv(join(self.path,'fea_mat/pr{}_fea_{}_{}.csv'.format(self.disease,self.ctr,self.thr)))
        self.pr_fea.to_pickle(join(self.path,'fea_mat/pr{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))

        #med feature
        med_idnm_ctr=pd.Series.from_csv(join(self.path,'med{}_nmset_{}.csv'.format(self.ctr,self.ctr_thr)))
        med_is_use=self.med_data.generic_nm_alt.isin(self.med_nm_set.union(set(med_idnm_ctr.index)))
        med_use=self.med_data.loc[med_is_use,['person_id','generic_nm_alt']]
        self.med_fea=med_use.groupby(['person_id','generic_nm_alt']).agg(lambda x: 1).unstack(fill_value=0)
        self.med_fea.to_csv(join(self.path,'fea_mat/med{}_fea_{}_{}.csv'.format(self.disease,self.ctr,self.thr)))
        self.med_fea.to_pickle(join(self.path,'fea_mat/med{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))

        self.lab_fea.columns = ['Lab_'+'_'.join(col).strip() for col in self.lab_fea.columns.values]
        self.pr_fea.columns='Procedure_'+self.pr_fea.columns.values
        self.med_fea.columns='Med_'+self.med_fea.columns.values

        self.curfea = pd.concat([self.lab_fea, self.pr_fea, self.med_fea], axis=1).fillna(value=0)
        self.curfea.to_csv(join(self.path,'fea_mat/fea{}_{}_{}.csv'.format(self.disease,self.ctr,self.thr)))
        self.curfea.to_pickle(join(self.path,'fea_mat/fea{}_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))
        
    def get_feamat(self):
        self.lab_fea=pd.read_pickle(join(self.path,'fea_mat/lab{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))
        self.pr_fea=pd.read_pickle(join(self.path,'fea_mat/pr{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))
        self.med_fea=pd.read_pickle(join(self.path,'fea_mat/med{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))
        self.curfea=pd.read_pickle(join(self.path,'fea_mat/fea{}_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))

    def mergectr(self):
        ctr_fea=pd.read_pickle(join(self.path,'fea_mat/fea{}_{}_{}.pkl'.format(self.ctr,self.disease,self.ctr_thr)))
        self.curfea=pd.read_pickle(join(self.path,'fea_mat/fea{}_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))
        self.fea_mat=pd.concat([self.curfea,ctr_fea]).fillna(value=0)
        self.X=self.fea_mat.as_matrix()
        self.y=np.concatenate((np.ones(self.curfea.shape[0]),np.zeros(ctr_fea.shape[0])))
        
    def classify_LR(self,allk=[0,10],usebest=False):
        self.r=[]
        for k in allk:
            if usebest:
                X=self.X[:,self.idx[:k]]
            else:
                X=self.X[:,self.idx[k:]]
            clf_l2_LR = LogisticRegression()
            auc=[]
            rec=[]
            prec=[]
            f1=[]
            sss = StratifiedShuffleSplit(n_splits=10, test_size=0.8, random_state=0)
            for train_index, test_index in sss.split(X, self.y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                print 'train set:',sum(y_train==1),sum(y_train==0)
                print 'test set:',sum(y_test==1),sum(y_test==0)
                clf_l2_LR.fit(X_train, y_train)
                y_pred=clf_l2_LR.predict(X_test)
                prob=clf_l2_LR.predict_proba(X_test)
                idx=clf_l2_LR.classes_.tolist().index(1)
                auc.append(roc_auc_score(y_test, prob[:,idx]))
                rec.append(recall_score(y_test,y_pred,pos_label=1))
                prec.append(precision_score(y_test,y_pred,pos_label=1))
                f1.append(f1_score(y_test,y_pred,pos_label=1))

            result=[np.mean(prec),np.mean(rec),np.mean(f1),np.mean(auc)]
            self.r.append((result,prec,rec,f1,auc))
            save_obj(self.r,'{}_{}result'.format(self.disease,self.ctr))
        
    def get_fea_sc(self,score_func=chi2):
        sc,pv=score_func(self.X,self.y)
        sc[np.isnan(sc)] = 0
        self.idx=np.argsort(sc)[::-1]
        self.sc=sc[self.idx]
        
    
        
        
        
        
        
        
