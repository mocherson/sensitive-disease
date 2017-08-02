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
from functools import partial

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
    
def LR_classify(X,y,metric='auc'):
    clf = LogisticRegression()
    m=[]
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.8, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        prob=clf.predict_proba(X_test)
        idx=clf.classes_.tolist().index(1)
        m.append(roc_auc_score(y_test, prob[:,idx]))        
    return np.mean(m)

def LR_performance(X,y,isincrs=True):
    perf=np.zeros(X.shape[1])
    if isincrs:
        for i in range(X.shape[1]):
            perf[i]=LR_classify(X[:,[i]],y)
    else:
        for i in range(X.shape[1]):
            perf[i]=1 - LR_classify(np.delete(X,i,1),y)
    return perf

class Sens_disease:
    def __init__(self,disease,ctr='nonSD5pct',ctr_thr=50,thr=50,savepath='/home/data/sensitive_disease/csm/'):
        self.disease=disease
        self.ctr=ctr
        self.thr=thr
        self.ctr_thr=ctr_thr
        self.path=savepath
        
    def load(self,onlyuse=False):
        disease=self.disease
        if onlyuse:
            self.lab_data=pd.DataFrame.from_csv(join(self.path,'usedata/{}_lab_data.csv'.format(self.disease)))
            self.pr_data=pd.DataFrame.from_csv(join(self.path,'usedata/{}_pr_data.csv'.format(self.disease)))
            self.med_data=pd.DataFrame.from_csv(join(self.path,'usedata/{}_med_data.csv'.format(self.disease)))
            return
        
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
        if psnum != 'all' and psnum>0 and psnum <len(self.pid):
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
        
    def generate_id_mapping(self):
        self.lab_data.groupby(['mrd_lab_id'])['lab_nm'].agg(lambda x: min(x,key=len)).to_csv( \
                    join(self.path,'mapping/{}lab_id_mapping.csv'.format(self.disease)))
        self.pr_data.groupby(['order_cpt_cd'])['order_nm'].agg(lambda x: min(x,key=len)).to_csv( \
                    join(self.path,'mapping/{}procedure_id_mapping.csv'.format(self.disease)))
        self.med_data.groupby(['mrd_med_id'])['generic_nm'].agg(lambda x: min(x,key=len)).to_csv( \
                    join(self.path,'mapping/{}med_id_mapping.csv'.format(self.disease)))
        
    def add_alt_nm(self,isctr=False):
        case_lab_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}lab_id_mapping.csv'.format(self.disease)))
        ctr_lab_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}lab_id_mapping.csv'.format(self.ctr)))
        case_pr_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}procedure_id_mapping.csv'.format(self.disease)))
        ctr_pr_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}procedure_id_mapping.csv'.format(self.ctr)))
        case_med_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}med_id_mapping.csv'.format(self.disease)))
        ctr_med_id_mapping=pd.Series.from_csv(join(self.path,'mapping/{}med_id_mapping.csv'.format(self.ctr)))
        if isctr:
            self.lab_id_map=case_lab_id_mapping.to_dict()
            self.pr_id_map=case_pr_id_mapping.to_dict()
            self.med_id_map=case_med_id_mapping.to_dict()

        else:
            self.lab_id_map=pd.concat([case_lab_id_mapping,ctr_lab_id_mapping]).to_dict()
            self.pr_id_map=pd.concat([case_pr_id_mapping,ctr_pr_id_mapping]).to_dict()
            self.med_id_map=pd.concat([case_med_id_mapping,ctr_med_id_mapping]).to_dict()

        self.lab_nm_map=pd.Series.from_csv(join(self.path,'mapping/lab_name_mapping.csv')).to_dict()
        self.pr_nm_map=pd.Series.from_csv('procedure_name_mapping.csv').to_dict()
        self.med_nm_map=pd.Series.from_csv('med_name_mapping.csv').to_dict()
        
        self.lab_data.loc[:,'lab_nm_alt']=self.lab_data['mrd_lab_id'].apply(  \
                    lambda x: self.normalize_lab_nm(self.lab_id_map[x]))
        self.pr_data.loc[:,'order_nm_alt']=self.pr_data['order_cpt_cd'].apply(  \
                    lambda x: self.normalize_pr_nm(self.pr_id_map[x]))
        self.med_data.loc[:,'generic_nm_alt']=self.med_data['mrd_med_id'].apply(  \
                    lambda x:self.normalize_med_nm(self.med_id_map[x]))  
        
        self.lab_data[['person_id','lab_nm_alt','low','high']].to_csv(
                    join(self.path,'usedata/{}_lab_data.csv'.format(self.disease)))
        self.pr_data[['person_id','order_nm_alt']].to_csv(join(self.path,'usedata/{}_pr_data.csv'.format(self.disease)))
        self.med_data[['person_id','generic_nm_alt']].to_csv(join(self.path,'usedata/{}_med_data.csv'.format(self.disease)))

    def generate_nm(self):        
        self.lab_ups=self.lab_data.groupby('lab_nm_alt')['person_id'].nunique()
        self.lab_nm_set=self.lab_ups.index[self.lab_ups>self.thr]
        self.lab_ups[self.lab_ups>self.thr].to_csv(join(self.path,'nmset/lab{}_nmset_{}.csv'.format(self.disease,self.thr)))
        
        self.pr_ups=self.pr_data.groupby('order_nm_alt')['person_id'].nunique()
        self.pr_nm_set=self.pr_ups.index[self.pr_ups>self.thr]
        self.pr_ups[self.pr_ups>self.thr].to_csv( join(self.path,'nmset/pr{}_nmset_{}.csv'.format(self.disease,self.thr)))
        
        self.med_ups=self.med_data.groupby('generic_nm_alt')['person_id'].nunique()
        self.med_nm_set=self.med_ups.index[self.med_ups>self.thr]
        self.med_ups[self.med_ups>self.thr].to_csv( join(self.path,'nmset/med{}_nmset_{}.csv'.format(self.disease,self.thr)))
        
    def get_nm_set(self, iforg=False):
        suff='_org' if iforg else ''
        self.lab_nm_set=pd.Series.from_csv(join(self.path,'nmset/lab{}_nmset_{}{}.csv'.format(self.disease,self.thr,suff))).index
        self.pr_nm_set=pd.Series.from_csv(join(self.path,'nmset/pr{}_nmset_{}{}.csv'.format(self.disease,self.thr,suff))).index
        self.med_nm_set=pd.Series.from_csv(join(self.path,'nmset/med{}_nmset_{}{}.csv'.format(self.disease,self.thr,suff))).index
        
    def generate_feamat(self):
        # lab feature
        lab_idnm_ctr=pd.Series.from_csv(join(self.path,'nmset/lab{}_nmset_{}.csv'.format(self.ctr,self.ctr_thr)))
        lab_is_use=self.lab_data.lab_nm_alt.isin(self.lab_nm_set.union(lab_idnm_ctr.index))
        lab_use=self.lab_data.loc[lab_is_use,['person_id','lab_nm_alt','low','high']]
        self.lab_fea=lab_use.groupby(['person_id','lab_nm_alt']).any().unstack(fill_value=0).apply(pd.to_numeric)
        self.lab_fea.to_csv(join(self.path,'fea_mat/lab{}_fea_{}_{}.csv'.format(self.disease,self.ctr,self.thr)))
        self.lab_fea.to_pickle(join(self.path,'fea_mat/lab{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))

        #pr feature
        pr_idnm_ctr=pd.Series.from_csv(join(self.path,'nmset/pr{}_nmset_{}.csv'.format(self.ctr,self.ctr_thr)))
        pr_is_use=self.pr_data.order_nm_alt.isin(self.pr_nm_set.union(pr_idnm_ctr.index))
        pr_use=self.pr_data.loc[pr_is_use,['person_id','order_nm_alt']]
        self.pr_fea=pr_use.groupby(['person_id','order_nm_alt']).agg(lambda x: 1).unstack(fill_value=0)
        self.pr_fea.to_csv(join(self.path,'fea_mat/pr{}_fea_{}_{}.csv'.format(self.disease,self.ctr,self.thr)))
        self.pr_fea.to_pickle(join(self.path,'fea_mat/pr{}_fea_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))

        #med feature
        med_idnm_ctr=pd.Series.from_csv(join(self.path,'nmset/med{}_nmset_{}.csv'.format(self.ctr,self.ctr_thr)))
        med_is_use=self.med_data.generic_nm_alt.isin(self.med_nm_set.union(med_idnm_ctr.index))
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
        
    def part_feamat(self,feaset='union',iforg=False):
        self.total_feamat(iforg=iforg)
        self.get_nm_set(iforg=iforg)
        suff='_org' if iforg else ''
        ctr_labnm_set=pd.Series.from_csv(join(self.path,'nmset/lab{}_nmset_{}{}.csv'.format(self.ctr,self.ctr_thr,suff))).index
        ctr_prnm_set=pd.Series.from_csv(join(self.path,'nmset/pr{}_nmset_{}{}.csv'.format(self.ctr,self.ctr_thr,suff))).index
        ctr_mednm_set=pd.Series.from_csv(join(self.path,'nmset/med{}_nmset_{}{}.csv'.format(self.ctr,self.ctr_thr,suff))).index
        curfeaset=('Lab_high_'+self.lab_nm_set)|('Lab_low_'+self.lab_nm_set)|('Procedure_'+self.pr_nm_set)|('Med_'+self.med_nm_set)
        ctrfeaset=('Lab_high_'+ctr_labnm_set)|('Lab_low_'+ctr_labnm_set)|('Procedure_'+ctr_prnm_set)|('Med_'+ctr_mednm_set)
        if feaset=='control':
            feaset=ctrfeaset
        elif feaset=='union':
            feaset=curfeaset | ctrfeaset
        elif feaset=='diff':
            feaset=ctrfeaset.difference (curfeaset)
        elif feaset=='intersect':
            feaset=ctrfeaset & curfeaset
        elif feaset=='case':
            feaset=curfeaset
            
        self.fea_mat=self.fea_mat[feaset]
        self.X=self.fea_mat.as_matrix()

    def total_feamat(self,iforg=False):
        if iforg:
            ctr_fea=pd.DataFrame.from_csv(join(self.path,'fea_mat/fea{}_{}_org.csv'.format(self.ctr,self.ctr_thr)))
            self.curfea=pd.DataFrame.from_csv(join(self.path,'fea_mat/fea{}_{}_org.csv'.format(self.disease,self.thr)))
        else:
            ctr_fea=pd.read_pickle(join(self.path,'fea_mat/fea{}_{}_{}.pkl'.format(self.ctr,self.disease,self.ctr_thr)))
            self.curfea=pd.read_pickle(join(self.path,'fea_mat/fea{}_{}_{}.pkl'.format(self.disease,self.ctr,self.thr)))
        self.fea_mat=pd.concat([self.curfea,ctr_fea]).fillna(value=0)
        self.fea_mat.to_csv(join(self.path,'fea_mat/fea{}_{}_{}_total.csv'.format(self.disease,self.ctr,self.thr)))
        self.X=self.fea_mat.as_matrix()
        self.y=np.concatenate((np.ones(self.curfea.shape[0]),np.zeros(ctr_fea.shape[0])))
        
    def classify_LR(self,allk=[0,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000],usebest=False,svtype='union_unify'):
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
            save_obj(self.r,join(self.path,'results/{}_{}_{}'.format(self.disease,self.ctr,svtype)))
        
    def get_fea_sc(self,score_func=chi2,svtype='union'):
        score_func_ret=score_func(self.X,self.y)
        sc=score_func_ret[0] if isinstance(score_func_ret, (list, tuple)) else score_func_ret
        sc[np.isnan(sc)] = 0
        self.idx=np.argsort(sc)[::-1]
        save_obj(sc[self.idx],join(self.path,'fearank/{}fea_rank_{}'.format(self.disease,svtype)))
        save_obj(self.fea_mat.columns[self.idx],join(self.path,'fearank/{}features_{}'.format(self.disease,svtype)))

    def org_nm_set(self):
        self.lab_ups=self.lab_data.groupby('lab_nm')['person_id'].nunique()
        self.lab_nm_set=self.lab_ups.index[self.lab_ups>self.thr]
        self.lab_ups[self.lab_ups>self.thr].to_csv(join(self.path,'nmset/lab{}_nmset_{}_org.csv'.format(self.disease,self.thr)))
        
        self.pr_ups=self.pr_data.groupby('order_nm')['person_id'].nunique()
        self.pr_nm_set=self.pr_ups.index[self.pr_ups>self.thr]
        self.pr_ups[self.pr_ups>self.thr].to_csv(join(self.path,'nmset/pr{}_nmset_{}_org.csv'.format(self.disease,self.thr)))
        
        self.med_ups=self.med_data.groupby('generic_nm')['person_id'].nunique()
        self.med_nm_set=self.med_ups.index[self.med_ups>self.thr]
        self.med_ups[self.med_ups>self.thr].to_csv(join(self.path,'nmset/med{}_nmset_{}_org.csv'.format(self.disease,self.thr)))

    def entire_feamat(self,iforg=False):
        suff,labnm,prnm,mednm=('_org','lab_nm','order_nm','generic_nm') if iforg else \
                ('','lab_nm_alt','order_nm_alt','generic_nm_alt')
        self.lab_fea=self.lab_data.groupby(['person_id',labnm])['high','low'].any().unstack(fill_value=0).apply(pd.to_numeric)
        self.pr_fea=self.pr_data.groupby(['person_id',prnm]).apply(lambda x: 1).unstack(fill_value=0)
        self.med_fea=self.med_data.groupby(['person_id',mednm]).apply(lambda x: 1).unstack(fill_value=0)
        
        self.lab_fea.columns = ['Lab_'+'_'.join(col).strip() for col in self.lab_fea.columns.values]
        self.pr_fea.columns='Procedure_'+self.pr_fea.columns.values
        self.med_fea.columns='Med_'+self.med_fea.columns.values

        self.curfea = pd.concat([self.lab_fea, self.pr_fea, self.med_fea], axis=1).fillna(value=0)
        self.curfea.to_csv(join(self.path,'fea_mat/fea{}_{}{}.csv'.format(self.disease,self.thr,suff)))
#        self.curfea.to_pickle(join(self.path,'fea_mat/fea{}_{}_org.pkl'.format(self.disease,self.thr)))
        
    def allrun(self,ps=5000,s=0,isctr=False,feaset='union',iforg=False,score_func=chi2):
        self.load()
        self.get_persons(psnum=ps,rs=s)
        if iforg:
            svtype=str(feaset)+'_org'
            self.org_nm_set()
            self.entire_feamat(iforg=iforg)
        else:
            svtype=str(feaset)+'_unify'
            self.generate_nm(isctr=isctr)
            self.generate_feamat()
        
        self.part_feamat(feaset=feaset,iforg=iforg)
        self.get_fea_sc(score_func=score_func)
        self.classify_LR(svtype=svtype)
        
    def midrun(self,feaset='union',iforg=False,score_func=chi2):
        svtype=str(feaset)+'_org' if iforg else str(feaset)+'_unify'
        self.part_feamat(feaset=feaset,iforg=iforg)
        self.get_fea_sc(score_func=score_func,svtype=feaset)
        self.classify_LR(svtype=svtype)

        
def firstrun(case,ctr='nonSD5pct'):
    sd_case=Sens_disease(case,ctr=ctr)
    sd_ctr=Sens_disease(ctr,ctr=case)
    sd_case.load()
    sd_ctr.load(onlyuse=True)
    sd_case.get_persons()
    sd_case.generate_id_mapping()
    sd_case.add_alt_nm()
    sd_case.generate_nm()
    sd_case.generate_feamat()
    sd_ctr.get_nm_set()
    sd_ctr.generate_feamat()
    
    sd_case.total_feamat()    
    sd_case.get_fea_sc()
    sd_case.classify_LR()
    
    sd_case.part_feamat(feaset='control')
    sd_case.get_fea_sc(svtype='control')
    sd_case.classify_LR(svtype='control_unify')
        
        
        
        
