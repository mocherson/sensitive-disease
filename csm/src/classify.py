
import pandas as pd
import numpy as np
import cPickle as pk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

case=311
thr=50

case_fea=pd.read_pickle('/home/data/sensitive_disease/csm/fea_mat/fea{}_{}.pkl'.format(case,thr))
ctr_fea=pd.read_pickle('/home/data/sensitive_disease/csm/fea_mat/feanonSD5pct_{}.pkl'.format(thr))

fea_mat=pd.concat([case_fea,ctr_fea]).fillna(value=0)

X=fea_mat.as_matrix()
y=np.concatenate((np.ones(case_fea.shape[0]),np.zeros(ctr_fea.shape[0])))
print sum(y==1),sum(y==0)

metr=['mean','precision','recall','f1-score','AUC']
def classify_LR(X,y):
    clf_l2_LR = LogisticRegression()
    auc=[]
    rec=[]
    prec=[]
    f1=[]
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print 'train set:',sum(y_train==1),sum(y_train==0)
        print 'test set:',sum(y_test==1),sum(y_test==0)
        clf_l2_LR.fit(X_train, y_train)
        y_pred=clf_l2_LR.predict(X_test)
        prob=clf_l2_LR.predict_proba(X_test)
        idx=clf_l2_LR.classes_.tolist().index(1)
        auc.append(roc_auc_score(y_test, prob[:,idx]))
    #    clf_l2_LR.score(X_test,y_test)
        rec.append(recall_score(y_test,y_pred,pos_label=1))
        prec.append(precision_score(y_test,y_pred,pos_label=1))
        f1.append(f1_score(y_test,y_pred,pos_label=1))

    result=[np.mean(prec),np.mean(rec),np.mean(f1),np.mean(auc)]
    return result,prec,rec,f1,auc

r=classify_LR(X,y)

sc_fun=[chi2, f_classif, mutual_info_classif]
allk=[10,20,30,40,50,100,150,200]
rst=[]
feamasks=[]
for k in allk:
    kb=SelectKBest(sc_fun[0],k=k).fit(X,y)
    msk=kb.get_support()
    feamasks.append(msk)
    X_new=X[:,~msk]
    rst.append(classify_LR(X_new,y))
   
with open('rst{}.csv'.format(case),'w') as fl:
    print>>fl, 'with all features:'
    print>>fl, r
    for i in range(len(allk)):
        print>>fl, 'exclude {} best features:,{}'.format(allk[i],fea_mat.columns[feamasks[i]].tolist())
        for j in range(len(metr)):
            print>>fl, rst[i][j]


        
        















