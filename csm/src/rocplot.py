#from __future__ import division
from os.path import join
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cPickle as pk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score,roc_curve, auc, recall_score,precision_score,f1_score,confusion_matrix,accuracy_score
from matplotlib.backends.backend_pdf import PdfPages

def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)
    
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)
    
path='/home/data/sensitive_disease/csm/ROC'
cases=[257, 294, 296, 300, 303, 305, 307, 309, 311, 607, 608, 616, 623, 626, 628, 632, 640, 642, 644, 646, 648, 650, 652, 654, 655, 656, 658, 659, 661, 663, 664, 669, 765, 770, 774, 779]

def rocplot(case):
    r=load_obj(join(path,'test_90pct/{}_allother_roc.pkl'.format(case)))
    y_true,prob = r[0][-1][0]
    fpr,tpr,thr=roc_curve(y_true,prob)
    roc_auc=auc(fpr,tpr)
    idx_max=(tpr-fpr).argmax()
    thr_opt=thr[idx_max]
    #thr_opt=0.5
    y_pred=prob>thr_opt
    rec=recall_score(y_true,y_pred,pos_label=1)
    prec=precision_score(y_true,y_pred,pos_label=1)
    f1=f1_score(y_true,y_pred,pos_label=1)
    acc=accuracy_score(y_true,y_pred)
    conmat=confusion_matrix(y_true,y_pred,labels=[1,0])
    pd.DataFrame(conmat,index=['case','control'],columns=[ \
        'predict to case','predict to control']).to_csv(join(path,'test_90pct/conmat_{}.csv'.format(case)))
    save_obj((prec,rec,f1,acc),join(path,'test_90pct/sc_{}.pkl'.format(case)))
    
    pdf=PdfPages(join(path,'test_90pct/roc_{}.pdf'.format(case)))
    plt.plot(fpr,tpr,'r-',lw=4)
    plt.plot([0, 1], [0, 1], ':')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.title('ROC curve for {}'.format(case),fontsize=18)
    plt.legend(['ICD9 {0} (AUC={1:.3f})'.format(case,roc_auc)],fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    pdf.close()
    plt.close('all')
    
for case in cases:
    print 'plot for case',case
    rocplot(case)