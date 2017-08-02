from __future__ import division
from os.path import join
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter



def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)
    
path='/home/data/sensitive_disease/csm/fearank'

allk=[0,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
index=['All']+['Excl '+str(x)+' best' for x in allk[1:]]
columns=['Precision','Recall','F-measure','AUC']
cases=[311,632,654,644,658,616,303,655,623,661,765,608,646,652]


def plot_feasc(case,sc_func='chi2'):
    pdf=PdfPages(join(path,sc_func,'fea_score_{}.pdf'.format(case)))
    sc_union=load_obj(join(path,sc_func,str(case)+'fea_rank_union.pkl'))
    sc_control=load_obj(join(path,sc_func,str(case)+'fea_rank_control.pkl'))

    plt.figure(1)
    plt.plot(sc_union,'r-')
    plt.plot(sc_control,'b--')
    plt.legend(['both features','control features'])
    plt.title('ICD9 {}: Feature predictive score in descending order'.format(case))
    plt.xlabel('Feature rank')
    plt.ylabel('chi2 score')
    pdf.savefig()
    pdf.close()
    plt.close('all')


def plot_fea_bar(case,sc_func='chi2'):
    pdf=PdfPages(join(path,sc_func,'fea_bar_{}.pdf'.format(case)))
    fea_union=load_obj(join(path,sc_func,str(case)+'features_union.pkl'))
    fea_control=load_obj(join(path,sc_func,str(case)+'features_control.pkl'))

    fea_type_union=[x[:3] for x in fea_union]
    fea_type_control=[x[:3] for x in fea_control]

    union_featype=[Counter(fea_type_union[:k]) for k in allk[1:]]
    fea_ratio=[(x['Lab']*100/sum(x.values()),x['Med']*100/sum(x.values()),x['Pro']*100/sum(x.values())) for x in union_featype]
    lab,med,pro=zip(*fea_ratio)
    ind = np.arange(len(fea_ratio))
    plt.figure(1,figsize=(8,6))
    plt.bar(ind, pro,color='r')
    plt.bar(ind, med,color='g',bottom=pro)
    plt.bar(ind, lab,color='b',bottom=np.array(med)+np.array(pro))
    plt.xticks(ind,allk[1:])
    plt.legend(['Procedure','Med','Lab'],loc=9,ncol=3)
    plt.title('ICD9 {} Percentage of med, lab, procedure features in top features (both)'.format(case))
    plt.xlabel('Number of excluded top features')
    plt.ylabel('Percentage (%)')
    plt.ylim((0,111))
    plt.yticks(np.arange(0, 101, 10))
    pdf.savefig()
    

    control_featype=[Counter(fea_type_control[:k]) for k in allk[1:]]
    fea_ratio=[(x['Lab']*100/sum(x.values()),x['Med']*100/sum(x.values()),x['Pro']*100/sum(x.values())) for x in union_featype]
    lab,med,pro=zip(*fea_ratio)
    ind = np.arange(len(fea_ratio))
    plt.figure(2,figsize=(8,6))
    plt.bar(ind, pro,color='r')
    plt.bar(ind, med,color='g',bottom=pro)
    plt.bar(ind, lab,color='b',bottom=np.array(med)+np.array(pro))
    plt.xticks(ind,allk[1:])
    plt.legend(['Procedure','Med','Lab'],loc=9,ncol=3)
    plt.title('ICD9 {} Percentage of med, lab, procedure features in top features (control)'.format(case))
    plt.xlabel('Number of excluded top features')
    plt.ylabel('Percentage (%)')
    plt.ylim((0,111))
    plt.yticks(np.arange(0, 101, 10))
    pdf.savefig()
    pdf.close()
    plt.close('all')


for c in cases:
    plot_feasc(c)
    plot_fea_bar(c)