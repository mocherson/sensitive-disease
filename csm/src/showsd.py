
from os.path import join
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cPickle as pk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)
    
path='/home/data/sensitive_disease/csm/'

allk=[0,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
index=['All']+['Excl '+str(x)+' best' for x in allk[1:]]
columns=['Precision','Recall','F-measure','AUC']
# cases=[311,632,654,644,658,616,303,655,623,661,765,608,646,652,257,663,300,296,659]
cases=[257, 294, 296, 300, 303, 305, 307, 309, 311, 607, 608, 616, 623, 626, 628, 632, 640, 642, 644, 646, 648, 650, 652, 654, 655, 656, 658, 659, 661, 663, 664, 669, 765, 770, 774, 779]


metric=['Precision','Recall','F-measure','AUC']

feanum=pd.DataFrame.from_csv('/home/data/sensitive_disease/csm/results/chi2_onetoall/featurenumber.txt',header=-1,sep=' ')

# ctr_lab_feaset=pd.Series.from_csv(join(path,'nmset/labctrl_dx_match_nmset_300.csv')).index
# ctr_pr_feaset=pd.Series.from_csv(join(path,'nmset/prctrl_dx_match_nmset_300.csv')).index
# ctr_med_feaset=pd.Series.from_csv(join(path,'nmset/medctrl_dx_match_nmset_300.csv')).index

def plotsd(case,ctr='ctrl_dx_match',sc_func='chi2'):
    pdf=PdfPages(join(path,'results',sc_func,'fig_{}.pdf'.format(case)))

    res_union=load_obj(join(path,'results',sc_func,'{}_{}_union.pkl'.format(case,ctr)))
    s_union=np.array(zip(*res_union)[0])
    pd.DataFrame(s_union,index=index[:s_union.shape[0]],columns=columns).to_csv( \
        join(path,'results',sc_func,'{}_both.csv'.format(case)))
    
    res_control=load_obj(join(path,'results',sc_func,'{}_{}_control.pkl'.format(case,ctr)))
    s_control=np.array(zip(*res_control)[0])
    pd.DataFrame(s_control,index=index[:s_control.shape[0]],columns=columns).to_csv( \
        join(path,'results',sc_func,'{}_ctr.csv'.format(case)))
    
    res_intersect=load_obj(join(path,'results',sc_func,'{}_{}_intersect.pkl'.format(case,ctr)))
    s_intersect=np.array(zip(*res_intersect)[0])
    pd.DataFrame(s_intersect,index=index[:s_intersect.shape[0]],columns=columns).to_csv(  \
        join(path,'results',sc_func,'{}_inter.csv'.format(case)))

    # case_lab_feaset=pd.Series.from_csv(join(path,'nmset/lab{}_nmset_50.csv'.format(case))).index
    # case_pr_feaset=pd.Series.from_csv(join(path,'nmset/pr{}_nmset_50.csv'.format(case))).index
    # case_med_feaset=pd.Series.from_csv(join(path,'nmset/med{}_nmset_50.csv'.format(case))).index
    # num_union=len(case_lab_feaset | ctr_lab_feaset)*2+len(case_pr_feaset | ctr_pr_feaset)+len(case_med_feaset | ctr_med_feaset)
    # num_intersect=len(case_lab_feaset & ctr_lab_feaset)*2+len(case_pr_feaset & ctr_pr_feaset)+len(case_med_feaset & ctr_med_feaset)
    # num_control=len( ctr_lab_feaset)*2+len( ctr_pr_feaset)+len( ctr_med_feaset)
    #res=np.concatenate([s_union[...,None],s_control[...,None],s_intersect[...,None]],axis=2)
    res=[s_union,s_control,s_intersect]
    #res=[s_union]
    
    for j in range(4):
        plt.figure(j)
        plt.plot(allk[:res[0].shape[0]],res[0][:,j],'r-',lw=4)
        plt.plot(allk[:res[1].shape[0]],res[1][:,j],'b--',lw=4)
        plt.plot(allk[:res[2].shape[0]],res[2][:,j],'g:',lw=4)
        plt.ylim( 0.5, 1 ) if j==3 else plt.ylim( 0, 1 )
        plt.legend(['union ({})'.format(feanum.loc[str(case),4]),  \
                    'control ({})'.format(feanum.loc[str(case),2]), \
                    'intersect ({})'.format(feanum.loc[str(case),3])],fontsize=20)
        #plt.legend(['union features (2213)'])
        plt.xlabel('Number of excluded top features',fontsize=20)
        plt.ylabel(metric[j],fontsize=20)
        plt.title('ICD9 {}'.format(case),fontsize=20)
        params={'xtick.labelsize':18,
         'ytick.labelsize':18}
        pylab.rcParams.update(params)
        plt.tight_layout()
        pdf.savefig()
        
    pdf.close()
    plt.close('all')
        

def feacomp(case):
    path='/home/data/sensitive_disease/csm/results'
    res_chi2=load_obj(join(path,'chi2','{}_nonSD5pct_union_unify.pkl'.format(case)))
    res_f=load_obj(join(path,'f_classif','{}_nonSD5pct_union_unify.pkl'.format(case)))
    res_mi=load_obj(join(path,'mi_classif','{}_nonSD5pct_union_unify.pkl'.format(case)))
    res_lr1=load_obj(join(path,'LR_classif1','{}_nonSD5pct_union_unify.pkl'.format(case)))
    res_rd=load_obj(join(path,'random','{}_nonSD5pct_random_unify.pkl'.format(case)))
    s_chi2=np.array(zip(*res_chi2)[0])
    s_f=np.array(zip(*res_f)[0])
    s_mi=np.array(zip(*res_mi)[0])
    s_lr1=np.array(zip(*res_lr1)[0])
    s_rd=np.array(zip(*res_rd)[0])
    res=np.concatenate((s_chi2[...,None],s_f[...,None],s_mi[...,None],s_lr1[...,None],s_rd[...,None]),axis=2)

    metric=['Precision','Recall','F-measure','AUC']
    for j in range(res.shape[1]):
        plt.figure(j,figsize=(8,6))
        for i in range(res.shape[2]):
            plt.plot(allk,res[:,j,i])
    #    plt.ylim( 0, 1 )
        plt.legend(['chi2','f_classif','mi_classif','LR1','random'])
        plt.xlabel('Number of excluded top features')
        plt.ylabel(metric[j])
        plt.title('ICD9 {}'.format(case))


    
for c in cases:
    print "printing figure for case ",c
    plotsd(c,ctr='allother_allctrl',sc_func='chi2_onetoall_90')




