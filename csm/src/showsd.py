
from os.path import join
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)
    
path='/home/data/sensitive_disease/csm/results'

allk=[0,10,20,30,40,50,100,200,300,400,500,600,700,800,900,1000]
index=['All']+['Excl '+str(x)+' best' for x in allk[1:]]
columns=['Precision','Recall','F-measure','AUC']
cases=[311,632,654,644,658,616,303,655,623,661,765,608,646,652]

def plotsd(case,sc_func='chi2'):
    pdf=PdfPages(join(path,sc_func,'fig_{}.pdf'.format(case)))

    prec=np.zeros((len(allk),2))
    rec=np.zeros((len(allk),2))
    f1=np.zeros((len(allk),2))
    auc=np.zeros((len(allk),2))

    s=np.zeros((len(allk),4))
    res=load_obj(join(path,sc_func,str(case)+'_nonSD5pct_union_unify.pkl'))
    for i,r in enumerate(res):
        s[i]=r[0]

    pd.DataFrame(s,index=index,columns=columns).to_csv(join(path,sc_func,'{}_both.csv'.format(case)))

    prec[:,0]=s[:,0]
    rec[:,0]=s[:,1]
    f1[:,0]=s[:,2]
    auc[:,0]=s[:,3]



    s=np.zeros((len(allk),4))
    res=load_obj(join(path,sc_func,str(case)+'_nonSD5pct_control_unify.pkl'))
    for i,r in enumerate(res):
        s[i]=r[0]

    pd.DataFrame(s,index=index,columns=columns).to_csv(join(path,sc_func,'{}_ctr.csv'.format(case)))
    prec[:,1]=s[:,0]
    rec[:,1]=s[:,1]
    f1[:,1]=s[:,2]
    auc[:,1]=s[:,3]



    plt.figure(1)
    plt.plot(allk,prec[:,0],'r-')
    plt.plot(allk,prec[:,1],'b--')
    #plt.xlim( 0 )
    plt.ylim( 0, 1 )
    plt.legend(['both features','control features'])
    plt.xlabel('Number of excluded top features')
    plt.ylabel('Precision')
    plt.title('ICD9 {}'.format(case))
    pdf.savefig() 


    plt.figure(2)
    plt.plot(allk,rec[:,0],'r-')
    plt.plot(allk,rec[:,1],'b--')
    plt.ylim( 0, 1 )
    plt.legend(['both features','control features'])
    plt.xlabel('Number of excluded top features')
    plt.ylabel('Recall')
    plt.title('ICD9 {}'.format(case))
    pdf.savefig() 


    plt.figure(3)
    plt.plot(allk,f1[:,0],'r-')
    plt.plot(allk,f1[:,1],'b--')
    plt.ylim( 0, 1 )
    plt.legend(['both features','control features'])
    plt.xlabel('Number of excluded top features')
    plt.ylabel('F-measure')
    plt.title('ICD9 {}'.format(case))
    pdf.savefig() 


    plt.figure(4)
    plt.plot(allk,auc[:,0],'r-')
    plt.plot(allk,auc[:,1],'b--')
    plt.ylim( 0.5, 1 )
    plt.legend(['both features','control features'])
    plt.xlabel('Number of excluded top features')
    plt.ylabel('AUC')
    plt.title('ICD9 {}'.format(case))
    pdf.savefig() 
    pdf.close()
    plt.close('all')

for c in cases:
    print "printing figure for case "+str(c)
    plotsd(c)




