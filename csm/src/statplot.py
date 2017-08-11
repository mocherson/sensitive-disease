from __future__ import division
from os.path import join
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab

    
path='/home/data/sensitive_disease'

# cases=[311,632,654,644,658,616,303,655,623,661,765,608,646,652,257,663,300,296,659]
cases=[257, 294, 296, 300, 303, 305, 307, 309, 311, 607, 608, 616, 623, 626, 628, 632, 640, 642, 644, 646, 648, 650, 652, 654, 655, 656, 658, 659, 661, 663, 664, 669, 765, 770, 774, 779,'control_cohort']

def statplot(case,nbin=50):
    data=pd.read_csv(join(path,'{}_dx_count.csv'.format(case)))
    dx_cnt=data['distinct_dx_cnt']
    pdf=PdfPages(join(path,'csm/statplot/{}_dx_histogram.pdf'.format(case)))
    
    plt.figure(1,figsize=(8,6))
    n, bins, patches=plt.hist(dx_cnt,bins=nbin)
    binwidth=(max(dx_cnt)-min(dx_cnt))/nbin
    mu=np.mean(dx_cnt)
    sigma=np.std(dx_cnt)
    y = mlab.normpdf( bins, mu, sigma)*binwidth*len(dx_cnt)
    plt.plot(bins,y,'r--')
    plt.xlabel('number of diseases')
    plt.ylabel('number of patients')
    plt.legend(['norm fitted'])
    plt.title('the histogram of diseases vs patients in case {}'.format(case))
    pdf.savefig()
    pdf.close()
    plt.close('all')
    
for c in cases:
    print 'ploting case',c
    statplot(c)