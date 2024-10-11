# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:27:24 2023

@author: Michael H. Udin
"""

#%% Timer
import time
start_time = time.time()

#%% Imports
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import glob
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed

#%% Path Info (Files will be removed by AE so maky a copy first!)
qq = glob.glob('E:/Sets/optino/*.png') #source folder for negative template set
rr = glob.glob('E:/Sets/optiyes/*.png') #source folder for positive template set
bb=qq.copy()+rr.copy()

#%% Create patient-level labels and counting number of files per patient
Z=0 #patient number counter
N=[] #list of number of images per patient
S=[] #patient-level labels
negative='noscar'

mismatch='x'
num=0

for file in bb:    
    match=os.path.basename(file)[:-10]
    if file == bb[0]: #if not first file in bb
        mismatch=match
    if match != mismatch:
        Z+=1
        S.append(0) if 'noscar' in mismatch else S.append(1)
        N+=[num]
        num=0
    mismatch=match
    num+=1
    if file == bb[-1]:
        Z+=1
        S.append(0) if 'noscar' in mismatch else S.append(1)
        N+=[num]

print('Number of patients:', Z)
print('Number of files:', len(bb))

#%% Create single-image labels
SS = [0 if 'noscar' in os.path.basename(file)[:-10] else 1 for file in bb] # forifelse in one line

#%% Define template-matching function
def match_it(gray, image):
    return(cv2.matchTemplate(cv2.imread(gray,0), cv2.imread(image,0), cv2.TM_CCOEFF_NORMED))

#%% Define starting threshold
tset=[0.90]

#%% Storage area
qstor1=np.zeros((len(qq), 1))
rstor1=np.zeros((len(rr), 1))
qstor2=np.zeros((len(qq), 1))
rstor2=np.zeros((len(rr), 1))

#%% Main loop
for thrsh in tset:
    
    tpc=0
    fnc=0
    tnc=0
    fpc=0
    
    count=0
    patnum=0
    
    goodn = 0
    goodp = 0
    badn = 0
    badp = 0
    
    pred=[]

    for num in N:
        
        tp=0
        fn=0
        tn=0
        fp=0
        starno=0
        staryes=0
        
        for patim in range(num):
            qqq = qq.copy() # make copies of lists so .remove works on them properly
            rrr = rr.copy() # make copies of lists so .remove works on them properly
            gray=bb[count]
    
            # remove all files from this patient from the test dataset
            patma = '\\' + os.path.basename(bb[count])[:-10] #adds backslash to base file name e.g. \scar151
            [qqq.remove(i) for i in qq if patma in i] #must be qq, if qqq, fails to remove all the files!
            [rrr.remove(i) for i in rr if patma in i] #must be rr, if rrr, fails to remove all the files!            

            # Vote count variables
            countyes=0
            countno=0
            
            # Threshold reducer
            sub=0
            
            chrk=negative in gray
            img_gray=cv2.imread(gray,0)

            while countno == 0 and countyes == 0:
                #%% No Counter
                for img in qqq:
                    template = cv2.imread(img,0)
                    w, h = template.shape[::-1] 
                    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
                    loc = np.where(res >= thrsh - sub)
                    if len(loc[0]) > 0:
                        countno+=1
                        if chrk == True:
                            qstor1[qq.index(img)]+=1
                        if chrk == False:
                            qstor2[qq.index(img)]-=1
        
                #%% Yes counter
                for img in rrr:
                    template = cv2.imread(img,0)
                    w, h = template.shape[::-1] 
                    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
                    loc = np.where(res >= thrsh - sub)
                    if len(loc[0]) > 0:
                        countyes+=1
                        if chrk == False:
                            rstor1[rr.index(img)]+=1
                        if chrk == True:
                            rstor2[rr.index(img)]-=1

                if countno == 0 and countyes == 0:
                   sub+=0.01
    
            countno=countno/len(qqq)
            countyes=countyes/len(rrr)
            
            if countyes == 0 and countno > 0:
                pred.append(0)
            elif countno == 0 and countyes > 0:
                pred.append(1)
            else:
                if countno == 0 and countyes == 0:
                    print('ERROR 1')
                else:
                    pred.append(countyes/(countno+countyes))
    
            chrk="noscar" in gray
            if countyes < countno and chrk == True:
                tn+=1
                goodn+=1
            elif countyes < countno and chrk == False:
                fn+=1
                badn+=1
            if countyes > countno and chrk == True:
                fp+=1
                badp+=1
            elif countyes > countno and chrk == False:
                tp+=1
                goodp+=1
            if countyes == countno:
                print('ERROR 2')
                
            starno+=countno
            staryes+=countyes
    
            if patim == num - 1 and S[patnum] == 0:
                if tn > fp:
                    tnc+=1
                if fp > tn:
                    fpc+=1
                if fp == tn:
                    if starno > staryes:
                        tnc+=1
                    if starno < staryes:
                        fpc+=1
                    if starno == staryes:
                        print('ERROR 3-1')
            if patim == num - 1 and S[patnum] == 1:
                if tp > fn:
                    tpc+=1
                if fn > tp:
                    fnc+=1
                if fn == tp:
                    if starno > staryes:
                        fnc+=1
                    if starno < staryes:
                        tpc+=1
                    if starno == staryes:
                        print('ERROR 3-2')
        
            count+=1
            
        patnum+=1
    
    iacc = (goodn+goodp)/(goodn+goodp+badn+badp)
    pacc = (tpc+tnc)/(tpc+tnc+fpc+fnc)
    
    print('')
    print(f'Threshold: {thrsh}')
    print('')
    print(f'True positives: {tpc}')
    print(f'False negatives: {fnc}')
    print(f'True negatives: {tnc}')
    print(f'False positives: {fpc}')
    print('')
    print(f'patient acc: {pacc}')
    print('')
    print(f'individual acc: {iacc}')
    print('')
    
    tp=goodp
    fn=badn
    tn=goodn
    fp=badp
    print(f'True positives: {tp}')
    print(f'False negatives: {fn}')
    print(f'True negatives: {tn}')
    print(f'False positives: {fp}')
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    print('Sensitivity/Recall: %.2f%%' % (sens * 100.0))
    print('Specificity: %.2f%%' % (spec * 100.0))
    prec = tp/(tp+fp)
    print('Precision: %.2f%%' % (prec * 100.0))
    f1 = (2*prec*sens)/(prec+sens)
    print('F1-score: %.2f%%' % (f1 * 100.0))
    
    # ROC stuff
    auc = roc_auc_score(SS, pred)
    print('AUCROC: %.3f' % (auc))
    rauc = round(auc,3)
    
    #%% Make a magical ROC curve from templates
    fpr, tpr, _ = roc_curve(SS, pred)
    
    plt.figure(figsize=(8, 8), dpi= 160, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 22})
    plt.gca().set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    plt.plot([0, 1], ls="--")
    plt.plot(fpr, tpr, marker='.', label='Curve, AUC: {: .3f}'.format(rauc))
    
    plt.legend(loc='lower right')
    plt.show()
    
count=0
for img in qq:
    if abs(qstor1[count]) < abs(qstor2[count]):
        os.remove(img)
    count+=1
    
count=0
for img in rr:
    if abs(rstor1[count]) < abs(rstor2[count]):
        os.remove(img)
    count+=1
    
#%% Timer                                
print(f"--- %.4f seconds ---" % (time.time() - start_time))