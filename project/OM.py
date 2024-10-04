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

#%% Path Info
qq = glob.glob('C:/LocationOfNegativeTemplateSet/*.png') #Make list of negative template cases
rr = glob.glob('C:/LocationOfPostiveTemplateSet/*.png') #Make list of positive template cases
bb = glob.glob('C:/LocationofExternalTestingDataset/*.png') #External testing dataset cases

#%% Create patient-level labels and counting number of files per patient
Z=0 #patient number counter
N=[] #list of number of images per patient
S=[] #patient-level labels

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

#%% Loop
    
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
rec=[]

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
        # gray = 'E:/Sets/' + switch + '\\' + os.path.basename(bb[count]) #get target files from switch set
        gray=bb[count]

        # remove all files from this patient from the test dataset
        patma = '\\' + os.path.basename(bb[count])[:-10] #adds backslash to base file name e.g. \scar151
        [qqq.remove(i) for i in qq if patma in i] #must be qq, if qqq, fails to remove all the files!
        [rrr.remove(i) for i in rr if patma in i] #must be rr, if rrr, fails to remove all the files!            
        

        img_gray=cv2.imread(gray,0)

        # Templates 1
        templates1 = [cv2.imread(template_path, 0) for template_path in qqq]
        template_names1 = [os.path.basename(template_path) for template_path in qqq]
        
        matches1 = []
        
        with ThreadPoolExecutor(20) as executor:
            for idx, (template_name, template) in enumerate(zip(template_names1, templates1), start=1):
                future = executor.submit(cv2.matchTemplate, img_gray, template, cv2.TM_CCOEFF_NORMED)
                matches1.append((idx, template_name, future))
        
        best_matches1 = []
        
        for _, _, future in matches1:
            best_matches1.append(future.result()[0][0])
        
        best_matches1.sort(reverse=True)

        # Templates 2
        templates2 = [cv2.imread(template_path, 0) for template_path in rrr]
        template_names2 = [os.path.basename(template_path) for template_path in rrr]
        
        matches2 = []
        
        with ThreadPoolExecutor(20) as executor:
            for idx, (template_name, template) in enumerate(zip(template_names2, templates2), start=1):
                future = executor.submit(cv2.matchTemplate, img_gray, template, cv2.TM_CCOEFF_NORMED)
                matches2.append((idx, template_name, future))
        
        best_matches2 = []
        
        for _, _, future in matches2:
            best_matches2.append(future.result()[0][0])
        
        best_matches2.sort(reverse=True)
        
        # Best 10 Method
        depth=1
        
        countno=sum(best_matches1[:depth])/depth
        countyes=sum(best_matches2[:depth])/depth
        
        if countyes == 0 and countno > 0:
            pred.append(0)
        elif countno == 0 and countyes > 0:
            pred.append(1)
        else:
            if countno == 0 and countyes == 0:
                print('ERROR 1')
            else:
                pred.append(countyes/(countno+countyes))
                rec.append(countyes/(countno+countyes))

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
print(f'True positives: {tpc}')
print(f'False negatives: {fnc}')
print(f'True negatives: {tnc}')
print(f'False positives: {fpc}')
print('')
print(f'patient acc: {pacc}')
print('')
print(f'individual acc: {iacc}')
print('')

sens = round(tpc/(tpc+fnc)*100,1)
spec = round(tnc/(tnc+fpc)*100,1)
print(f"Sensitivity/Recall: {sens}%")
print(f"Specificity: {spec}%")
prec = round(tpc/(tpc+fpc)*100,1)
print(f"Precision: {prec}%")
f1 = round(((2*prec*sens)/(prec+sens)),1)
print(f"F1 score: {f1}%")
print('')
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
print('Sensitivity/Recall: %.1f%%' % (sens * 100.0))
print('Specificity: %.1f%%' % (spec * 100.0))
prec = tp/(tp+fp)
print('Precision: %.1f%%' % (prec * 100.0))
f1 = (2*prec*sens)/(prec+sens)
print('F1-score: %.1f%%' % (f1 * 100.0))

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
    
#%% Timer                                
print(f"--- %.4f seconds ---" % (time.time() - start_time))
