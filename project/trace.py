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
qq = glob.glob('C:/LocationOfNegativeCases/*.png') #Make list of negative cases
rr = glob.glob('C:/LocationOfPositiveCases/*.png') #Make list of positive cases
bb = glob.glob('C:/LocationofTraceImage/*.png') #Image to trace

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
        chk1 = []
        
        for _, _, future in matches1:
            best_matches1.append(future.result()[0][0])
        
        best_matches1.sort(reverse=True)
        
        for idx, template_name, future in matches1:
            result = future.result()
            _, max_val, _, _ = cv2.minMaxLoc(result)
            chk1.append((idx, template_name, max_val))
        
        # Sort best matches by index
        chk1.sort(key=lambda x: x[2], reverse=True)

        # Templates 2
        templates2 = [cv2.imread(template_path, 0) for template_path in rrr]
        template_names2 = [os.path.basename(template_path) for template_path in rrr]
        
        matches2 = []
        chk2 = []
        
        with ThreadPoolExecutor(20) as executor:
            for idx, (template_name, template) in enumerate(zip(template_names2, templates2), start=1):
                future = executor.submit(cv2.matchTemplate, img_gray, template, cv2.TM_CCOEFF_NORMED)
                matches2.append((idx, template_name, future))
        
        best_matches2 = []
        
        for _, _, future in matches2:
            best_matches2.append(future.result()[0][0])
        
        best_matches2.sort(reverse=True)
        
        for idx, template_name, future in matches2:
            result = future.result()
            _, max_val, _, _ = cv2.minMaxLoc(result)
            chk2.append((idx, template_name, max_val))
        
        # Sort best matches by index
        chk2.sort(key=lambda x: x[2], reverse=True)
        
        # Best X Method
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

        chrk="noscar" in gray
        if countyes < countno and chrk == True:
            tn+=1
            goodn+=1
            dpath1=chk1[0][1]
            dpath2=chk2[0][1]
        elif countyes < countno and chrk == False:
            fn+=1
            badn+=1
            print(chk1[0])
            dpath1=chk1[0][1]
            dpath2=chk2[0][1]
        if countyes > countno and chrk == True:
            fp+=1
            badp+=1
            print(chk2[0])
            dpath1=chk2[0][1]
            dpath2=chk1[0][1]
        elif countyes > countno and chrk == False:
            tp+=1
            goodp+=1
            dpath1=chk2[0][1]
            dpath2=chk1[0][1]
        if countyes == countno:
            print('ERROR 2')
    
        count+=1
        
    patnum+=1

#%% Visual Comparison Creator

# Load correct images
dpath1 = 'C:/BaseFolderLocation/' + dpath1 #This should have all cases in it. If not, combine or rewrite to separately pull files depending on label

imageA = cv2.imread(gray, 0)
imageB = cv2.imread(dpath1, 0)

# Create masks for highlighting
mask_diff_A_bright = imageA > (imageB + 50)  # imageA brighter than imageB
mask_diff_B_bright = imageB > (imageA + 50)  # imageB brighter than imageA

# Define circular mask
center = (imageA.shape[1] // 2, imageA.shape[0] // 2)
radius = 37
mask = np.zeros_like(imageA)
cv2.circle(mask, center, radius, 255, -1)
mask_inv = cv2.bitwise_not(mask)

# Create white background
white_region = np.ones_like(imageA) * 255

# Apply masks to images
masked_imageA = cv2.bitwise_and(imageA, mask)
outside_circular_regionA = cv2.bitwise_and(white_region, mask_inv)
imageA_with_background = cv2.add(masked_imageA, outside_circular_regionA)

masked_imageB = cv2.bitwise_and(imageB, mask)
outside_circular_regionB = cv2.bitwise_and(white_region, mask_inv)
imageB_with_background = cv2.add(masked_imageB, outside_circular_regionB)

diff = cv2.bitwise_and(imageA - imageB, mask)
outside_circular_regionDiff = cv2.bitwise_and(white_region, mask_inv)
diff_with_background = cv2.add(diff, outside_circular_regionDiff)

# Create highlight overlays
highlight_color_A = np.zeros_like(np.stack([imageB_with_background] * 3, axis=-1))
highlight_color_A[mask_diff_A_bright] = [0, 200, 200]  # Red

highlight_color_B = np.zeros_like(np.stack([imageA_with_background] * 3, axis=-1))
highlight_color_B[mask_diff_B_bright] = [200, 0, 200]  # Green

# Apply alpha blending
alpha = 0.5  # Transparency level
highlighted_image = cv2.addWeighted(np.stack([imageB_with_background] * 3, axis=-1), 1, highlight_color_A, alpha, 0)
highlighted_imageB = cv2.addWeighted(np.stack([imageA_with_background] * 3, axis=-1), 1, highlight_color_B, alpha, 0)

# Overlay power
overlay = cv2.addWeighted(highlighted_image, 0.5, highlighted_imageB, 0.5, 0)

# Plot images
fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=200, facecolor='white')
axes[0].imshow(imageA_with_background, cmap='gray', vmin=0, vmax=255)
axes[0].axis('off')

axes[1].imshow(imageB_with_background, cmap='gray', vmin=0, vmax=255)
axes[1].axis('off')

axes[2].imshow(overlay)
axes[2].axis('off')

plt.tight_layout()
plt.show()

#%% Timer                                
print(f"--- %.4f seconds ---" % (time.time() - start_time))