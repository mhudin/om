# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:26:37 2023

@author: Pesto
"""

qq = glob.glob('E:/Sets/optino/*.png') #source folder for negative cases
rr = glob.glob('E:/Sets/optiyes/*.png') #source folder for positive cases


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