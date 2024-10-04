# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:00:17 2023

@author: Michael H. Udin
"""
#%% Timer
import time
start_time = time.time()

#%% Imports
from tensorflow.keras.applications.resnet50 import ResNet50 as RN
from tensorflow.keras.applications.resnet_v2 import ResNet152V2 as RN2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
import tensorflow as tf
import cv2
import glob
import numpy as np    
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import psutil
import os
import multiprocessing
import pandas as pd
import logging
import scipy.stats as st
import copy

#check
import PIL
from PIL import Image
import tensorflow_datasets as tfds
import pathlib
from shutil import copyfile
import shutil

#%% Memory control
apple = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(apple[0], True)

#%% TF output control (hides non-critical output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#%% Input parameters – Less than 2 minutes though YTMV depending on your video card.
# Path information
setname = 'xmde7' #which set to draw images from #Modify as needed to match your computer
files = glob.glob('E:/Sets/' + setname + '/*.png') #Modify as needed to match your computer
switch = 'xmde1' #set which set to take images from if switching #Modify as needed to match your data
switchon = 0 #0 for off, 1 for on

# Machine learning variables
if switchon == 0:
    size, _ = cv2.imread(files[0], 0).shape # extract image size (size of one dimension, e.g. 512 for a 512x512)
else:
    fileswitch = glob.glob('E:/Sets/' + switch + '/*.png')
    size, _ = cv2.imread(fileswitch[0], 0).shape # extract image size (size of one dimension, e.g. 512 for a 512x512)
ep1 = 20 #number of epochs
split = 20 #number of folds
rss = 24 #set number for random number generator
bat = 64 #batch size – higher numbers should be paired with higher learning rates
insh = (size, size, 1) #input shape (size, size, 3) or (size, size, 1) for greyscale
opt = RMSprop(learning_rate=0.0005) #set learning rate, RMSprop default = 0.001
useModel = RN #pick which CNN to use (RN=ResNet50, RN2=Resnet152V2)

#%% ROC Creator
def rocCreator(fprs,tprs): 
                   
    idx_max = fprs.index((max(fprs, key=len)))
    fpr_max = fprs[idx_max]
        
    fpr_interp = list()
    tprs_interp = list()
    
    for b in range(len(fprs)):
        fpr_temp =  interp1d(fprs[b],tprs[b])
        fpr_interp.append(fpr_temp) 
        
        tpr_temp = fpr_interp[b](fpr_max)
        tprs_interp.append(tpr_temp)
        
    tpr_stack =  np.stack(tprs_interp,axis=0)
    tpr_std = np.std(tpr_stack, axis=0)
    tpr_avg = np.mean(tpr_stack ,axis=0)
        
    avg_roc_auc = auc(fpr_max, tpr_avg)
    

    plt.figure(figsize=(8, 8), dpi= 160, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 22})
    plt.gca().set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    tpr_avg = np.append([0],tpr_avg)
    fpr_max = np.append([0],fpr_max)
    tpr_std = np.append([0],tpr_std)
    
    plt.plot(fpr_max, tpr_avg, label='Mean curve, AUC: {:.4f}'.format(avg_roc_auc))
    plt.fill_between(fpr_max, tpr_avg+tpr_std, tpr_avg-tpr_std, alpha=0.5)

    plt.plot([0, 1], ls="--")

    plt.legend(loc='best')
    plt.show()
    
    print('Average ROC AUC =', avg_roc_auc)

#%% Find unique patients and create labels patient labels
# Create list of base filenames
S = [os.path.basename(file)[:-10] for file in files] #create list of base filenames

# Keep the unique patients only
V=[]
for x in S:
    if x not in V:
        V.append(x)

# Create labels for unique patients
W = [0 if 'noscar' in val else 1 for val in V]

# Display as a check
print(V)
print(W)

# Prepare for use in stratified shuffle split
V = np.array(V)
W = np.array(W)


#%% Create patient-level labels and counting number of files per patient
Z=0 #patient number counter
N=[] #list of number of images per patient
S=[] #patient-level labels

bb=glob.glob('E:/Sets/' + 'xmde8' + '/*.png') #Modify as needed to match your computer

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

#%% External testing dataset
setname2 = 'xmde8' #which set to draw images from, modify as needed to match your data
xx = glob.glob('E:/Sets/' + setname2 + '/*.png') #Modify as needed to match your computer
XX=[]
exTest=[]

for im in xx:
    image = cv2.imread(im) #in (file, 0) the , 0  reads as grayscale!
    nm = os.path.basename(im)[:-10]
    dc = 0 if 'noscar' in nm else 1
    exTest.append(image)
    XX.append(dc)

exTest = np.array(exTest)
XX=np.array(XX)

#%% Main loop
# Create stratified shuffle split for x splits(folds)
from sklearn.model_selection import StratifiedShuffleSplit    
sss = StratifiedShuffleSplit(n_splits=split, test_size=0.20, random_state=rss)
sss.get_n_splits(V, W)

count=0

sen=[]
spe=[]
pre=[]
f1s=[]
ac=[]
tprs=[]
fprs=[]
roc=[]
ztpc=[]
zfnc=[]
ztnc=[]
zfpc=[]
savpred=[]

zero = ['0'] * 10 + [''] * 90

# The main loop
for train_index, test_index in sss.split(V, W):
    
    V_train, V_test = V[train_index], V[test_index]
    W_train, W_test = W[train_index], W[test_index]
    
    Y_test = []
    
    count+=1
    
    #Create directories for noscar and scar within a dataset holder space
    nopath = 'E:/Sets/xmdex/noscar/' #Modify as needed to match your computer
    scpath = 'E:/Sets/xmdex/scar/' #Modify as needed to match your computer
    os.makedirs(nopath, exist_ok=True) #creates spath directory if doesn't exist
    os.makedirs(scpath, exist_ok=True) #creates spath directory if doesn't exist
    
    for file in files:
        tr = os.path.basename(file)[:-10]
        if tr in V_train:
            nm = os.path.basename(file)
            if 'noscar' in nm:
                copyfile(file,nopath+nm)
            else:
                copyfile(file,scpath+nm)

    data_dir = pathlib.Path('E:/Sets/xmdex/') #Modify as needed to match your data location
    image_count = len(list(data_dir.glob('*/*.png')))
    img_height=80
    img_width=80
    batch_size=32

    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.3,
      subset="training",
      seed=count*10,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.3,
      subset="validation",
      seed=count*10,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = tf.keras.Sequential([
      tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
      optimizer='adam',
      loss='binary_crossentropy',
      metrics=['accuracy'])
    
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    
    class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
    
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    
    
    def get_label(file_path):
      # Convert the path to a list of path components
      parts = tf.strings.split(file_path, os.path.sep)
      # The second to last is the class-directory
      one_hot = parts[-2] == class_names
      # Integer encode the label
      return tf.argmax(one_hot)
    
    def decode_img(img):
      # Convert the compressed string to a 3D uint8 tensor
      img = tf.io.decode_jpeg(img, channels=3)
      # Resize the image to the desired size
      return tf.image.resize(img, [img_height, img_width])
    
    def process_path(file_path):
      label = get_label(file_path)
      # Load the raw data from the file as a string
      img = tf.io.read_file(file_path)
      img = decode_img(img)
      return img, label
    
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    
    def configure_for_performance(ds):
      ds = ds.cache()
      ds = ds.shuffle(buffer_size=1000)
      ds = ds.batch(batch_size)
      ds = ds.prefetch(buffer_size=AUTOTUNE)
      return ds
    
    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    
    image_batch, label_batch = next(iter(train_ds))
     
    
    savespot='C:/Envs/'+'SC'+setname+zero[count]+str(count)+'weight.h5' #update savespot
    
    MC = ModelCheckpoint(
    filepath=savespot,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    
    model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=30,
      callbacks=[MC]
    )
    
    model.load_weights(savespot)
    
    test_loss, test_acc = model.evaluate(exTest, XX, verbose=2)
    
    predictions = model.predict(exTest)
    savpred.append(predictions)
    # predZ = np.argmax(predictions, axis=1)
    predZ = np.round(predictions, 0)
    
    tn, fp, fn, tp = confusion_matrix(XX, predZ).ravel() #confusion matrix to get tn, fp, fn, tp

    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives: {tp}")
    print('')
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    print(f"Sensitivity/Recall: {sens}")
    print(f"Specificity: {spec}")
    print('')
    prec = tp/(tp+fp)
    print(f"Precision: {prec}")
    f1 = (2*prec*sens)/(prec+sens)
    print(f"F1 score: {f1}")
    print('')
    macc = (tp+tn)/(tp+tn+fp+fn)
    print('Accuracy: %.3f%%' % (macc * 100.0))
    print('')
    sen.append(sens)
    spe.append(spec)
    pre.append(prec)
    f1s.append(f1)
    ac.append(macc)
    
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(XX, predictions)
    rocs=roc_auc_score(XX, predictions)
    print(f"ROC AUC: {rocs}")
    print('')
    roc.append(rocs)
    fprs.append(list(false_positive_rate1))
    tprs.append(list(true_positive_rate1))

    # Patient
    icount=0
    pcount=0
    tpc=0
    fnc=0
    tnc=0
    fpc=0
    for num in N:
        ptp=0
        pfn=0
        ptn=0
        pfp=0
        grot=S[pcount]
        for pat in range(num):
            if grot == 0 and predZ[icount] == 0:
                ptn+=1
            elif grot == 0 and predZ[icount] == 1:
                pfp+=1
            elif grot == 1 and predZ[icount] == 0:
                pfn+=1
            elif grot == 1 and predZ[icount] == 1:
                ptp+=1            
            icount+=1
        pcount+=1
        
        if ptp + pfn > 0:
            if ptp > pfn:
                tpc+=1
            else:
                fnc+=1
        elif ptn + pfp > 0:
            if ptn > pfp:
                tnc+=1
            else:
                fpc+=1
                        
    ztpc.append(tpc)
    zfnc.append(fnc)
    ztnc.append(tnc)
    zfpc.append(fpc)
    
aa=np.mean(ztpc)
bb=np.mean(zfnc)
cc=np.mean(ztnc)
dd=np.mean(zfpc)

pacc = round((aa+cc)/(aa+bb+cc+dd)*100,1)

print('')
print(f'True positives: {aa}')
print(f'False negatives: {bb}')
print(f'True negatives: {cc}')
print(f'False positives: {dd}')
print('')
print(f'patient acc: {pacc}%')

tp=aa
fn=bb
tn=cc
fp=dd

sens = round(tp/(tp+fn)*100,1)
spec = round(tn/(tn+fp)*100,1)
print(f"Sensitivity/Recall: {sens}%")
print(f"Specificity: {spec}%")
prec = round(tp/(tp+fp)*100,1)
print(f"Precision: {prec}%")
f1 = round(((2*prec*sens)/(prec+sens)),1)
print(f"F1 score: {f1}%")
print('')
print('')

rocCreator(fprs,tprs)
rocn=np.mean(roc)
senn=np.mean(sen)
spen=np.mean(spe)
pren=np.mean(pre)
f1sn=np.mean(f1s)
mn=np.mean(ac)
dsen=np.std(sen)
dspe=np.std(spe)
dpre=np.std(pre)
df1s=np.std(f1s)
dac=np.std(ac)
droc=np.std(roc)

# Output final performance metrics
print('Mean ROC AUC: %.3f' % (rocn))
print('Mean Sensitivity: %.3f%%' % (senn * 100.0))
print('Mean Specificity: %.3f%%' % (spen * 100.0))
print('Mean Precision: %.3f%%' % (pren * 100.0))
print('Mean F1 Score: %.3f%%' % (f1sn * 100.0))
print('Mean Accuracy: %.3f%%' % (mn * 100.0))
print('')
print('RECORD VALUES')
print('Mean Accuracy ± stdev: %.1f%% ± %.2f%%' % (mn * 100.0, dac * 100.00))
print('Mean Sensitivity ± stdev: %.1f%% ± %.2f%%' % (senn * 100.0, dsen * 100.00))
print('Mean Specificity ± stdev: %.1f%% ± %.2f%%' % (spen * 100.0, dspe * 100.00))
print('Mean Precision ± stdev: %.1f%% ± %.2f%%' % (pren * 100.0, dpre * 100.00))
print('Mean F1-score ± stdev: %.1f%% ± %.2f%%' % (f1sn * 100.0, df1s * 100.00))
print('Mean ROC AUC: %.3f' % (rocn))
print('ROC AUC stdev: ±%.3f' % (droc))
print('')
print('')

#%% Timer                                
print(f"Total time: %.4f seconds" % (time.time() - start_time))