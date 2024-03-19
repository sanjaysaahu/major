#%%
import sys
import os
import glob
import random
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import imgaug.augmenters as iaa
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


#%%
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import save_model,load_model
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Conv2DTranspose, GlobalMaxPooling2D,Lambda,MaxPooling2D, GlobalAveragePooling2D,UpSampling2D,concatenate,Multiply,Conv2DTranspose,AvgPool2D
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
from segmentation_models import Unet
import segmentation_models as sm

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix

sm.set_framework('tf.keras')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

K.set_image_data_format('channels_last')
K.set_learning_phase(1)
tf.keras.backend.set_image_data_format('channels_last')

plt.style.use("dark_background")

#%%
train_files = []
mask_files = glob.glob('archive/lgg-mri-segmentation/kaggle_3m/*/*_mask*')
data=pd.read_csv("archive/lgg-mri-segmentation/kaggle_3m/data.csv")
for i in mask_files:
    train_files.append(i.replace('_mask',''))
    
    
#%%
df=pd.DataFrame()
df['img']=train_files
df['mask']=mask_files



#%%
def labels(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0 : return 1
    else: return 0

#%%
df['label']=df['mask'].apply(labels)

#%%

import seaborn as sns
import pandas as pd

# Assuming your label column needs to be converted to categorical data
df['label'] = df['label'].astype('category')

# Plot count of 0s and 1s separately
sns.countplot(data=df, x='label')

#%%
def path(x):
  y=x.split("/")[-1]
  print(y)
  
  z=y.split(".")[0]
  print(z)
  z1=z.split("\\")[-1]
  z1 = z1.split("_")
  print(z1)
  return "_".join(z1[:-2])

df['Patient']=df.img.apply(path)

#%%
df['Patient']

#%%
k=df.groupby(df.Patient)
l=k.size()
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
sns.barplot(x=l.index,y=l.values)
plt.show()


#%%
for name ,group in k:
    print(f"Group {name}:")
    print("\n")

#%%
rows,cols=5,5
l=k.get_group('TCGA_CS_4941')
fig=plt.figure(figsize=(16,16))
plt.title('TCGA_CS_4941')
for i in range(1,l.shape[0]):
    fig.add_subplot(rows,cols,i)
    img=cv2.imread(l['img'].iloc[i], cv2.IMREAD_UNCHANGED)
    msk_path=l['mask'].iloc[i]
    img=img/255
    msk=cv2.imread(msk_path)
    plt.imshow(img)
    #plt.imshow(msk,alpha=0.5)
plt.show()
#%%

rows,cols=3,3

fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    fig.add_subplot(rows,cols,i)
    img_path=df['img'].iloc[i]
    msk_path=df['mask'].iloc[i]
    img=cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    msk=cv2.imread(msk_path)
    plt.imshow(img)
    plt.imshow(msk,alpha=0.5)
    #plt.title(df['img'].iloc[i])

plt.show()

#%%
df['labmsk']=df['label'].apply(lambda x: str(x))

#%%
data.isnull().sum()

#%%
k=data.columns
imputer = KNNImputer(n_neighbors=4)
x=pd.DataFrame(np.round(imputer.fit_transform(data.drop('Patient',axis=1))),columns=k[1:])
for i in k[1:]:
  data[i]=x[i]

#%%
data
#%%
data_numeric = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(20,10))
sns.heatmap(data_numeric.corr(), cmap="YlGnBu", annot=True)#tumor tissue site contains same value throughout thus is dropped

#%%
print("Unique values in df['Patient']:", df['img'])
print("Unique values in data['Patient']:", data['Patient'].unique())

#%%
Data=pd.merge(df, data, how='inner', left_on = 'Patient', right_on = 'Patient')
Data.head(10)
#%%
df_train1=df.sample(frac = 0.8)
df_test1=df.drop(df_train1.index)
df_train1=df_train1.reset_index()
df_test1=df_test1.reset_index()

#%%
# df_train = df.sample(frac=0.8, random_state=42)
# df_temp = df.drop(df_train.index)
# df_val = df_temp.sample(frac=0.5, random_state=42)
# df_test = df_temp.drop(df_val.index)
# df_train = df_train.reset_index(drop=True)
# df_val = df_val.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

#%%
print(df_val)

#%%
aug = iaa.Sharpen(alpha=(1.0), lightness=(1.5))

def adjust_data(img,mask):
    #img = img[:,:,1]
    #print(img.shape)
    img = tf.where(img < 0.2, 0.5, img)
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img,mask)

class Dataset:
    # we will be modifying this CLASSES according to your data/problems
    
    # the parameters needs to changed based on your requirements
    # here we are collecting the file_names because in our dataset, both our images and maks will have same file name
    # ex: fil_name.jpg   file_name.mask.jpg
    def __init__(self, dataframe):
        
        self.ids = dataframe['Patient']
        # the paths of images
        self.images_fps   = dataframe['img']
        # the paths of segmentation images
        self.masks_fps    = dataframe['mask']
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_UNCHANGED)
        image = aug.augment_image(image)
        image=image[:,:,1]
        image= np.reshape(image, (256,256,1))
        
        image = image.astype(np.float32)
        
        mask  = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
        mask = np.reshape(mask, (256,256,1))
        image_mask = mask
        image_mask = image_mask.astype(np.float32)
        
        image,image_mask= adjust_data(image, image_mask)
        return (image,image_mask)
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        batch = [np.stack(samples) for samples in zip(*data)]
        
        return tuple(batch)
    
    def __len__(self):
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


#%%
train_dataset1 = Dataset(df_train1)
test_dataset1  = Dataset(df_test1)

BATCH_SIZE=32
train_dataloader1 = Dataloder(train_dataset1, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader1 = Dataloder(test_dataset1, batch_size=BATCH_SIZE, shuffle=True)
print(train_dataloader1)

#%%
# train_dataset = Dataset(df_train)
# val_dataset = Dataset(df_val)
# test_dataset = Dataset(df_test)

# BATCH_SIZE = 32


# train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_dataloader = Dataloder(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_dataloader = Dataloder(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# val_test_dataloader = Dataloder(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# val_test_dataloader.extend(test_dataset)


# print("Training Dataloader:", train_dataloader)
# print("Validation Dataloader:", val_dataloader)
# print("Test Dataloader:", test_dataloader)
#%%
rows,cols=3,3

fig=plt.figure(figsize=(10,10))
for i in range(1,rows*cols+1):
    index=np.random.randint(1,len(train_dataloader))
    #index=i+20
    fig.add_subplot(rows,cols,i)
    img_path=train_dataloader[index-1][0][0]
    msk_path=train_dataloader[index-1][1][0]
    plt.imshow(np.reshape(img_path,(256,256)),cmap='Greens')
    #plt.imshow(msk_path,alpha=0.5, cmap="gray")

plt.show()
#%% MODEL AND LOSESSS
def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + 100) / (K.sum(y_truef) + K.sum(y_predf) + 100))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + 100) / (sum_ - intersection + 100)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    return 1-iou(y_truef, y_predf)

#%%
def changeLearningRate(epochs,lr):
  return lr*0.95
lrschedule = LearningRateScheduler(changeLearningRate)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=4, verbose=0,mode='min')
callbacks_list = [lrschedule]
optim = tf.keras.optimizers.Adam(0.0001)

#%%

plt.rc('axes', grid=False)

def trace(model_hist):
  '''
  Plot model values(loss,dice and iou) for each epochs 
  '''
 
  plt.style.use('dark_background')  # Use the 'ggplot' style for a bright background
  # plt.rcParams['text.color'] = 'black'

  a = model_hist.history
  list_traindice = a['dice_coef']
  list_validationdice = a['val_dice_coef']

  list_trainjaccard = a['iou']
  list_validationjaccard = a['val_iou']

  list_trainloss = a['loss']
  list_validationloss = a['val_loss']
  
  list_train_accuracy = a['binary_accuracy']
  list_validation_accuracy = a['val_binary_accuracy']
  
  plt.grid(False)
  
  plt.figure(1)
  plt.plot(list_validationloss, 'b-',label='validation loss')
  plt.plot(list_trainloss,'r-',label = 'train loss')
  plt.xlabel('Number of Epochs')
  plt.ylabel('loss')
  plt.title('Loss graph', fontsize = 12)
  plt.legend()
  
  plt.figure(2)
  plt.plot(list_traindice, 'r-',label='train dice coef')
  plt.plot(list_validationdice, 'b-', label = 'validation dice coef')
  plt.xlabel('Number of Epochs')
  plt.ylabel('dice coefficient')
  plt.title('Dice coefficient graph', fontsize = 12)
  plt.legend()
  
  plt.figure(3)
  plt.plot(list_trainjaccard, 'r-', label='training IOU')
  plt.plot(list_validationjaccard, 'b-', label='validation IOU')
  plt.xlabel('Number of Epochs')
  plt.ylabel('Jaccard Index (IOU)')
  plt.title('Jaccard Index Graph', fontsize=12)
  plt.legend()  
  
  plt.figure(4)
  plt.plot(list_train_accuracy, 'r-', label='training accuracy')
  plt.plot(list_validation_accuracy, 'b-', label='validation accuracy')
  plt.xlabel('Number of Epochs')
  plt.ylabel('Accuracy')
  plt.title('Accuracy Graph', fontsize=12)
  plt.legend()  # Add legend

  
  plt.show()
  
  #%%
def plyt(model):
    '''
    Plot image, mask and predicted mask for passed model
    '''
    rows,cols=3,3
    for i in range(1,10):
        index=np.random.randint(1,len(test_dataloader))
        fig.add_subplot(rows,cols,i)
        img_path=test_dataset[index-1][0]
        msk_path=test_dataset[index-1][1]
        predicted  = model.predict(img_path[np.newaxis,:,:,:])
        predicted[predicted <0.05]=0
        plt.figure(figsize=(10,3))
        plt.subplot(131)
        plt.imshow(np.reshape(img_path,(256,256)))
        plt.title('Original Image')
        plt.subplot(132)
        plt.imshow(np.reshape(msk_path,(256,256)),cmap='gray')
        plt.title('Mask')
        plt.subplot(133)
        plt.imshow(predicted[0,:,:,0],cmap="gray")
        plt.title('Predicted')
        plt.show()



  #%%
base_model = Unet(backbone_name='inceptionv3', classes=1 ,encoder_weights='imagenet', encoder_freeze=False)

inp = Input(shape=(256, 256, 1))
l1 = Conv2D(3, (1, 1))(inp)
out = base_model(l1)
print(base_model.summary())
unet = Model(inp, out, name=base_model.name)

#%%
unet.compile(optimizer=optim, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])
unet.summary()


#%%
if os.path.exists("unet_inception.h5"):
  unet = load_model('unet_inception.h5',compile=False)
  unet.compile(optimizer=optim, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])
  print("Evaluate on test data")
  results = unet.evaluate(test_dataloader1)
  print("Test loss:{} \nTest IOU:{} \nTest Dice Coeff:{}".format(round(results[0],3),round(results[2],3),round(results[3],3)))
  

else:
  unet_hist = unet.fit(train_dataloader, steps_per_epoch=len(train_dataloader), epochs=30,validation_data=val_dataloader,verbose=1,callbacks=callbacks_list)
  trace(unet_hist)
  save_model(unet, "unet_inception.h5")
  history = unet_hist.history
  np.save('unet_history_inception.npy', history)
  
  
#%%
def print_epoch_history(model_hist):
    '''
    Print training history for each epoch.
    '''
    for epoch in range(1, len(model_hist.history['loss']) + 1):
        print(f'Epoch {epoch}:')
        print(f'Training Loss: {model_hist.history["loss"][epoch-1]}')
        print(f'Training Binary Accuracy: {model_hist.history["binary_accuracy"][epoch-1]}')
        print(f'Training Dice Coefficient: {model_hist.history["dice_coef"][epoch-1]}')
        print(f'Training Intersection over Union (IOU): {model_hist.history["iou"][epoch-1]}')
        # Add more metrics as needed
        print('-' * 30)

# Assuming `unet_hist` is your training history
print_epoch_history(unet_hist)

#%%
trace(unet_hist)
#%%
plyt(unet)

#%%
def plot_final(Data,return_image=False):
    image1=cv2.imread(Data)
    image = aug.augment_image(image1)
    image=image[:,:,1]
    image[image <0.2]=0.5
    image = image / 255
    image = np.expand_dims(image, axis=-1)
    predicted  = unet.predict(image[np.newaxis,:,:])
    predicted[predicted <0.25]=0
    img = predicted[0,:,:,0]
    mean,std=cv2.meanStdDev(img)
    
    pixels = cv2.countNonZero(img)
    image_area = img.shape[0] * img.shape[1]
    area_ratio = (pixels / image_area) * 100
    img = img*255
    img[img<1]=1
    img[img>100]=255
    M= cv2.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    if return_image:
      return img,area_ratio,std,(cX,cY)
    else:
      return area_ratio,std,(cX,cY)
  
#%%
Area=[];CX=[];CY=[];STD=[]
for i in range(len(Data)):
  area,std,coordinates = plot_final(Data['img'].iloc[i])
  Area.append(area)
  CX.append(float(coordinates[0]))
  CY.append(float(coordinates[1]))
  STD.append(float(std))
Data['area']=Area
Data['Cx']=CX
Data['Cy']=CY
Data['std']=STD

#%%
Data.drop(['mask','Patient','labmsk','label'],axis=1,inplace=True)

#%%
Data.head(10)

#%%
def rep(cor):
  if cor==127.0:
    cor=0
  return cor
Data['Cx']=Data.Cx.apply(rep)
Data['Cy']=Data.Cy.apply(rep)
#%%
X_train, X_test,y_train,y_test=train_test_split(Data.drop(['img','death01'],axis=1),Data.death01,test_size=0.2,stratify=Data.death01)

#%%
xgb=XGBClassifier(tree_method='gpu_hist', gpu_id=0)
xgb.fit(X_train, y_train)

#%%
#print(plot_confusion_matrix(xgb,X_test,y_test))

#%%
fig=plt.figure(figsize=(16,16))
rows,cols=3,3
for i in range(1,rows*cols+1):
  fig.add_subplot(rows,cols,i)
  index=np.random.randint(1,len(test_dataloader))
  mask,area,std,coordinates = plot_final(Data['img'].iloc[index],return_image=True)
  plt.imshow(cv2.imread(Data['img'].iloc[index]))
  plt.imshow(cv2.circle(mask, coordinates, int(std*area*10), (0, 255, 0), 2),alpha=0.5,cmap='gray')
  plt.imshow(mask,alpha=0.4,cmap='gray')
  if area==0.0:
    plt.title("No tumor detected", fontsize=20)
  else:
    plt.title("Area={} \n STD={} \n Centroid={}".format(area,std,(coordinates[0],coordinates[1])))
  plt.xticks([])
  plt.yticks([])
plt.show()