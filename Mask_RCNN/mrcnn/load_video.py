# -*- coding: utf-8 -*-
"""
Script Name: load_video.py

Description: This script is for performing ladle tracking inference without any Kalman filtering.

Original Author: Callum O'Donovan

Original Creation Date: April 20th 2021

Email: callumodonovan2310@gmail.com
    
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mrcnn import utils
import visualize
import mrcnn.model as modellib
import ladle
import cv2
import pandas as pd


ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_WEIGHTS_PATH = ROOT_DIR +"/mask_rcnn_coco.h5"
config = ladle.CustomConfig()
LADLE_DIR = ROOT_DIR+"\Mask_RCNN\dataset"

# Override the training configurations with a few changes for inferencing
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Set target device
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


dataset = ladle.CustomDataset()
dataset.load_custom(LADLE_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

weights_path = "C:/Users/Callum/Anaconda3/envs/Mask_RCNN/maskpool_3.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

framePath = "C:/Users/Callum/Anaconda3/envs/myenv/Mask_RCNN/dataset/val"
frameCount = len(os.listdir(framePath))
frameCount = frameCount-1 # Count number of files (frames) in "val" folder -1 for .json

framePath2 = "C:/Users/Callum/Anaconda3/envs/myenv/Mask_RCNN/dataset/val_ladle"
frameCount2 = len(os.listdir(framePath2))
frameCount2 = frameCount2-1 # Count number of files (frames) in "val" folder -1 for .json
  
resultsTable = pd.DataFrame(columns=['Frame Number','Inference Time','Precision','Recall','Overlap','AP'])
analysisTable = pd.DataFrame(columns=['Frame Number','Pouring Height','Rotation'])
roi_values = np.zeros([1,4])
trackingValues = np.zeros([10, 1])

counter = 0
kalmanCheck = 0

previousMPH = 0
previousCPH = 0
PHcounter = 0
previousRot = 0
PHlist = []

rotList = []
movingRot = 0
rotCounter = 0
previousMRot = 0
pixelBrightness = []
correctionPos = 0
correctionRot = 0

for currentImage in range(1,frameCount+1):
  print(currentImage)
  image_id = dataset.image_ids[currentImage-1] 
  image, image_meta, gt_class_id, gt_bbox, gt_mask, centers =\
      modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
  info = dataset.image_info[image_id]
  #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                       #dataset.image_reference(image_id)))
  counter += 1
  
# Run object detection
  start = time.time()
  results = model.detect([image], verbose=1)
  end = time.time()
  
  if currentImage == 1:
      rgbImage = cv2.imread(r"C:/Users/Callum/Anaconda3/envs/Mask_RCNN/dataset/val/C_Pour160_side_1.jpg", cv2.IMREAD_GRAYSCALE)
  else:
      rgbImage = cv2.imread(r"C:/Users/Callum/Anaconda3/envs/Mask_RCNN/dataset/val/C_Pour160_side_%d.jpg" %int(1+(currentImage-1)*30), cv2.IMREAD_GRAYSCALE)
      
  leftHalf =  rgbImage[0:1488,250:1000]
  
  for i in range(0,leftHalf.shape[0]):
      for j in range(0,leftHalf.shape[1]):
          brightness = leftHalf[i,j]

          if brightness > 235:
              pixelBrightness.append(0)
              
  brightPixels = len(pixelBrightness)
  print(len(pixelBrightness))
  pixelBrightness = []

# Display results
  ax = get_ax(1)
  r = results[0]

  if currentImage < 8:
      flameSeverity = 0
  elif (0 <= brightPixels <= 1000):
      flameSeverity = 0
  elif (1000 <= brightPixels <= 2000):
      flameSeverity = 1
  elif (2000 <= brightPixels <= 6000):
      flameSeverity = 2
  elif (6000 <= brightPixels <= 10000):
      flameSeverity = 3
  elif (10000 <= brightPixels <= 15000):
      flameSeverity = 4
  else:
      flameSeverity = 5

  kalmanCheck = 0 
 
  leftValue = 100000000
  # Pouring height and rotation if no masks detected
  if r['masks'].shape[2] == 0:
      #print("NO MASKS")
      df = pd.read_csv('predicted_centers%d.csv' %(counter)) #use previous mask
      df.to_csv('predicted_centers%d.csv' %(counter), index=False)
      roiDisplayArray = savedRoi
      maskDisplayArray = savedMask
      cv2.imwrite("mrcnnOut%d.png" %(counter-1), maskDisplayArray*255)
      currentPH = np.min(np.where(maskDisplayArray)[0]) #calculate current pouring height

      if previousMPH != 0:
          if previousCPH == 0:
              previousCPH = previousMPH
              
          if (currentPH - previousMPH > 30) or (previousMPH - currentPH > 30):
              movingPH = previousCPH
              currentPH = previousMPH
              
          if PHcounter%5 != 0: # If PHcounter is not a multiple of 10
              movingPH = previousCPH
              PHlist.append(currentPH)
          else:
              movingPH = np.mean(PHlist)
              PHlist = []
          previousMPH = currentPH
          previousCPH = movingPH
      else:
          movingPH = currentPH
          previousMPH = currentPH 
          previousCPH = currentPH
          
      PHcounter +=1
      movingPH = movingPH - 40 # Correction for difference between mask edge and ladle edge
   
      lPixel = np.min(np.where(maskDisplayArray)[1])
      
      # Calculate rotation and display mask
      finalPH, finalRot, previousRot, rotList, movingRot, rotCounter, previousMRot, correctionRot, correctionPos = visualize.display_instances(image, kalmanCheck, currentImage, movingPH, previousRot, rotList, movingRot, rotCounter, previousMRot, lPixel, correctionPos, correctionRot, roiDisplayArray, maskDisplayArray, r['class_ids'], 
                                  dataset.class_names, r['scores'], ax=ax,
                                  title="Predictions")  

      continue
  # Pouring height and rotation if one mask detected
  elif r['masks'].shape[2] == 1:
      maskArray = r['masks'][:,:,0]

      q = 0
      roiDisplayArray = r['rois'][q]
      maskDisplayArray = np.reshape(maskArray, (1024,1024,1))

      cv2.imwrite("mrcnnOut%d.png" %(counter-1), maskDisplayArray*255)
      lPixel = np.min(np.where(maskDisplayArray)[1])
      currentPH = np.min(np.where(maskDisplayArray)[0]) # Calculate current pouring height
 
      if previousMPH != 0:
          if previousCPH == 0:
              previousCPH = previousMPH
              
          if (currentPH - previousMPH > 30) or (previousMPH - currentPH > 30):
              movingPH = previousCPH
              currentPH = previousMPH

          if PHcounter%5 != 0: # If PHcounter is not a multiple of 5
              movingPH = previousCPH
              PHlist.append(currentPH)
          else:
              movingPH = np.mean(PHlist)
              PHlist = []
          previousMPH = currentPH
          previousCPH = movingPH
      else:
          movingPH = currentPH
          previousMPH = currentPH 
          previousCPH = currentPH
      PHcounter +=1
      movingPH = movingPH - 40 # Correction for difference between mask edge and ladle edge
     
      savedRoi = roiDisplayArray
      savedMask = maskDisplayArray
      
      # Calculate rotation and display mask
      finalPH, finalRot, previousRot, rotList, movingRot, rotCounter, previousMRot, correctionRot, correctionPos = visualize.display_instances(image, kalmanCheck, currentImage, movingPH, previousRot, rotList, movingRot, rotCounter, previousMRot, lPixel, correctionPos, correctionRot, savedRoi, savedMask, r['class_ids'], 
                                  dataset.class_names, r['scores'], ax=ax,
                                  title="Predictions") 

  # Pouring height and rotation in first 30 frames (before ladle has reached furnace)   
  elif (currentImage <= 30):      
      bestIndices = np.argmax(r['scores'])
      maskArray = r['masks'][:,:,bestIndices]
      q = bestIndices
      roiDisplayArray = r['rois'][q]
      maskDisplayArray = np.reshape(maskArray, (1024,1024,1))
      cv2.imwrite("mrcnnOut%d.png" %(counter-1), maskDisplayArray*255)
      currentPH = np.min(np.where(maskDisplayArray)[0]) # Calculate current pouring height
      lPixel = np.min(np.where(maskDisplayArray)[1])
 
      if previousMPH != 0:
          if previousCPH == 0:
              previousCPH = previousMPH
          if (currentPH - previousMPH > 30) or (previousMPH - currentPH > 30):
              movingPH = previousCPH
              currentPH = previousMPH

          if PHcounter%5 != 0: # If PHcounter is not a multiple of 5
              movingPH = previousCPH
              PHlist.append(currentPH)
          else:
              movingPH = np.mean(PHlist)
              PHlist = []
          previousMPH = currentPH
          previousCPH = movingPH
      else:
          movingPH = currentPH
          previousMPH = currentPH 
          previousCPH = currentPH
      PHcounter +=1
      movingPH = movingPH - 40 # Correction for difference between mask edge and ladle edge
   
      savedRoi = roiDisplayArray
      savedMask = maskDisplayArray
      finalPH, finalRot, previousRot, rotList, movingRot, rotCounter, previousMRot, correctionRot, correctionPos = visualize.display_instances(image, kalmanCheck, currentImage, movingPH, previousRot, rotList, movingRot, rotCounter, previousMRot, lPixel, correctionPos, correctionRot, roiDisplayArray, maskDisplayArray, r['class_ids'], 
                                  dataset.class_names, r['scores'], ax=ax,
                                  title="Predictions") 
  
  else:
        # Pouring height and rotation if multiple masks detected
        maskArray = r['masks'][:,:,0]
        for maskSelection in range(0,r['masks'].shape[2]): # Loop through each mask in frame
           leftPixel = np.min(np.where(r['masks'][:,:,maskSelection])[1])
           rightPixel = np.max(np.where(r['masks'][:,:,maskSelection])[1])
           newLeftValue = leftPixel + rightPixel

           if newLeftValue < leftValue: 
               leftValue = newLeftValue # Choose left-most mask
               maskArray = r['masks'][:,:,maskSelection]
               q = maskSelection
               roiDisplayArray = r['rois'][q]
               maskDisplayArray = np.reshape(maskArray, (1024,1024,1))
               lPixel = np.min(np.where(maskDisplayArray)[1])
               cv2.imwrite("mrcnnOut%d.png" %(counter-1), maskDisplayArray*255)
               currentPH = np.min(np.where(maskDisplayArray)[0]) # Calculate current pouring height
 
               if previousMPH != 0:
                   if previousCPH == 0:
                       previousCPH = previousMPH
                   if (currentPH - previousMPH > 30) or (previousMPH - currentPH > 30):
                       movingPH = previousCPH
                       currentPH = previousMPH

                   if PHcounter%5 != 0: # If PHcounter is not a multiple of 5
                       movingPH = previousCPH
                       PHlist.append(currentPH)
                   else:
                       movingPH = np.mean(PHlist)
                       PHlist = []
                   previousMPH = currentPH
                   previousCPH = movingPH
               else:
                   movingPH = currentPH
                   previousMPH = currentPH 
                   previousCPH = currentPH
               PHcounter +=1
               movingPH = movingPH - 40 # Correction for difference between mask edge and ladle edge
   
               savedRoi = roiDisplayArray
               savedMask = maskDisplayArray
        
        # Calculate rotation and display mask
        finalPH, finalRot, previousRot, rotList, movingRot, rotCounter, previousMRot, correctionRot, correctionPos = visualize.display_instances(image, kalmanCheck, currentImage, movingPH, previousRot, rotList, movingRot, rotCounter, previousMRot, lPixel, correctionPos, correctionRot, roiDisplayArray, maskDisplayArray, r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")  
     
  plt.text(375, 60, "Flame Severity: %d" %(flameSeverity), size=35, rotation=0.,
           ha="center", va="center",
           bbox=dict(boxstyle="round",
                     ec=(0., 0.5, 0.5),
                     fc=(0., 0.8, 0.8),
                     )
           )                

  coords = np.argwhere(maskArray==True)
  centers2 = coords # centers2 is x,y of every pixel in predicted mask
  
  # Convert centers2 to edge
  backgroundImage = np.zeros(np.shape(maskArray))
  
  for pixels in range(0,len(coords)):
      backgroundImage[coords[pixels,0], coords[pixels,1]] = 1

  # Add 1-pixel thick padding on both axis
  backgroundImagePad0 = np.pad(backgroundImage,[(0,1),(0,0)], mode='edge')
  backgroundImagePad1 = np.pad(backgroundImage,[(0,0),(1,0)], mode='edge')
  
  edges = np.where((np.diff(backgroundImagePad0, axis=0) > 0),1,0)
  edges2 = np.where((np.diff(backgroundImagePad1, axis=1) > 0),1,0)
  
  backgroundImage = np.zeros(np.shape(maskArray))
  edgeCoords1 = np.transpose(np.asarray(np.where(edges==1))) 
  edgeCoords2 = np.transpose(np.asarray(np.where(edges2==1)))
  edgeCoords = np.concatenate((edgeCoords1, edgeCoords2), axis=0)
  
  for edgePixels in range(0,len(edgeCoords1)):
      backgroundImage[edgeCoords1[edgePixels,0], edgeCoords1[edgePixels,1]] = 1
  
  edgeCoords4 = []
  backgroundImage2 = np.zeros(np.shape(maskArray))
  
  for pixels2 in range(0,len(coords)):
      backgroundImage2[coords[pixels2,0], coords[pixels2,1]] = 1
      
  # Loop through every pixel and check adjacent pixels
  pCols = backgroundImage2.shape[0]
  pRows = backgroundImage2.shape[1]
  
  for ix in range(pCols):
      for iy in range(pRows):
          currentPixel = backgroundImage2[ix,iy]
          
          if currentPixel == 1.0 and (backgroundImage2[ix+1,iy] == 0.0 or backgroundImage2[ix-1,iy] == 0.0 or backgroundImage2[ix,iy+1] == 0.0 or backgroundImage2[ix,iy-1] == 0.0 
                                      or backgroundImage2[ix+1,iy+1] == 0.0 or backgroundImage2[ix+1,iy-1] == 0.0 or backgroundImage2[ix-1,iy-1] == 0.0 or backgroundImage2[ix-1,iy+1] == 0.0):
              tempXY = [[ix, iy]]
              edgeCoords4.append(tempXY)
              
  edgeCoords4 = np.asarray(edgeCoords4)
  edgeCoords4 = np.reshape(edgeCoords4, (len(edgeCoords4),2))
  
  for edgePixels in range(0,len(edgeCoords4)):
      backgroundImage[edgeCoords4[edgePixels,0], edgeCoords4[edgePixels,1]] = 1

  # Create final centers for predicted mask
  centers3 = np.array([[0,1]])
  
  for pointNum in range(0,len(edgeCoords4)):                
      xc = edgeCoords4[pointNum, 0]
      yc = edgeCoords4[pointNum, 1]
      centers3[pointNum,0] = xc
      centers3[pointNum,1] = yc
      centers3Add = np.array([[xc,yc]])
      centers3 = np.concatenate((centers3, centers3Add), axis=0)
      
      np.savetxt("predicted_centers%d.csv" %(counter-1), centers3, delimiter=",")
      
  # Calculate metrics
  AP, Precisions, Recalls, Overlaps, precisions, recalls, overlaps = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, roiDisplayArray, r['class_ids'], r['scores'], maskDisplayArray, iou_thresholds=None, verbose=1)
  
  # Log results
  resultsTable = resultsTable.append({'File Name': "ladle"}, ignore_index=True)
  resultsTable.loc[currentImage,'Frame Number'] = image_id
  resultsTable.loc[currentImage,'Inference Time'] = (end-start)
  resultsTable.loc[currentImage,'AP'] = AP
  resultsTable.loc[currentImage,'Precision'] = Precisions
  resultsTable.loc[currentImage,'Recall'] = Recalls
  resultsTable.loc[currentImage,'Overlap'] = Overlaps
  resultsTable.loc[currentImage,'final PH'] = finalPH
  resultsTable.loc[currentImage,'final Rotation'] = finalRot
  analysisTable.loc[currentImage,'Frame Number'] = image_id
  analysisTable.loc[currentImage,'Pouring Height'] = finalPH
  analysisTable.loc[currentImage,'Rotation'] = movingRot 
  analysisTable.loc[currentImage,'Bright Pixels'] = brightPixels
  analysisTable.loc[currentImage,'Flame Severity'] = flameSeverity
  
  #print("Inference time: %.5f seconds" %(end - start))
  print("AP: %.5f" %AP)

# Calculate final results
meanResultsTable = pd.DataFrame(columns=['mInference Time','mPrecision','mRecall','mOverlap','mAP'])     
meanResultsTable.loc[0,'mInference Time'] = resultsTable["Inference Time"].mean()
meanResultsTable.loc[0,'mAP'] = resultsTable["AP"].mean()
meanResultsTable.loc[0,'mPrecision'] = resultsTable["Precision"].mean()
meanResultsTable.loc[0,'mRecall'] = resultsTable["Recall"].mean()
meanResultsTable.loc[0,'mOverlap'] = resultsTable["Overlap"].mean()
 
#resultsTable.to_csv(r'C:/Users/Callum/Anaconda3/envs/Mask_RCNN/v2_1_raw.csv', index=False)
#meanResultsTable.to_csv(r'C:/Users/Callum/Anaconda3/envs/Mask_RCNN/v2_1.csv', index=False)
#analysisTable.to_csv(r'C:/Users/Callum/Anaconda3/envs/Mask_RCNN/analysisTable.csv', index=False)

print("FINAL MAP: ", resultsTable["AP"].mean())