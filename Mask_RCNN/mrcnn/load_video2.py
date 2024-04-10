"""
Script Name: load_video2.py

Description: This script is for performing ladle tracking inference using Mask R-CNN and Kalman filter results.

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
from visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import ladle
import cv2
import utils
import time
import pandas as pd
import csv
import decimal

ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_WEIGHTS_PATH = ROOT_DIR +"/mask_rcnn_coco.h5"
config = ladle.CustomConfig()
LADLE_DIR = ROOT_DIR+"\Mask_RCNN\dataset"

# Override the training configurations with a few changes for inferencing.
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
    
weights_path = "C:/Users/Callum/Anaconda3/envs/Mask_RCNN/logs/Experiments2/v2_4.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

framePath = "C:/Users/Callum/Anaconda3/envs/Mask_RCNN/dataset/val"
frameCount = len(os.listdir(framePath))
frameCount = frameCount-1 # Count number of files (frames) in "val" folder -1 for .json
  
resultsTable2 = pd.DataFrame(columns=['Frame Number','Inference Time','Precision','Recall','Overlap','AP'])

roi_values = np.zeros([1,4])
trackingValues = np.zeros([10, 1])
counter = 0
previousPH = 0
kalmanCheck = 1

for currentImage in range(1,frameCount+1):
  print(currentImage)
  image_id = dataset.image_ids[currentImage-1] 
  image, image_meta, gt_class_id, gt_bbox, gt_mask, centers =\
      modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False) ##########################
  print("gt_mask SHAPE: ", gt_mask.shape)
  info = dataset.image_info[image_id]
  #print(gt_mask)
  #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                     #  dataset.image_reference(image_id)))
  counter += 1

# Run object detection
  start = time.time()
  results = model.detect([image], verbose=0)
  end = time.time()

# Display results
  ax = get_ax(1)
  r = results[0]
  
  kalmanMask2 = cv2.imread("kalmanOut%d.png" %(currentImage-1))
  kalmanMask2 = np.array(kalmanMask2, dtype=bool)
  kalmanMask2 = kalmanMask2[:,:,0]
  kalmanMask2 = np.reshape(kalmanMask2, (1024,1024,1))

  currentPH = np.min(np.where(kalmanMask2)[0]) # Calculate current pouring height
  
  if previousPH != 0:
      movingPH = (previousPH + currentPH) / 2 
  else:
      movingPH = currentPH
  previousPH = currentPH

  lPixel = np.min(np.where(kalmanMask2)[1])
  kalmanCheck = 1
  # Display mask and calculate metrics (no detections)
  # Select furthest box left
  if r['rois'].shape == (0,4):
      print("no boxes")
      kalmanMask2 = cv2.imread("kalmanOut%d.png" %(currentImage-2))
      kalmanMask2 = np.array(kalmanMask2, dtype=bool)
      kalmanMask2 = kalmanMask2[:,:,0]
      kalmanMask2 = np.reshape(kalmanMask2, (1024,1024,1))
      
      visualize.display_instances(image, kalmanCheck, movingPH, lPixel, currentImage, kalmanBox, kalmanMask2, savedClassids, 
                        dataset.class_names, savedScores, ax=ax,
                        title="Predictions")
      
      AP2, Precisions2, Recalls2, Overlaps2, precisions2, recalls2, overlaps2 = utils.compute_ap_range(savedGTBOX, savedGTCLASSID, savedGTMASK, savedKBOX, savedClassids, savedScores, savedKMASK, iou_thresholds=None, verbose=0)
  #Display mask and calculate metrics (one detection)
  elif r['rois'].shape == (1,4):
      kalmanBox = r['rois'][0]
      visualize.display_instances(image, kalmanCheck, movingPH, lPixel, currentImage, boxes=kalmanBox, masks=kalmanMask2, class_ids=r['class_ids'], 
                             class_names=dataset.class_names, scores=r['scores'], ax=ax,
                             title="Predictions")
      
      savedGTBOX = gt_bbox
      savedGTCLASSID = gt_class_id
      savedGTMASK = gt_mask
      savedKBOX = kalmanBox
      savedKMASK = kalmanMask2
      savedRois = r['rois']
      savedClassids = r['class_ids']
      savedScores = r['scores']
      
      AP2, Precisions2, Recalls2, Overlaps2, precisions2, recalls2, overlaps2 = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, kalmanBox, r['class_ids'], r['scores'], kalmanMask2, iou_thresholds=None, verbose=0)

  else:
      #Display mask and calculate metrics (multiple detections)
      kalmanBox = r['rois'][0]
      boxLeftness = 1000000
      
      for box in range(0,r['rois'].shape[0]):
          currentBox = r['rois'][box]
          currentBoxLeftness = currentBox[1]+ currentBox[3]
          
          if currentBoxLeftness < boxLeftness:
             kalmanBox = r['rois'][box]
             boxLeftness = currentBoxLeftness
             
      visualize.display_instances(image, kalmanCheck, movingPH, lPixel, currentImage, kalmanBox, kalmanMask2, r['class_ids'], 
                        dataset.class_names, r['scores'], ax=ax,
                        title="Predictions")  
      
      savedGTBOX = gt_bbox
      savedGTCLASSID = gt_class_id
      savedGTMASK = gt_mask
      savedKBOX = kalmanBox
      savedKMASK = kalmanMask2
      savedRois = r['rois']
      savedClassids = r['class_ids']
      savedScores = r['scores']
      
      AP2, Precisions2, Recalls2, Overlaps2, precisions2, recalls2, overlaps2 = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask, kalmanBox, r['class_ids'], r['scores'], kalmanMask2, iou_thresholds=None, verbose=0)
  #Log results
  resultsTable2 = resultsTable2.append({'File Name': "ladle"}, ignore_index=True)
  resultsTable2.loc[currentImage,'Frame Number'] = image_id
  resultsTable2.loc[currentImage,'Inference Time'] = (end-start)
  resultsTable2.loc[currentImage,'AP'] = AP2
  resultsTable2.loc[currentImage,'Precision'] = Precisions2
  resultsTable2.loc[currentImage,'Recall'] = Recalls2
  resultsTable2.loc[currentImage,'Overlap'] = Overlaps2
 
  print("Inference time: %.5f seconds" %(end - start))
  print("AP: %.5f" %AP2)
  
  print(resultsTable2.loc[[currentImage]])  
  
#Calculate final results
meanResultsTable2 = pd.DataFrame(columns=['mInference Time','mPrecision','mRecall','mOverlap','mAP'])     
meanResultsTable2.loc[0,'mInference Time'] = resultsTable2["Inference Time"].mean()
meanResultsTable2.loc[0,'mAP'] = resultsTable2["AP"].mean()
meanResultsTable2.loc[0,'mPrecision'] = resultsTable2["Precision"].mean()
meanResultsTable2.loc[0,'mRecall'] = resultsTable2["Recall"].mean()
meanResultsTable2.loc[0,'mOverlap'] = resultsTable2["Overlap"].mean()

#resultsTable2.to_csv(r'C:/Users/Callum/Anaconda3/envs/Mask_RCNN/v2_4.csv', index=False)
#meanResultsTable2.to_csv(r'C:/Users/Callum/Anaconda3/envs/Mask_RCNN/v2_4.csv', index=False)


print("FINAL MAP: ", resultsTable2["AP"].mean())