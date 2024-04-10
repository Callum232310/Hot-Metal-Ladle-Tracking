"""
Script Name: visualize.py

Description: This script contains some key functions for visualising ladle model results.

Original Author: Matterport
Modified by: Callum O'Donovan

Original Creation Date: 2017
Modification Date: April 20th 2021

Email: callumodonovan2310@gmail.com
    
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""


"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from mrcnn import utils
import cv2
import math

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR) 

############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


#def display_instances(image, kalmanCheck, movingPH, lPixel, currentImage, boxes, masks, class_ids, class_names, # for load_video2.py
def display_instances(image, kalmanCheck, currentImage, movingPH, previousRot, rotList, movingRot, rotCounter, previousMRot, lPixel, correctionPos, correctionRot, boxes, masks, class_ids, class_names, #for load_video.py
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    degree_sign = u'\N{DEGREE SIGN}'
    # Number of instances
    N = 1

    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    
    for i in range(N):
        color = colors[i]
        #if boxes.shape == (0): # When executing load_video2.py
        if boxes.shape == (0,4): # When executing load_video.py
            # Skip this instance. Has no bbox. Likely lost in image cropping       
            continue

        y1, x1, y2, x2 = boxes # When executing load_video2.py
        #y1, x1, y2, x2 = boxes[i] # When executing load_video.py
        
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            if boxes.shape == (0):
                continue
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]

        if correctionPos == 0:   
            mask2 = np.zeros((1024, 1024), dtype=np.uint8)      
            fixedMaskPts = np.array([[800,350],[560,440],[465,420],[440,520],[600,895],[895,790]],np.int32)
            fixedMaskPts = fixedMaskPts.reshape((-1, 1, 2))
            mask2 = cv2.fillPoly(mask2, [fixedMaskPts], (255, 255, 255))
            color2 = (255,0,255)
            mask2 = mask2.astype(np.bool)
            
            if show_mask:
                masked_image = apply_mask(masked_image, mask2, color) 

        else:
            if correctionPos[0] < -350:
                correctionPos[0] = correctionPos[0]
            mask2 = np.zeros((1024, 1024), dtype=np.uint8)  
            
            fixedMaskPts2 = np.array([[800,350],[560,440],[465,420],[440,520],[600,895],[895,790]],np.int32)
            
            for currentPts in fixedMaskPts2:
                currentPts[0] = currentPts[0] + correctionPos[0]
                currentPts[1] = currentPts[1] + correctionPos[1]
                
            if correctionRot > 14:
                angle = math.radians((correctionRot-14)*-0.8)  
                center = (x2, y1)
                shape_coords = fixedMaskPts2
                
                # Loop through each coordinate point in the shape
                rotated_coords = []
                
                for coord in shape_coords:
                    # Translate the coordinate point
                    translated_coord = (coord[0] - center[0], coord[1] - center[1])
                    
                    # Apply the rotation formula
                    rotated_x = translated_coord[0] * math.cos(angle) - translated_coord[1] * math.sin(angle)
                    rotated_y = translated_coord[0] * math.sin(angle) + translated_coord[1] * math.cos(angle)
                    
                    # Translate the rotated point back to the original position
                    rotated_coord = [rotated_x + center[0], rotated_y + center[1]]
                    rotated_coords.append(rotated_coord)    
                    
                fixedMaskPts2 = np.array(rotated_coords, dtype=np.int32)
                
            #print(fixedMaskPts2.shape)
            #print(fixedMaskPts2.dtype)
            fixedMaskPts2 = fixedMaskPts2.reshape((-1, 1, 2))
            mask2 = cv2.fillPoly(mask2, [fixedMaskPts2], (255, 255, 255))
            color2 = (255,0,255)
            mask2 = mask2.astype(np.bool)
            
            if show_mask:
                masked_image = apply_mask(masked_image, mask2, color)                        
        
        #if show_mask:
            #masked_image = apply_mask(masked_image, mask, color)


        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        # padded_mask = np.zeros(
        #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # padded_mask[1:-1, 1:-1] = mask
        # contours = find_contours(padded_mask, 0.5)
        # for verts in contours:
        #     # Subtract the padding and flip (y, x) to (x, y)
        #     verts = np.fliplr(verts) - 1
        #     p = Polygon(verts, facecolor="none", edgecolor=color)
        #     ax.add_patch(p)
        
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig('C:/Users/Callum/Anaconda3/envs/Mask_RCNN/Output/prediction.png', bbox_inches='tight') 

    if kalmanCheck == 1:
        finalRot = 0
        # Pouring height
        lineY = [movingPH, movingPH]
        lineX = [0, 1024]
        plt.plot(lineX,lineY, color="blue", linewidth=3)
        plt.savefig('C:/Users/Callum/Anaconda3/envs/Mask_RCNN/mrcnn/prediction2.png', bbox_inches='tight')
        finalPH = 1024-movingPH
        plt.text(800, 30, "Pouring height: %dpx" %(finalPH), size=30, rotation=0.,
          ha="center", va="center",
          bbox=dict(boxstyle="round",
                    ec=(0., 0.5, 0.5),
                    fc=(0., 0.8, 0.8),
                    )
          )        
    elif kalmanCheck == 0:
        
        # Pouring height
        lineY = [movingPH, movingPH]
        lineX = [0, 1024]
        plt.plot(lineX,lineY, color="blue", linewidth=3)
        plt.savefig('C:/Users/Callum/Anaconda3/envs/myenv/Mask_RCNN/mrcnn/prediction2.png', bbox_inches='tight')
        finalPH = 1024-movingPH+50
        plt.text(800, 30, "Pouring height: %dpx" %(finalPH), size=30, rotation=0.,
          ha="center", va="center",
          bbox=dict(boxstyle="round",
                    ec=(0., 0.5, 0.5),
                    fc=(0., 0.8, 0.8),
                    )
          )                
        # Rotation angle
        imgB = cv2.imread("mrcnnOut%d.png" %(currentImage-1)) # Load mrcnn prediction mask
        imgB = imgB[:,:,0]

        ret,thresh = cv2.threshold(imgB,127,255,0)
        referenceMaskX = 795
        referenceMaskY = 610
        maskPts = np.argwhere(thresh)
        
        maskX1 = x1
        maskX2 = x2
        maskY1 = y1
        maskY2 = y2
        
        correctionPos = [(maskX1 + maskX2)/2,(maskY1 + maskY2)/2] # Current mask position
        correctionPos = [correctionPos[0] - referenceMaskX, correctionPos[1] - referenceMaskY]
        
        contours, hierarchy = cv2.findContours(thresh, 1, 2) # Get contours
        cnt = contours[0]

        bottomY = np.min(np.where(imgB)[0]) # Make sample square
        topY = np.max(np.where(imgB)[0])
        rightX = np.max(np.where(imgB)[1])
        rightY = np.max(np.where(imgB)[0])

        maskSampleCenter = [rightY, rightX] # topY is lowest Y value, rightX is highest X value            
        sY = int(maskSampleCenter[0])
        sX = int(maskSampleCenter[1])

        # Moving from initial for final positioning based on previous rotation
        boxLeft = int(sY-50 - (35*((previousRot/18)-0.5))) 
        boxRight = int(sY+100 - (45*((previousRot/18)-0.5)))
        boxBottom = int(sX-10 + (36*((previousRot/18)-0.5)))
        boxTop = int(sX-170 + (30*((previousRot/18)-0.5)))

        imgB = imgB.copy()
        
        maskSampleSquare = imgB[boxLeft:boxRight, boxTop:boxBottom]
        
        imgE = imgB.copy()
        
        sampleEdgeX = []
        sampleEdgeY = []

        for pY in range(0,maskSampleSquare.shape[0]-1): # Get edge from sample
            for pX in range(0,maskSampleSquare.shape[1]-1):

                if maskSampleSquare[pY,pX] == 255 and (maskSampleSquare[pY+1,pX] == 0 or maskSampleSquare[pY,pX+1] == 0):
                    sampleEdgeX.append(pX)
                    sampleEdgeY.append(pY)
                    
        sampleEdgeX = np.asarray(sampleEdgeX)
        sampleEdgeY = np.asarray(sampleEdgeY)
        
        # Draw line of best fit from edge
        m, b = np.polyfit(sampleEdgeX, sampleEdgeY, 1)
        plt.plot(sampleEdgeX, sampleEdgeY, 'o') 
        lobf = plt.plot(sampleEdgeX, m*sampleEdgeX + b) 
        lor = plt.axvline(x=30)
  
        angleInRadians = math.atan(m)
        angleInDegrees = math.degrees(angleInRadians)
        finalRot = (angleInDegrees*-1)
        
        if (finalRot < 0) or (85 < finalRot < 89.9):
            finalRot = 90
            
        previousRot = finalRot
        
        currentRot = finalRot

        if rotCounter%4 != 0: # If rotCounter is not a multiple of 4
            movingRot = previousMRot #
            rotList.append(currentRot)

        else:
            rotList.append(currentRot)
            
            if rotList == []:
                movingRot = 0
            else:
                movingRot = np.mean(rotList)

                rotList = []
        previousRot = currentRot
        previousMRot = movingRot
        rotCounter +=1  

        if lPixel > 300: # If ladle is not in position to pour
            finalRot2 = 14
            correctionRot = finalRot2
            plt.text(800, 90, "Rotation: %d%s" %(finalRot2, degree_sign), size=30, rotation=0.,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(0., 0.5, 0.5),
                               fc=(0., 0.8, 0.8),
                               )
                     )   
        else:
            correctionRot = movingRot
            plt.text(800, 90, "Rotation: %d%s" %(movingRot, degree_sign), size=30, rotation=0.,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(0., 0.5, 0.5),
                               fc=(0., 0.8, 0.8),
                               )
                     )                

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            imgB = imgB.copy()
            imgC = np.zeros((1024,1024))
            
            cv2.drawContours(imgC,[box],0,(255,0,0),3)

            imgIndices = np.where(imgC) # Get indices of all white pixels
    
            leftY = np.max(imgIndices[1]) # Find lowest x value
            leftIndex = np.argmax(imgIndices[1]) # Find index of lowest x value
            leftX = imgIndices[0][leftIndex] # Find corresponding y value using index
            
            rightY = np.min(imgIndices[1])
            rightIndex = np.argmin(imgIndices[1])
            rightX = imgIndices[0][rightIndex]
    
            topX = np.min(imgIndices[0])
            topIndex = np.argmin(imgIndices[0])
            topY = imgIndices[1][topIndex]
            
            bottomX = np.max(imgIndices[0])
            bottomIndex = np.argmax(imgIndices[0])
            bottomY = imgIndices[1][bottomIndex]

            imgD = np.zeros((1024,1024))
            imgC[leftX, leftY] = 125
            imgD[rightX, rightY] = 255
            imgD[topX, topY] = 255
            imgD[bottomX, bottomY] = 255

            midTop = int((topX-leftX)/2), int((topY-leftY)/2)
            midBottom  = int((rightX+bottomX)/2), int((rightY+bottomY)/2)

            imgC = cv2.line(imgC, midTop, midBottom, (255,0,0),3)

    color2 = (255, 0, 255)

    mask2 = np.zeros((1024, 1024), dtype=np.uint8)

    fixedMaskPts = np.array([[884,990],[760, 680],[520, 771],[382, 481]],np.int32)
    
    fixedMaskPtsCorrected = np.array([[418, 374],[884,990],[680,335],[693,390],[760, 680],[520, 771],[382, 481], [375,400]],
              np.int32)
    
    fixedMaskPts = fixedMaskPts.reshape((-1, 1, 2))
    
    mask2 = cv2.fillPoly(mask2, [fixedMaskPts], (255, 255, 255))
    masked_image = apply_mask(masked_image, mask2, color)
    ax.imshow(masked_image.astype(np.uint8))
    
    if auto_show:
        plt.show()
        
    return finalPH, finalRot, previousRot, rotList, movingRot, rotCounter, previousMRot, correctionRot, correctionPos



def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)
    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)

