"""
Script Name: objTrackingMulti.py

Description: This script is for performing ladle mask point tracking using predicted masks.

Original Author: Rahmad Sadlii
Modified by: Callum O'Donovan

Original Creation Date: February 20th 2020
Modification Date: April 20th 2021

Email: callumodonovan2310@gmail.com
    
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import cv2
from KalmanFilter import KalmanFilter
import os,sys
import csv
import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors
from skimage.draw import polygon2mask
import PIL.ImageDraw as ImageDraw
from python_tsp.distances import great_circle_distance_matrix
from python_tsp.heuristics import solve_tsp_simulated_annealing
import time
import math
local_vars = {}
np.set_printoptions(threshold=sys.maxsize)


def match_resizing(image, min_dim=800, max_dim=1024, min_scale=0):

     # Keep track of image dtype and return results in the same dtype
     image_dtype = image.dtype
     # Default window (y1, x1, y2, x2) and default scale == 1.
     h, w = image.shape[:2]
     window = (0, 0, h, w)
     scale = 1
     padding = [(0, 0), (0, 0), (0, 0)]
     crop = None
    
     if min_dim:
         # Scale up but not down
         scale = max(1, min_dim / min(h, w))
         
     if min_scale and scale < min_scale:
         scale = min_scale
 
     if max_dim:
         image_max = max(h, w)
         if round(image_max * scale) > max_dim:
             scale = max_dim / image_max
 
     # Resize image using bilinear interpolation
     if scale != 1:
         image = resize(image, (round(h * scale), round(w * scale)),
                        preserve_range=True)
 
     # Get new height and width
     h, w = image.shape[:2]
     top_pad = (max_dim - h) // 2
     bottom_pad = max_dim - h - top_pad
     left_pad = (max_dim - w) // 2
     right_pad = max_dim - w - left_pad
     padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
     image = np.pad(image, padding, mode='constant', constant_values=0)
     window = (top_pad, left_pad, h + top_pad, w + left_pad)
     
     return image.astype(image_dtype), window, scale, padding

def match_resizing_pad64(image, min_dim=800, max_dim=1024, min_scale=0): #Use for pad64

    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
        
    if min_scale and scale < min_scale:
        scale = min_scale

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                        preserve_range=True)


    h, w = image.shape[:2]
    
    # Both sides must be divisible by 64
    min_dim = 832
    assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
    
    # Height
    if h % 64 > 0:
        max_h = h - (h % 64) + 64
        top_pad = (max_h - h) // 2
        bottom_pad = max_h - h - top_pad
    else:
        top_pad = bottom_pad = 0
        
    # Width
    if w % 64 > 0:
        max_w = w - (w % 64) + 64
        left_pad = (max_w - w) // 2
        right_pad = max_w - w - left_pad
    else:
        left_pad = right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

    return image.astype(image_dtype), window, scale, padding



def main():
 
    pointSample = 50
    neighbourSample = 10000
    
    # Create KalmanFilter object KF
    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
    counter = 0
    
    # List files in val folder and count them
    path = r"C:\Users\Callum\Anaconda3\envs\Mask_RCNN\dataset\val"
    dirs = os.listdir(path)
    filecount = len(dirs)
    #print("FILECOUNT: ", filecount)
    print(os.path.isfile('Pour160 (1).jpg')) # Check if files in mrcnn folder

    for file in dirs:
        start = time.time()

        frame = cv2.imread(r"C:/Users/Callum\Anaconda3/envs/Mask_RCNN/dataset/valkalman/Pour160 (%d).jpg" %(counter+1))          
        frame, window, scale, padding =  match_resizing(frame, 800, 1024, 0)  
        frameFile = Image.open(r"C:/Users/Callum\Anaconda3/envs/Mask_RCNN/dataset/valkalman/Pour160 (%d).jpg" %(counter+1))
        frame = np.array(frame)
                                                                                         
        loc = (r"C:\Users\Callum\Anaconda3\envs\Mask_RCNN\mrcnn\predicted_centers%d.csv" %(counter))
        
        if not loc:
            continue

        # Inputting x,y of measured point
        print(counter)
        
        rowCount = 0
        mycsv = csv.reader(open(loc))
        totalRows = sum(1 for row in mycsv)
        wbx = [0]*totalRows
        wby = [0]*totalRows
        xv = [0]*totalRows
        yv = [0]*totalRows
        x1 = [0]*totalRows
        y1 = [0]*totalRows

        mycsv = csv.reader(open(loc))
        
        for row in mycsv:
            point = row
            wby[rowCount] = float(point[0])  
            wbx[rowCount] = float(point[1])
            rowCount += 1

        centers = []
        centers = [[wby],[wbx]]
        
        # If centroids are detected then track them
        if (len(centers) > 0):

            pointCount = 0
            
            # For evenly distant points
            pointStop = len(centers[0][0])
            pointStep = int(pointStop/10)

            nnCount = 0
            pCount = 0

            # Initially 0 to pointStop (centers length, centers is every point)
            # To sample: pointStop/50 is step size, 
            for points in range(0,pointStop, int(pointStop/pointSample)):
                pCount +=1

                # Draw the detected circle
                cv2.circle(frame, (int(float(centers[1][0][points])), int(float(centers[0][0][points]))), 3, (0, 191, 255), 2)
    
                # Predict
                (x, y) = KF.predict()

                xv[pointCount] = int(x.item((0)))
                yv[pointCount] = int(y.item((-1)))

                xv[pointCount] = float(xv[pointCount])
                yv[pointCount] = float(yv[pointCount])
                
                # Draw a rectangle as the predicted object position
                #cv2.rectangle(frame, (int(xv[pointCount]) - 15, int(yv[pointCount]) - 15), (int(xv[pointCount]) + 15, int(yv[pointCount]) + 15), (255, 0, 0), 2)
    
                # Update
                updateCenters = [[int(float(centers[1][0][points])), int(float(centers[0][0][points]))]]

                (x1[pointCount], y1[pointCount]) = KF.update(updateCenters[0])#*0.3
                x1[pointCount] = x1[pointCount][0,0]
                y1[pointCount] = y1[pointCount][0,1]
                x1[pointCount] = float(x1[pointCount])
                y1[pointCount] = float(y1[pointCount])
                
                cv2.rectangle(frame, (int(x1[pointCount]) - 3, int(y1[pointCount]) - 3), (int(x1[pointCount]) + 3, int(y1[pointCount]) + 3), (0, 0, 255), 2) 

                mx = int(float(centers[1][0][points]))
                my = int(float(centers[0][0][points]))
                mm = []
                mm.append(np.array([[mx], [my]]))
            
                pointCount +=1

            pointCount2 = 0
            
            x1a = np.asarray(x1)
            y1a = np.asarray(y1)

            x1a = np.reshape(x1a, (len(x1a), 1))
            y1a = np.reshape(y1a, (len(y1a), 1))

            pointArray = np.concatenate((x1a,y1a), axis=1)

            neigh = NearestNeighbors(n_neighbors=2)
            neigh.fit(pointArray)

            img = frame
            
            image_shape = (1024,1024)

            xx = np.array(centers[0][0])
            yy = np.array(centers[1][0])
            xx = np.reshape(xx, (len(xx),1))
            yy = np.reshape(yy, (len(yy),1))
            zz = np.concatenate((xx,yy), axis=1)
            
            pts = np.asarray(centers)

            isClosed = True
            color = (255, 0, 0)
            thickness = 1
            image = np.int32(frame)
            
            x1 = [i for i in x1 if i != 0]
            y1 = [i for i in y1 if i != 0]
            xxxxx1 = []
            yyyyy1 = []
            
            for pointsRemoved in range(0,pointStop, int(pointStop/pointSample)):
                
               # Use KNN to filter out Kalman point predictions
                neigh_dist, neigh_ind = neigh.kneighbors([pointArray[pointCount2]])
                
                neigh_dist_x = neigh_dist[[0]][0][0]
                neigh_dist_y = neigh_dist[[0]][0][1]

                if neigh_dist_y > neighbourSample or neigh_dist_x > neighbourSample:
                    cv2.rectangle(frame, (int(x1[pointCount2]) - 3, int(y1[pointCount2]) - 3), (int(x1[pointCount2]) + 3, int(y1[pointCount2]) + 3), (0, 255, 0), 2)
                else:
                    xxxxx1.append(int(x1[pointCount2]))
                    yyyyy1.append(int(y1[pointCount2]))

                pointCount2 +=1      

        xy = []
        xx1 = xxxxx1
        yy1 = yyyyy1
        xx1 = [i for i in xx1 if i != 0]
        yy1 = [i for i in yy1 if i != 0]
        xy2 = []
        
        for xyPoints in range(0,len(xx1)):
            xy.append(xx1[xyPoints])
            xy.append(yy1[xyPoints])
            
        xx = xx.astype(np.int64)
        yy = yy.astype(np.int64)
        xx1 = np.array(xx1)
        xx1 = np.reshape(xx1, (len(xx1),1))
        yy1 = np.array(yy1)
        yy1 = np.reshape(yy1, (len(yy1),1))
        xy1 = np.concatenate((xx1, yy1), axis=1)

        img = Image.fromarray(img)
        img2 = img.copy()

        draw = ImageDraw.Draw(img2)
        
        distance_matrix = great_circle_distance_matrix(xy1) 

        permutation, distance = solve_tsp_simulated_annealing(distance_matrix)

        xxx1 = [xx1[i] for i in permutation]
        yyy1 = [yy1[i] for i in permutation]
        
        for xyPoints in range(0,len(xxx1)):
            xy2.append(xxx1[xyPoints])
            xy2.append(yyy1[xyPoints])

        # Create center point
        centerX = np.mean(xxx1)
        centerY = np.mean(yyy1)
        centerPoint = [centerX, centerY]
        
        # Calculate angle for every point
        pointAngles = []
        
        for anglePoints in range(0,len(xxx1)):
            pointAngle = math.atan2(yyy1[anglePoints]-centerY, xxx1[anglePoints]-centerX)
            pointAngles.append(pointAngle)

        oldIndicesList = []
        newIndicesList = []

        for pointIndex in range(0, len(pointAngles)):
            oldIndicesList.append(pointAngles.index(pointAngles[pointIndex]))
        pointAngles2 = sorted(pointAngles) # Change order of pointAngles
        
        for pointIndex in range(0, len(pointAngles)):
            newIndicesList.append(pointAngles2.index(pointAngles[pointIndex]))
                
        xxxx1 = [xxx1[i] for i in newIndicesList] # Order xxx1 in order of pointAngles
        yyyy1 = [yyy1[i] for i in newIndicesList] # Order yyy1 in order of pointAngles

        # Smoothing by interpolation
        from scipy.interpolate import interp1d

        t = np.arange(len(xxxx1))
        ti = np.linspace(0, t.max(), 10 * t.size)

        xxxxi = np.array(xxxx1)
        xxxxi = np.reshape(xxxxi, len(xxxxi))
        yyyyi = np.array(yyyy1)
        yyyyi = np.reshape(yyyyi, len(yyyyi))

        xxxxi = interp1d(t, xxxxi, kind='cubic', axis=0)(ti)
        yyyyi = interp1d(t, yyyyi, kind='cubic', axis=0)(ti)
        
        xxxx1 = xxxxi
        yyyy1 = yyyyi

        #XXXX1
        zipped_lists = zip(newIndicesList, xxx1) 
        sorted_zipped_lists = sorted(zipped_lists)
        sorted_list1 = [element for _, element in sorted_zipped_lists]

        xxxx1 = sorted_list1
        
        #YYYY1
        zipped_lists2 = zip(newIndicesList, yyy1) 
        sorted_zipped_lists2 = sorted(zipped_lists2)
        sorted_list2 = [element for _, element in sorted_zipped_lists2]
     
        yyyy1 = sorted_list2

        polygon2 = draw.polygon(tuple(zip(xxxx1, yyyy1)), fill = (255,0,0)) 
        img3 = Image.blend(img, img2, 0.5)
        
        pts = np.array(xy, np.int32)
        pts = pts.reshape((-1,1,2))
        
        img4 = np.asarray(img3)

        cv2.imwrite('out%d.png' %counter, img4)

        cv2.destroyAllWindows()
        
        # Get indices of every kalman mask pixel and save in kalmanIndices
        # For every cell of image array, if value == value of mask colour, save indices to kalmanIndices
        # For every cell of kalmanMask, if indices match indices in kalmanIndices, set value to 1 (black/white)
        kalmanIndices = []
        kalmanMask = np.zeros((1024,1024,1))        
        kalmanIndices = np.where(img4 == 255)

        # Create new empty array for coordinates in correct format for polygon2mask
        xyxy1 = np.zeros((len(xxxx1),2))
        
        for xyxy in range(0,len(xxxx1)):
            xyxy1[xyxy,1] = xxxx1[xyxy]
            xyxy1[xyxy,0] = yyyy1[xyxy]

        # Use image shape and coordinates to create mask based on polygon
        kalmanMask2 = polygon2mask(img4.shape, xyxy1)
        kalmanMask2 = np.array(kalmanMask2, np.int32)

        kalmanMask2[kalmanMask2 > 0] = 255
        kalmanMask2 = kalmanMask2[:,:,1]
        print(kalmanMask2.shape)


        cv2.imwrite("kalmanOut%d.png" %(counter), kalmanMask2)
        kalmanMask2 = cv2.imread('kalmanOut%d.png'%(counter), cv2.IMREAD_GRAYSCALE)
        contours, hierarchy = cv2.findContours(kalmanMask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        # Create hull array for convex hull points
        hull = []
        
        # Calculate points for each contour
        for i in range(len(contours)):
            # Creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        drawing = np.zeros((kalmanMask2.shape[0], kalmanMask2.shape[1], 3), np.uint8)
        
        # Draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # Green - color for contours
            color = (255, 0, 0) # Blue - color for convex hull
            # Draw ith contour
            cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # Draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color=(255, 255, 255), thickness=cv2.FILLED)#, 1, 8)

        #Fill pixels within hull with white 
        cv2.imwrite("kalmanOut%d.png" %counter, drawing)
        
        counter += 1
        end = time.time()
        timed = end-start
        print(timed)

       
        
if __name__ == "__main__":
    kalmanMask2 = main()
