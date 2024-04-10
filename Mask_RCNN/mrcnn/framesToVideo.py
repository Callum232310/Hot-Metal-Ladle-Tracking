"""
Script Name: framesToVideo.py

Description: This script is for converting individual frames to video format.

Original Author: Callum O'Donovan

Original Creation Date: April 20th 2021

Email: callumodonovan2310@gmail.com
    
Disclaimer: This script is part of a project focusing on practical application in engineering.
            For full code quality and contribution guidelines, see the README file. 
            
"""

import re
import os
import moviepy.video.io.ImageSequenceClip

def sorted_alphanumeric(data): # Sort image files in alphanumeric order
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


image_folder= r'C:\Users\Callum\Anaconda3\envs\Mask_RCNN\Output\flameSeverity4'
fps=3

image_files = [image_folder+'/'+img for img in sorted_alphanumeric(os.listdir(image_folder)) if img.endswith(".png")]
print(image_files)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('flameSeverity4.mp4')

