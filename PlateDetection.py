import cv2
import numpy as np
import easyocr
import os
import csv
import time
from datetime import datetime

# Custom filters as provided
def color_filter(img, b):
"""
Custom filter that preserves more information from dark areas
"""
result = img.copy()
for i in range(result.shape[0]):
for j in range(result.shape[1]):
if result[i, j] > b:
result[i, j] = 255
else:
result[i, j] = np.interp(result[i, j], [0, b], [0, 200])
return result

def black_filter(img, b):
"""
Binary filter that converts the image to pure black and white
"""
result = img.copy()
for i in range(result.shape[0]):
for j in range(result.shape[1]):
if result[i, j] > b:
result[i, j] = 255
else:
result[i, j] = 0
return result


