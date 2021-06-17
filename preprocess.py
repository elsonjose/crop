import cv2 as cv
import numpy as np
import sys
import math
from scipy import ndimage
import os

def houghtransform(test_img):
    rotated_img = test_img
    try:
        low_range = np.array([18,94,140])                               #light yellow
        high_range=np.array([255,255,255])                              #pure white
        mask = cv.inRange(src=test_img,lowerb=low_range,upperb=high_range)
        edges = cv.Canny(image=mask,threshold1=50,threshold2=100)
        lines = cv.HoughLinesP(image=edges,rho=1,theta=np.pi/180,threshold=30,maxLineGap=300)
        slopes=[]
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line[0]
                # print(x1,y1,x2,y2)                                        
                slope=0
                try:
                    # in opencv the cordinate system is a bit different 
                    # top left is (0,0) and bottom right is (width,height)
                    # hence the slope has to be calculated accordingly
                    slope=((int(y1)-int(y2))/(int(x2)-int(x1)));
                except ZeroDivisionError:
                    # vertical line
                    # slope = infinity
                    slope=float('inf')
                if(slope>=-1 and slope<=1):
                    # horizontal line
                    # find average of slope 
                    # print(slope)                                          
                    slopes.append(slope)

            average_slope=np.average(slopes)  
            angle = math.degrees(average_slope)
            # print("average=",average_slope)                              
            # print("angle=",angle)                                         
            rotated_img = ndimage.rotate(input=test_img, angle=-angle)
    except Exception as e:
        print("Houghs transform failed")
    return rotated_img 


# In[ ]:


# dir_path = r"C:\Users\elson\Desktop\Main project\CR\dataset"
# images=[]
# for file in os.listdir(dir_path):
#     test_img_path = dir_path+"\\"+file
#     print(test_img_path)
#     test_img = cv.imread(filename=test_img_path)
#     test_img = cv2.resize(test_img, (240, 80))
#     cv.imshow('Originial image',test_img)
#     rotated_img =  houghtransform(test_img)
#     cv.imshow('Rotated image',rotated_img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()


# In[ ]:






































