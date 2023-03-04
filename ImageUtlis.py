import cv2
import numpy as np

def WarpImage(img, scale, debug):
   
    width = 210 * scale
    height = 297 * scale

    image = img
    #image = cv2.resize(image,(480,640))
    original=image.copy()

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  

    blurred=cv2.medianBlur(gray,7)
    
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours,hierarchy=cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
    contours=sorted(contours,key=cv2.contourArea,reverse=True)

    for c in contours:
        p=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*p,True)

        if len(approx)==4:
            target=approx
            break

    for corner in target:
        x,y = corner.ravel()
        cv2.circle(image,(x,y),10,(0,0,255),-1)

    approx=ReorderPoints(target) 

    points=np.float32([[0,0],[width,0],[width,height],[0,height]])

    order_points=cv2.getPerspectiveTransform(approx,points)  
    output=cv2.warpPerspective(original,order_points,(width ,height))

    cv2.drawContours(image,[target],-1,(255,0,0),2)

    if debug:
        cv2.imwrite("Debug/WarpGray.jpg",gray)
        cv2.imwrite("Debug/WarpBlur.jpg",blurred)
        cv2.imwrite("Debug/WarpThreshold.jpg",threshold)
        cv2.imwrite("Debug/WarpCornersAndOutline.jpg",image)
        cv2.imwrite("Debug/WarpOriginal.jpg",original)
        cv2.imwrite("Debug/WarpFinal.jpg",output)

    return output

def ReorderPoints(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def ImagePre(img,debug):  

    shifted = cv2.pyrMeanShiftFiltering(img, 20, 30)

    blurred = cv2.medianBlur(shifted,1)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    ddept=cv2.CV_16S
    x = cv2.Sobel(gray, ddept, 1,0, ksize=1, scale=0.1)
    y = cv2.Sobel(gray, ddept, 0,1, ksize=1, scale=0.1)
    absx= cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absx, 10, absy, 10,0)

    if debug:
        cv2.imwrite("Debug/Shifted.jpg",shifted)
        cv2.imwrite("Debug/BlurPre.jpg",blurred)
        cv2.imwrite("Debug/GrayPre.jpg",gray)
        cv2.imwrite("Debug/SobelPre.jpg",sobel)

    return sobel
