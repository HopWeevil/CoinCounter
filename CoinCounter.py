import cv2
import random
import colorsys
import numpy as np

def AddCoin(coin):
    if(coin != 1):
        return coin/100
    else:
       return coin

def CountCoins(imgPre, img, svm, fontScale, fontThickness, debug):

    records = []
    coins_sum = 0
    circles = cv2.HoughCircles(imgPre, cv2.HOUGH_GRADIENT, 1, 20, param1=200, param2=30, minRadius=20, maxRadius=42)

    for detected_circle in circles[0]:
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        color = (r,g,b)
        x_coor, y_coor, detected_radius = detected_circle
        org = (int(x_coor), int(y_coor))

        cv2.circle(img,org,int(detected_radius),color,1)
        #центр кола
        #cv2.circle(img, org, 2, (0, 0, 255), 3)
        x1, y1 = org[0]-10, org[1]-10
        x2, y2 = org[0]+10, org[1]+10
        #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        square_pixels = img[y1:y2, x1:x2]
        # Compute the mean RGB values of the pixels in the square
        avg_color_per_row = np.average(square_pixels, axis=0)
        b, g, r = np.average(avg_color_per_row, axis=0)
        b, g, r = np.round([b, g, r], 1)
        # площа кругу
        area = round(3.14159 * detected_circle[2] * detected_circle[2])
        
        #coin = round(svm.predict(np.array([[area]], dtype=np.float32))[1][0][0])    
        
        coin_data = np.array([area, b, g, r]).astype(np.float32)
        coin = int(svm.predict(coin_data.reshape(1, -1))[1])

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if debug:
            img = cv2.putText(img, str(area), org, font, fontScale, color, fontThickness, cv2.LINE_AA)
        else:
            img = cv2.putText(img, str(coin), org, font, fontScale, color, fontThickness, cv2.LINE_AA)

        coins_sum = coins_sum + AddCoin(coin)
        records.append([coin, area, b, g, r])  

    if debug:
        cv2.imwrite("Debug/Coins.jpg",img)
        
    return coins_sum, records

