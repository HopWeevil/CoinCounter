import cv2
import ctypes
import os
import ImageUtlis as iu
import CoinCounter

if os.name == 'nt':
    # Стаф для коректного розміру вікна на широкоформатних моніторах Windows 7-10 (Dpi)
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(awareness))
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(2)
    success = ctypes.windll.user32.SetProcessDPIAware()

scale = 3
debug = True
svm = cv2.ml.SVM_load('svm_model.dat')
image =cv2.imread('Results/arc.jpg')
warpedimage = iu.WarpImage(image, scale, debug)
imgPre = iu.ImagePre(warpedimage, debug)

coins_sum = CoinCounter.CountCoins(imgPre, warpedimage,svm,1,2,debug)[0]
img = cv2.putText(warpedimage, ("Coins sum: {:.2f}".format(coins_sum)), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

cv2.imshow("Output", img)

cv2.waitKey(0)
cv2.destroyAllWindows()