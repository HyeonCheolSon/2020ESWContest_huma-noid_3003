import cv2
import numpy as np

def nothing(x):
	# any operation
	pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",0,180,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",180,180,nothing)
cv2.createTrackbar("U-S","Trackbars",236,255,nothing)
cv2.createTrackbar("U-V","Trackbars",152,255,nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    _,frame = cap.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
  
    l_h = cv2.getTrackbarPos("L-H","Trackbars")
    l_s = cv2.getTrackbarPos("L-S","Trackbars")
    l_v = cv2.getTrackbarPos("L-V","Trackbars")
    u_h = cv2.getTrackbarPos("U-H","Trackbars")
    u_s = cv2.getTrackbarPos("U-S","Trackbars")
    u_v = cv2.getTrackbarPos("U-V","Trackbars")

    #위에 트랙바 쓸경우
    lower_black = np.array([l_h, l_s, l_v])
    upper_black = np.array([u_h, u_s, u_v])

    # 트랙바 안쓸 경우
    #lower_black = np.array([0, 0, 0])
    #upper_black = np.array([180, 235, 150])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel)

    contours,_ = cv2.findContours(mask , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt,True),True)


        
        # 실제로봇에서 한번 조절필요 
        if area > 1800 :

            points = []
	        
            if len(approx)==7:

                for i in range(7):
                   points.append([approx.ravel()[2*i], approx.ravel()[2*i+1]])

                points.sort()
               
                minimum = points[1][0] - points[0][0]
                maximum = points[6][0] - points[5][0]

                if maximum < minimum :
                	cv2.putText(frame, "left", (points[0][0], points[0][1]), font,1,(0,0,0))
                else:
                	cv2.putText(frame, "right", (points[6][0], points[6][1]), font,1,(0,0,0))
                
                cv2.drawContours(frame,[approx],0,(0,0,0),5)
                
            
              
	    
    cv2.imshow("Frame",frame)
    cv2.imshow("MASK",mask)

    key = cv2.waitKey(1)
    if key ==27:
        break

cap.release()
cv2.destroyAllWindows()		
