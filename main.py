#Simple Webcam Face Blur Using OpenCV
import cv2


webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)#0 is the inbuilt webcam
frame = webcam.read() #capture webcam
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#get face data from pre-trained model provided by opencv

while True:
   successful_frame_read, frame = webcam.read()

   count=1
   bw_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #bg convert
   coordinates = trained_data.detectMultiScale(bw_img, minNeighbors= 9) #select minimum neighbours and get face coordinates

   for (x,y,w,h) in coordinates: #main loop

        region_of_interest = frame[y:(y+h), x:x+w] #mark face

        region_of_interest = cv2.GaussianBlur(region_of_interest, (65, 65), 30) #guassian blur on roi

    # impose this blurred image on original image to get final image
        frame[y:y+region_of_interest.shape[0], x:x+region_of_interest.shape[1]] = region_of_interest

        count+=1
    #When no coordinates recieved
   if not (x,y,w,h) in coordinates:
      cv2.putText(frame, f'Not detecting', (15, 30), cv2.FONT_HERSHEY_PLAIN, 2, (210,200,0), 1)
#Create Custom Window using OpenCV
   cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
   cv2.resizeWindow("webcam", 700, 500)
   cv2.imshow('webcam',frame)
   key = cv2.waitKey(1)
#If 'Q' is detected 
   if key==81 or key==113:
       print(f'Blurred {count-1} faces and exited')
       webcam.release()
       break

  
