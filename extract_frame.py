import cv2 as cv

cap = cv.VideoCapture('data/IMG_5128.mp4')

if not cap.isOpened():
    print("Cannot open the source")
    quit()

cnt = 0

while True:
        # Capture frame-by-frame
        ret, frame = cap.read() 
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
            
        # Display the resulting frame
        cv.imshow('frame', frame)
        cv.imwrite(f'data/result/result_{cnt}.jpg', frame)
        cnt += 1

        if cv.waitKey(20) == ord('q'):
            break

# When everything done, release the capture
cap.release()

cv.destroyAllWindows()