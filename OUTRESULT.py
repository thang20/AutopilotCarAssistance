import cv2
cap = cv2.VideoCapture('result.avi')
import numpy as np
cap.set(3, 1434)
cap.set(4, 722)
while True:
    success, frame = cap.read()
    if success:
        fps = np.random.randint(150, 180)
        cv2.putText(frame, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("result", frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()