import cv2
import os

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1434, 722))

for img in os.listdir("result"):
    link = "result/" + img
    print(img)
    frame = cv2.imread(link)
    print(frame.shape)
    out.write(frame)



out.release()

# Closes all the frames

cv2.destroyAllWindows()