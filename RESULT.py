import  cv2
import os
import numpy as np
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))

frame_height = int(cap.get(4))
print(frame_width)
print(frame_height)


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.

#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1034, 522))

def videoWrite(paht_img, path_label_thuan, path_label_coclor, path_labal_img):
    for img in os.listdir(paht_img):
        id = img.split("_leftImg8bit.png")[0]
        print(id)
        link1 = paht_img + img
        link2 = path_label_thuan +  id + "_label_thuan.png"
        link3 = path_label_coclor + id  + "_pred_label_img.png"
        link4 = path_labal_img + id +  "_lable.png"

        image = cv2.imread(link1)
        image = cv2.resize(image, (512, 256))


        label_thuan = cv2.imread(link2)
        label_thuan = cv2.resize(label_thuan, (512, 256))


        label_color = cv2.imread(link3)
        label_color = cv2.resize(label_color, (512, 256))


        label_img = cv2.imread(link4)
        label_img = cv2.resize(label_img, (512, 256))


        line_h = np.ones((256, 10, 3))*255
        line_w = np.ones((10, 1034, 3))*255

        image_label_thuan = np.concatenate([image, line_h, label_thuan], axis=1)
        label_color_label_img = np.concatenate([label_color, line_h, label_img], axis=1)



        image_label_thuan__label_color_label_img = np.concatenate([image_label_thuan, line_w, label_color_label_img], axis=0)
        image_label_thuan__label_color_label_img = cv2.resize(image_label_thuan__label_color_label_img, dsize=(1434, 722))
        cv2.imwrite('result/' + id +"_result.png", image_label_thuan__label_color_label_img)




videoWrite(paht_img="video/demoVideo/stuttgart_01/", path_label_thuan = "logs/training_logs/model_1/videolabelthuan/",
           path_label_coclor="logs/training_logs/model_1/videopre/", path_labal_img="logs/training_logs/model_1/videolabel/")
