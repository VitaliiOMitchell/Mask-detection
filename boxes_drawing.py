import cv2 as cv
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

class Boxes_draving:
    def draw_boxes(self, img_path, boxes_and_labels):
        path = img_path
        preprocessed_data = []
        for image, data in zip(os.listdir(path), boxes_and_labels):
            img = cv.imread(os.path.join(img_path, image))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            labels = list(data[0])
            boxes = data[1]
            for label, arr in zip(labels, boxes):
                x = arr
                y = label
                if y == 'with_mask':
                    cv.rectangle(img, (x[0], x[1]), (x[2], x[3]), (70, 255, 100), thickness=2)
                elif y == 'without_mask':
                    cv.rectangle(img, (x[0], x[1]), (x[2], x[3]), (255, 0, 0), thickness=2)
                elif y == 'mask_weared_incorrect':
                    cv.rectangle(img, (x[0], x[1]), (x[2], x[3]), (255, 255, 0), thickness=2)
            preprocessed_data.append(img)
        data_with_boxes = np.array(preprocessed_data, dtype='object')
        
        return data_with_boxes
            
if __name__ == '__main__':
    path = 'F:/Python/Projects/Mask-detection/images_masks'
    with open('data_for_detection.pkl', 'rb') as d:
        data = pickle.load(d)
    #IP = Image_prep()
    #output = ip.draw_boxes(path, data)
    #plt.imshow(output[353])
    #plt.show()