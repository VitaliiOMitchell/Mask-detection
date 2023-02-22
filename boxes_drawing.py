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
                    cv.rectangle(img, (x[0], x[1]), (x[2], x[3]), (70, 255, 100), thickness=1)
                    cv.putText(img, 'Mask', (x[0]-10, x[1]-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (70, 255, 100), 1)
                elif y == 'without_mask':
                    cv.rectangle(img, (x[0], x[1]), (x[2], x[3]), (255, 0, 0), thickness=1)
                    cv.putText(img, 'No Mask', (x[0]-10, x[1]-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0), 1)
                elif y == 'mask_weared_incorrect':
                    cv.rectangle(img, (x[0], x[1]), (x[2], x[3]), (255, 255, 0), thickness=1)
                    cv.putText(img, 'Incorrect Mask', (x[0]-10, x[1]-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 0), 1)
            preprocessed_data.append(img)
        data_with_boxes = np.array(preprocessed_data, dtype='object')
        
        return data_with_boxes
            
if __name__ == '__main__':
    path = 'F:/Python/Projects/Mask-detection/images_masks'
    with open('data_for_detection.pkl', 'rb') as d:
        data = pickle.load(d)
    IP = Boxes_draving()
    output = IP.draw_boxes(path, data)
    plt.imshow(output[425])
    plt.show()