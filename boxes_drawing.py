import cv2 as cv
import os
import pickle
import matplotlib.pyplot as plt

class Boxes_drawing:
    def draw_boxes(self, img_path, boxes_and_labels):
        path = img_path
        preprocessed_data = []
        for image, data in zip(os.listdir(path), boxes_and_labels):
            img = cv.imread(os.path.join(img_path, image))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            labels = list(data[0])
            boxes = data[1]
            for label, arr in zip(labels, boxes):
                box = arr
                y = label
                if y == 'with_mask':
                    cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (70, 255, 100), thickness=2)
                    cv.putText(img, 'Mask', (box[0]-10, box[1]-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (70, 255, 100), 1)
                elif y == 'without_mask':
                    cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=2)
                    cv.putText(img, 'No Mask', (box[0]-10, box[1]-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 0, 0), 1)
                elif y == 'mask_weared_incorrect':
                    cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), thickness=2)
                    cv.putText(img, 'Incorrect Mask', (box[0]-10, box[1]-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 0), 1)
            preprocessed_data.append(img)
        
        return preprocessed_data
            
if __name__ == '__main__':
    path = 'F:/Python/Projects/Mask-detection/images_masks'
    with open('data_for_detection.pkl', 'rb') as d:
        data = pickle.load(d)
    IP = Boxes_drawing()
    output = IP.draw_boxes(path, data)
    #plt.imshow(output[0])
    #plt.show()