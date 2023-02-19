from bs4 import BeautifulSoup
import os
import pickle
import numpy as np
import torch

class Data_extractor:
    def data_extraction(self, data_path):
        output = []
        path = data_path
        for annot in os.listdir(path):
            file = os.path.join(path, annot)
            with open(file, 'r') as f:
                content = f.readlines()
            content = ''.join(content)
            soup = BeautifulSoup(content, 'lxml')
            xmin_data = soup.find_all('xmin')
            ymin_data = soup.find_all('ymin')
            xmax_data = soup.find_all('xmax')
            ymax_data = soup.find_all('ymax')
            labels = soup.find_all('name')
            
            labels_output = self.labels_(labels)
            box_data = np.array([self.box_(xmin_data), 
                                 self.box_(ymin_data),
                                 self.box_(xmax_data),
                                 self.box_(ymax_data)]).T
            data = [labels_output, box_data]
            output.append(data)
        
        return output[2][1]

    def box_(self, boxes):
        coordinates = []
        for data in boxes:
            val = int(data.text)
            coordinates.append(val)
        return coordinates
    
    def labels_(self, labels):
        labels_arr = []
        for label in labels:
            labels_arr.append(label.text)
        return tuple(labels_arr)

if __name__ == '__main__':
    annot_path = 'F:/Python/Projects/Mask-detection/annot_masks'
    extractor = Data_extractor()
    output = extractor.data_extraction(annot_path)
    #with open('data_for_detection.pkl', 'wb') as d:
        #pickle.dump(output, d)
    print(output)