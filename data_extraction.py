from bs4 import BeautifulSoup
import os
import pickle
import numpy as np

class Data_extractor:
    def data_extraction(self, data_path):
        data = []
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
            
            box_data = np.array([self.labels(labels),
                                 [self.box(xmin_data), 
                                 self.box(ymin_data),
                                 self.box(xmax_data),
                                 self.box(ymax_data)]], dtype=np.ndarray)
            box_data[1] = np.array(box_data[1]).T
            data.append(box_data)
            
        return data

    def box(self, boxes):
        coordinates = []
        for data in boxes:
            val = int(data.text)
            coordinates.append(val)
        return coordinates
    
    def labels(self, labels):
        labels_arr = []
        for label in labels:
            labels_arr.append(label.text)
        return tuple(labels_arr)


if __name__ == '__main__':
    annot_path = 'F:/Python/Projects/Mask-detection/annot_masks'
    extractor = Data_extractor()
    output = extractor.data_extraction(annot_path)
    with open('data_for_detection.pkl', 'wb') as d:
        pickle.dump(output, d)
    