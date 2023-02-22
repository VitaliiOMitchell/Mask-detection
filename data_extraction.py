from bs4 import BeautifulSoup
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Data_extractor:
    def __init__(self, data_im, data_annot):
        self.ims = data_im
        self.annot = data_annot

    def data_extraction(self):
        output = []
        path = self.annot
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
            #labels_for_torch.append(labels_output)
            data = [labels_output, box_data]
            output.append(data)
        
        return output
    
    
    def dataframe(self):
        box_vals = []
        labels_vals = []
        imgs = []
        images_list = os.listdir(self.ims)
        data = np.array(self.data_extraction(), dtype='object')
        labels, boxes = data[:, 0], data[:, 1] 
        for image, label, box in zip(images_list, labels, boxes):
            for label_val, box_val in zip(label, box):
                labels_vals.append(label_val)
                box_vals.append(box_val)
                imgs.append(image)
        df = pd.DataFrame(data=box_vals)
        df['image'] = imgs
        df['labels'] = labels_vals
        df.rename({0:'xmin', 1:'ymin', 2:'xmax', 3:'ymax'}, axis=1, inplace=True)
        LE = LabelEncoder()
        df['labels'] = LE.fit_transform(df['labels'])
        train_size = int(0.8 * len(df)) 
        val_size = len(df) - train_size
        df_train = df.iloc[0:train_size, :]
        df_val = df.iloc[train_size:val_size + len(df), :]
        df_val.index = range(0, len(df_val))
        
        return df_train, df_val

    # Helper functions
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
    extractor = Data_extractor('F:/Python/Projects/Mask-detection/images_masks', 'F:/Python/Projects/Mask-detection/annot_masks')
    output = extractor.data_extraction()
    train, val = extractor.dataframe()
    #with open('data_for_detection.pkl', 'wb') as d:
        #pickle.dump(output, d)
    #train.to_csv('df_for_train.csv', index=False)
    #val.to_csv('df_for_val.csv', index=False)    