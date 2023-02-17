from bs4 import BeautifulSoup
import os
import pickle


def data_extraction(path):
    box_data = []
    
    for annot in os.listdir(path):
        annot = os.path.join(path, annot)
        with open(annot, 'r') as file:
            content = file.readlines()
        content = ''.join(content)
        
        soup = BeautifulSoup(content,'lxml')
        
        xmin_data = soup.findAll('xmin')
        ymin_data = soup.findAll('ymin')
        xmax_data = soup.findAll('xmax')
        ymax_data = soup.findAll('ymax')
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for coordinate_x in xmin_data:
            coordinate_x = str(coordinate_x)
            for i in range(len(coordinate_x)-1):
                try:
                    if len(coordinate_x) == 14:
                        elem = coordinate_x[i] 
                        elem = int(elem)
                        xmin.append(elem)
                    elif len(coordinate_x) == 15:
                        elem = coordinate_x[i] + coordinate_x[i+1]
                        elem = int(elem)
                        xmin.append(elem)
                    elif len(coordinate_x) == 16:
                        elem = coordinate_x[i] + coordinate_x[i+1] + coordinate_x[i+2]
                        elem = int(elem)
                        xmin.append(elem)
                except:
                    pass
        
        for coordinate_y in ymin_data:
            coordinate_y = str(coordinate_y)
            for i in range(len(coordinate_y)-1):
                try:
                    if len(coordinate_y) == 14:
                        elem = coordinate_y[i] 
                        elem = int(elem)
                        ymin.append(elem)
                    elif len(coordinate_y) == 15:
                        elem = coordinate_y[i] + coordinate_y[i+1]
                        elem = int(elem)
                        ymin.append(elem)
                    elif len(coordinate_y) == 16:
                        elem = coordinate_y[i] + coordinate_y[i+1] + coordinate_y[i+2]
                        elem = int(elem)
                        ymin.append(elem)
                except:
                    pass
        
        for coordinate_x in xmax_data:
            coordinate_x = str(coordinate_x)
            for i in range(len(coordinate_x)-1):
                try:
                    if len(coordinate_x) == 14:
                        elem = coordinate_x[i] 
                        elem = int(elem)
                        xmax.append(elem)
                    elif len(coordinate_x) == 15:
                        elem = coordinate_x[i] + coordinate_x[i+1]
                        elem = int(elem)
                        xmax.append(elem)
                    elif len(coordinate_x) == 16:
                        elem = coordinate_x[i] + coordinate_x[i+1] + coordinate_x[i+2]
                        elem = int(elem)
                        xmax.append(elem)
                except:
                    pass
        
        for coordinate_y in ymax_data:
            coordinate_y = str(coordinate_y)
            for i in range(len(coordinate_y)-1):
                try:
                    if len(coordinate_y) == 14:
                        elem = coordinate_y[i] 
                        elem = int(elem)
                        ymax.append(elem)
                    elif len(coordinate_y) == 15:
                        elem = coordinate_y[i] + coordinate_y[i+1]
                        elem = int(elem)
                        ymax.append(elem)
                    elif len(coordinate_y) == 16:
                        elem = coordinate_y[i] + coordinate_y[i+1] + coordinate_y[i+2]
                        elem = int(elem)
                        ymax.append(elem)
                except:
                    pass
        
        data_dict = {'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax}
        im_info = (annot[46:], data_dict)
        box_data.append(im_info)

    return box_data

def labels_extraction(path):
    labels_arr = []
    
    for annot in os.listdir(path):
        annot = os.path.join(path, annot)
        with open(annot, 'r') as data:
            content = data.readlines()
        content = ''.join(content)
        soup = BeautifulSoup(content, 'lxml')
        data = soup.find_all('name')
        labels = {'with_mask': 0, 
                'without_mask': 0, 
                'mask_weared_incorrect': 0
                }
        for val in data:
            val = str(val)
            if 'with_mask' in val:
                labels['with_mask'] += 1
            elif 'without_mask' in val:
                labels['without_mask'] += 1
            elif 'mask_weared_incorrect' in val:
                labels['mask_weared_incorrect'] += 1
        labels_arr.append(labels) 
    
    return labels_arr


if __name__ == '__main__':
    annot_path = os.path.join('F:/Python/Projects/Mask-detection/annot_masks')
    box_data = data_extraction(annot_path)
    labels = labels_extraction(annot_path)
    
    with open('box_data.pkl', 'wb') as bd:
        pickle.dump(box_data, bd)
    with open('labels.pkl', 'wb') as l:
        pickle.dump(labels, l)