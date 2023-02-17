import requests
from bs4 import BeautifulSoup
import os
import pickle

def data_extraction(path):
    output = []
    
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
                if len(coordinate_x) == 15:
                    try:
                        elem = coordinate_x[i] + coordinate_x[i+1]
                        elem = int(elem)
                        xmin.append(elem)
                    except:
                        pass
                else:
                    try:
                        elem = coordinate_x[i] + coordinate_x[i+1] + coordinate_x[i+2]
                        elem = int(elem)
                        xmin.append(elem)
                    except:
                        pass
        
        for coordinate_y in ymin_data:
            coordinate_y = str(coordinate_y)
            for i in range(len(coordinate_y)-1):
                if len(coordinate_y) == 15:
                    try:
                        elem = coordinate_y[i] + coordinate_y[i+1]
                        elem = int(elem)
                        ymin.append(elem)
                    except:
                        pass
                else:
                    try:
                        elem = coordinate_y[i] + coordinate_y[i+1] + coordinate_y[i+2]
                        elem = int(elem)
                        ymin.append(elem)
                    except:
                        pass
        
        for coordinate_x in xmax_data:
            coordinate_x = str(coordinate_x)
            for i in range(len(coordinate_x)-1):
                if len(coordinate_x) == 15:
                    try:
                        elem = coordinate_x[i] + coordinate_x[i+1]
                        elem = int(elem)
                        xmax.append(elem)
                    except:
                        pass
                else:
                    try:
                        elem = coordinate_x[i] + coordinate_x[i+1] + coordinate_x[i+2]
                        elem = int(elem)
                        xmax.append(elem)
                    except:
                        pass
        
        for coordinate_y in ymax_data:
            coordinate_y = str(coordinate_y)
            for i in range(len(coordinate_y)-1):
                if len(coordinate_y) == 15:
                    try:
                        elem = coordinate_y[i] + coordinate_y[i+1]
                        elem = int(elem)
                        ymax.append(elem)
                    except:
                        pass
                else:
                    try:
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
        output.append(im_info)

    return output

def labels_extraction(path):
    labels = []
    with open(path, 'r') as data:
        content = data.readlines()
    content = ''.join(content)
    soup = BeautifulSoup(content, 'lxml')
    data = soup.find_all('name')
    return data

if __name__ == '__main__':
    annot_path = os.path.join('F:/Python/Projects/Mask detection/annot_masks/maksssksksss3.xml')
    print(labels_extraction(annot_path))