from PIL import Image
import os
import numpy as np
import cv2
import sys
import shutil
np.set_printoptions(threshold=sys.maxsize)

def get_indices(directory):
    for filename in os.listdir(directory):
        print(filename, len(filename))
        print(os.getcwd())
        if len(filename) >= 17:
            shutil.move(directory+filename, os.path.join(directory+"/../", filename))
        # img = cv2.imread(directory + filename)
        # img_array = np.array(img)
        # img_annotation_indices = np.where(np.all(img_array[:,:] == [39,30,255], axis=-1))
        # np.save(directory+"/annotation_indices.npy", img_annotation_indices)
        # img_annotation_indices = np.load(directory+"/../annotation_indices.npy")
        # test_indices = np.copy(img_array)
        # for (i,j) in zip(img_annotation_indices[0], img_annotation_indices[1]):
        #     j_list = np.arange(0,j)
        #     # print(i,j)
        #     test_indices[i,j_list] = [0,0,0]
        # cv2.imwrite(directory+filename[:-4]+"_no_bg.png",test_indices)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_indices(os.getcwd() + "\\test_png_conversion\\")