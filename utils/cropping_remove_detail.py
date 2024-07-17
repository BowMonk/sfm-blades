from PIL import Image
import os
import numpy as np
import cv2
import sys
import shutil
np.set_printoptions(threshold=sys.maxsize)

def get_indices(directory):
    post_dir = os.path.join(os.getcwd(),"/new_mapping")
    for subdir in os.listdir(directory):
        if "subsection" in subdir:
            new_dir = os.path.join(directory, subdir+"/")
            for filename in os.listdir(new_dir):

                print(filename, len(filename))
                print(os.getcwd())
            # if len(filename) >= 17:
            #     shutil.move(directory+filename, os.path.join(directory+"/../", filename))
                img = cv2.imread(new_dir + filename)
                img_array = np.array(img)
                img_annotation_indices = np.copy(img_array)
                remove_detail_indices = np.copy(img_array)
                for i in range(388,415):
                    # np.save(new_dir+"/annotation_indices.npy", img_annotation_indices)
                    # img_annotation_indices = np.load(new_dir+"/../annotation_indices.npy")
                    test_indices = np.copy(img_array)
                    # for (i,j) in zip(img_annotation_indices[0], img_annotation_indices[1]):
                    j_list = np.arange(95,121)
                    # print(i,j)
                    test_indices[i,j_list] = [0,0,0]
                    new_img = test_indices[60:418, 80:559]
                    cv2.imwrite(post_dir+"/"+filename[:-4]+"_detail_removed.png",new_img)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_indices(os.getcwd() + "/mapping")