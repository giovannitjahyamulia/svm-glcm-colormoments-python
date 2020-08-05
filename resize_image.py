from PIL import Image
import cv2
 
# img = cv2.imread('/home/img/python.png', cv2.IMREAD_UNCHANGED)
# img = cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED)
# original_width, original_height = img.size

# wanted_width = 300
# scale_percent = 100 - ((wanted_width/original_width) * 100) # percent of original size
# new_width = int(original_width * scale_percent / 100)
# new_height = int(original_height * scale_percent / 100)

# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)

import os

source_directory = 'F:\\SKRIPSI 2019-2020\\6. Giovanni Tjahyamulia - 1620250081\\indonesian_traditional_food_dataset'
resized_directory = 'F:\\SKRIPSI 2019-2020\\6. Giovanni Tjahyamulia - 1620250081\\image_resized'

for folder in os.listdir(source_directory):
    current_path = "".join((source_directory, "\\", folder))
    file_list = os.listdir(current_path)

    resize_success = 0
    resize_failed = 0

    path_resized = "".join((resized_directory, "\\", folder))

    try:
        os.makedirs(path_resized)    
        print(folder,  "directory has created ")

        for file in os.listdir(current_path):
            img = cv2.imread(current_path + "\\" + file, cv2.IMREAD_UNCHANGED)
            
            # original_width, original_height = img.size

            # wanted_width = 300
            
            # scale_percent = 100 - ((wanted_width/original_width) * 100) # percent of original size
            # new_width = int(original_width * scale_percent / 100)
            # new_height = int(original_height * scale_percent / 100)

            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)

            # dim = (width, height)

            
            dim = (300, 400)

            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            status = cv2.imwrite(path_resized + "\\" + file, resized)
            
            if(status):
                resize_success = resize_success + 1
            else:
                resize_failed = resize_failed + 1
        
        print(folder ,  "resize is finished with result: ", str(resize_success) , "success and", str(resize_failed) , "failed")
        print("")
    except FileExistsError:
        print(folder, "directory already resized")