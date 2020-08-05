# import os, random
# import shutil

# m = 13779
# n = 4133
# x = 9647

# src_dir = "D:/Kuliah/Semester 7/Skripsi/RegionContrast/SaliencyRC-master/cell_images/Parasitized/"
# dst_dir_test = "D:/Kuliah/Semester 7/Skripsi/RegionContrast/SaliencyRC-master/image_test/PArasitized/"
# dst_dir_train = "D:/Kuliah/Semester 7/Skripsi/RegionContrast/SaliencyRC-master/image_train/Parasitized/"

# file_list = os.listdir(src_dir)

# for f in range(n):
#     a = random.choice(file_list)
#     file_list.remove(a)
#     shutil.copy(src_dir + a, dst_dir_test + a)

# for g in range(x):
#     a = random.choice(file_list)
#     file_list.remove(a)
#     shutil.copy(src_dir + a, dst_dir_train + a)

import os, random
import shutil
import math

source_directory = 'F:\\SKRIPSI 2019-2020\\6. Giovanni Tjahyamulia - 1620250081\\image_resized'
test_directory = 'F:\\SKRIPSI 2019-2020\\6. Giovanni Tjahyamulia - 1620250081\\image_test'
train_directory = 'F:\\SKRIPSI 2019-2020\\6. Giovanni Tjahyamulia - 1620250081\\image_train'

for folder in os.listdir(source_directory):
    current_path = "".join((source_directory, "\\", folder))
    file_list = os.listdir(current_path)

    path_test = "".join((test_directory, "\\", folder))
    path_train = "".join((train_directory, "\\", folder))

    try:
        os.makedirs(path_test)    
        print("Directory", folder, "Test Created ")
    except FileExistsError:
        print("Directory", folder, "Test already exists")
    
    try:
        os.makedirs(path_train)    
        print("Directory", folder, "Train Created ")
    except FileExistsError:
        print("Directory", folder, "Train already exists")
    
    images = len(file_list)
    train_images = math.floor(images * 70 / 100)
    test_images = math.ceil(images * 30 / 100)

    
    for i in range(train_images):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.copy(current_path + "\\" + a, path_train + "\\" + a)

    for i in range(test_images):
        a = random.choice(file_list)
        file_list.remove(a)
        shutil.copy(current_path + "\\" + a, path_test + "\\" + a)