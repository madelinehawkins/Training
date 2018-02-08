"""

Creates a test and train csv for the Pokemon Image set

"""


# Creates a CSV of all the image names in the folder.

import os
import csv

TRAIN_PATH = 'train/'
TEST_PATH = 'test/'


dirs_train = os.listdir(TRAIN_PATH)
dirs_test = os.listdir(TEST_PATH)
is_pic = True
is_poke = True
pokemon_name_map = {}
name_map_vector_index = 0
# Create an array of images with tags then put 25 of each in train.csv and 5 in test.csv



with open('train_out.csv', 'w') as csvfile:
    to_csv = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    to_csv.writerow(['image_name', 'tag'])
    for root, dirs, files in os.walk('train/'):
        for f in files:
            if f == '.DS_Store':
                is_poke = False
            if is_poke:
                label = root.split('/')[1]
                if label not in pokemon_name_map:
                    pokemon_name_map[label] = name_map_vector_index
                    name_map_vector_index += 1

                label = pokemon_name_map[label]
                to_csv.writerow([f, label])
            is_poke = True



# Todo : Make it so it doesn't have repeated code (helper function)
#
# with open('train/train_out.csv', 'w') as csvfile:
#     animal = 'None'
#     to_csv = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
#     to_csv.writerow(['image_name', 'tag'])
#     for file in dirs_train:
#         if 'out' in file:
#             is_pic = False
#             print(file)
#         tempLabel = root.split('/')[2]
#         if is_pic:
#             to_csv.writerow([file, animal])
#         is_pic = True
#
with open('test/test_out.csv', 'w') as csvfile:
    animal = 'None'
    to_csv = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    to_csv.writerow(['image_name'])
    for file in dirs_test:
            to_csv.writerow([file])

