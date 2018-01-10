# Creates a CSV of all the image names in the folder.

import os
import csv

IMG_PATH = 'train/'

dirs = os.listdir(IMG_PATH)

with open('train/out.csv', 'wb') as csvfile:
    animal = 'None'
    to_csv = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    to_csv.writerow(['image_name', 'tag'])
    for file in dirs:
        if 'dog' in file:
            animal = 'dog'
        else:
            animal = 'cat'
        to_csv.writerow([file, animal])





