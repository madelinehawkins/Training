# Creates a CSV of all the image names in the folder.

import os
import csv

IMG_PATH = 'train/'

dirs = os.listdir(IMG_PATH)

with open('train/out.csv', 'wb') as csvfile:
    to_csv = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    for file in dirs:
        to_csv.writerow([file])





