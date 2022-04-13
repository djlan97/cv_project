import csv
import os.path
import shutil

data_dir = 'data/dataset/val'

with open(data_dir + '.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for i,row in enumerate(csv_reader):
        if i!=0:
            os.makedirs(os.path.join(data_dir, row[1]), exist_ok=True)
            shutil.move(os.path.join(data_dir, row[0]), os.path.join(data_dir, row[1], row[0]))
        