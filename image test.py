import os
import torchvision.io as t

directory = os.fsencode('data/dataset/val')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print('data/dataset/val/{}'.format(filename))
    #print('data/dataset/val/{}'.format(filename))
    t.read_image('data/dataset/val/{}'.format(filename))

#t.read_image('script/S9_F04_R2_IMAG2760.JPG')
