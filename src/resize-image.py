from PIL import Image
import pandas
import os
import glob

ORIGINAL_WIDTH = 512
ORIGINAL_HEIGHT = 384

# calculated from https://andrew.hedges.name/experiments/aspect_ratio/
NEW_WIDTH = 64
NEW_HEIGHT = 48

img_src_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/dataset-resized/full-data'
img_dest_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/dataset-resized-64-48/'
label_filepath = 'C:/Users/Kai/Desktop/CS3244/Project/data/labels/zero-indexed-files.txt'


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def read_data(label_filepath, img_src_dir, img_dest_dir):
    '''
    transforms data into the following:
    images = each row is a vector of the pixels of the RGB image
    labels = a column vector representing the respective labels of the image vector
    
    '''
    df = pandas.read_csv(label_filepath, sep = " ", header = None, names = ['images','label'])
    image_names = df['images'].values # list of the image names
    img_dest_dir = make_dir(img_dest_dir)

   # print(image_names[:50])
    
    for image_name in image_names:
        image = Image.open(img_src_dir + '/' + image_name)
        resized_img = image.resize((NEW_WIDTH, NEW_HEIGHT))
        # print(img_dest_dir)
        resized_img.save(img_dest_dir + '/' + image_name, "JPEG")

    print('Finished reading data. ' + str(len(image_names)) + ' images resized.')
    return

read_data(label_filepath, img_src_dir, img_dest_dir)
    
