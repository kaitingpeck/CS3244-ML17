from PIL import Image
import pandas
import os
import glob

# original pixel dimensions of images
ORIGINAL_WIDTH = 512
ORIGINAL_HEIGHT = 384

# calculated from https://andrew.hedges.name/experiments/aspect_ratio/
NEW_WIDTH = 128
NEW_HEIGHT = 96

# Set project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_src_dir = project_dir + '/data/dataset-resized/full-data' # this file needs to contain all the images from all classes
img_dest_dir = project_dir + '/data/dataset-resized-' + str(NEW_WIDTH) + '-' + \
              str(NEW_HEIGHT) + '/' 
label_filepath = project_dir + '/data/labels/zero-indexed-files.txt' # this file contains the image names and corresponding labels


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def resize_images(label_filepath, img_src_dir, img_dest_dir):
    '''
    resizes images into a new folder
    '''
    df = pandas.read_csv(label_filepath, sep = " ", header = None, names = ['images','label'])
    image_names = df['images'].values # list of the image names
    img_dest_dir = make_dir(img_dest_dir)
    
    for image_name in image_names:
        image = Image.open(img_src_dir + '/' + image_name)
        resized_img = image.resize((NEW_WIDTH, NEW_HEIGHT))
        resized_img.save(img_dest_dir + '/' + image_name, "JPEG")

    print('Finished reading data. ' + str(len(image_names)) + ' images resized.')
    return

# "main" program
resize_images(label_filepath, img_src_dir, img_dest_dir)
    
