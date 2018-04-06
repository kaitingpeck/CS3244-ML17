import imageio
import pandas
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# File paths
# root_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/test-runs-svm/' + str(run_idx)
img_src_dir = 'C:/Users/Kai/Desktop/CS3244/Project/data/dataset-resized-64-48/'
label_filepath = 'C:/Users/Kai/Desktop/CS3244/Project/data/labels/zero-indexed-files.txt'

# Training paramaters
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NUM_CLASSES = len(CLASSES)
INPUT_WIDTH = 64
INPUT_HEIGHT = 48
INPUT_DEPTH = 3 #RGB

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def read_data(label_filepath, img_src_dir):
    '''
    transforms data into the following:
    images = each row is a vector of the pixels of the RGB image
    labels = a column vector representing the respective labels of the image vector
    
    '''
    df = pandas.read_csv(label_filepath, sep = " ", header = None, names = ['images','label'])
    image_names = df['images'].values # list of the image names
    labels = df['label'].values

    '''print('Image Names: ' + str(image_names[:50]))
    print('Labels: ' + str(labels[:50]))'''

    vector_length = INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH
    images = np.zeros((len(image_names), vector_length))
    # print('Size of image matrix: ' + str(images.shape))
    
    # replace the image name with the respective image vectors
    for i in range(len(image_names)):
        image_name = image_names[i]
        image_vector = imageio.imread(str(img_src_dir + '/' + image_name))
        image_vector = image_vector.flatten() # flatten into a 1-D vector
        images[i] = image_vector # stack the image vectors

    print('Finished reading data. ' + str(len(image_names)) + ' images found in ' + \
          str(NUM_CLASSES) + ' classes.')
    return images, labels

def run_svm(images, labels, num_folds=5):
    '''
    takes in a matrix of images and a vector labels
    runs C-SVM with given C on the data matrix
    returns the score (accuracy in our case) calculated from k-fold validation

    '''
    clf = svm.SVC()
    scores = cross_val_score(clf, images, labels, cv=num_folds)
    return scores

'''def k_fold(images, labels):
    kf = KFold(n_splits=2)
    for train, test in kf.split(images):
        images_train, images_test = images[train], images[test]
        labels_train, labels_test = labels[train], labels[test]
        svm = run_svm(images_train, labels_train) # trained SVM model
        
    return '''
        
images, labels = read_data(label_filepath, img_src_dir)
scores = run_svm(images, labels, 3)
print(scores)
