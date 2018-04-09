import imageio
import pandas
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from hpsklearn import HyperoptEstimator, svc
import scipy.io
from sklearn.model_selection import GridSearchCV

# this is the new input width and heigh
INPUT_WIDTH = 128
INPUT_HEIGHT = 96
INPUT_DEPTH = 3 #RGB

# Set project directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_src_dir = project_dir + '/data/dataset-resized-' + str(INPUT_WIDTH) + '-' + \
              str(INPUT_HEIGHT) + '/'
label_filepath = project_dir + '/data/labels/zero-indexed-files.txt' # this file contains the image names and corresponding labels

# Set parameters
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NUM_CLASSES = len(CLASSES)

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

    vector_length = INPUT_WIDTH * INPUT_HEIGHT * INPUT_DEPTH
    num_images = len(image_names)
    images = np.zeros((num_images, vector_length))

    # replace the image name with the respective image vectors
    for i in range(num_images):
        image_name = image_names[i]
        image_vector = imageio.imread(str(img_src_dir + '/' + image_name))
        image_vector = image_vector.flatten() # flatten into a 1-D vector
        images[i] = image_vector # stack the image vectors

    print('Finished reading data. ' + str(num_images) + ' images found in ' + \
          str(NUM_CLASSES) + ' classes.')
    return images, labels

def run_svm(images, labels, num_folds=5):
    '''
    takes in a matrix of images and a vector labels
    runs C-SVM with given C on the data matrix
    returns the score (accuracy in our case) calculated from k-fold validation,
    for different values of the parameters in 'parameters' list

    Can add in other parameters to optimize if using svm.SVC() e.g. kernels, etc.

    see:
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    for different types of params

    '''
    # set parameters to "optimize"
    parameters = {'C': [0.5, 1.0, 10]}
    svc = svm.LinearSVC(verbose=2, max_iter=10000) # Runs a linear SVM
    
    clf = GridSearchCV(svc, parameters, cv=num_folds, return_train_score=True)
    cv_results = clf.fit(images, labels).cv_results_

    # Collate results
    # mean_test_score = cv_results['mean_test_score'].tolist()
    # params = cv_results['params']

    # Collect index of best test score
    # best_test_score_idx = mean_test_score.index(max(mean_test_score))
    
    return cv_results

# "main" function
images, labels = read_data(label_filepath, img_src_dir)
cv_results = run_svm(images, labels, 5)
print(cv_results)

# print results
# print('Best score: ' + str(best_score))
# print('Achieved with C: ' + str(params))
