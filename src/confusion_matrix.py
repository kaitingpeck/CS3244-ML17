import itertools
import imageio
import pandas
import numpy as np
from sklearn import svm
    
from sklearn.model_selection import cross_val_score
from hpsklearn import HyperoptEstimator, svc
import scipy.io
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn import preprocessing
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    title = title + '.png'
    plt.show()
    plt.savefig(title, format='png')

def generate_confusion_matrix(y_val, y_pred, plot_title, list_classes=['survive','don\'t survive']):
    '''
    takes in y_val (ground truth for validation set), y_pred (predicted values for validation set),
    plot_title (name of plot), list_classes (['survive', 'dont survive'])
    '''
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cnf_matrix, classes=list_classes, normalize=True,
                              title= plot_title)
 def random_forest(X_train, y_train, num_trees):
    rf = RandomForestClassifier(n_estimators = num_trees, random_state=0, oob_score = True, criterion='gini')
    rf.fit(X_train, y_train)
    return rf
  
def k_fold_rf(X, y, num_trees, num_folds=5):
    kf = StratifiedKFold(n_splits=5)
    i = 1
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # For printing purposes
        if i == 1:
            rf_title = str(num_trees) + ' trees (1st fold)'
        elif i == 2:
            rf_title = str(num_trees) + ' trees (2nd fold)'
        elif i == 3:
            rf_title = str(num_trees) + ' trees (3rd fold)'
        else:
            rf_title = str(num_trees) + ' trees (' + str(i) + 'th fold)'

        # Obtained trained model using this set of training data
        model = random_forest(X_train, y_train, num_trees)
        print('Model training completed')

        # save the model to disk
        filename = 'finalized_model-' + str(num_trees) + '-trees-' + str(i) + '-fold' + '.sav'
        pickle.dump(model, open(filename, 'wb'))
        
        # Report ooberror
        print(rf_title + ' Out-of-bag (accuracy) estimate: ' + str(model.oob_score_))

        # Run model on validation data
        y_pred = model.predict(X_val)

        # Compute and plot normalized confusion matrix
        plot_title = rf_title + ' - Normalized confusion matrix'
        generate_confusion_matrix(y_val, y_pred, plot_title)
        
        # set index for next fold
        i += 1

def run_rf_diff_trees(X, y, num_folds=5):
    param_trees = [20,30,40,50,60]
    for num_trees in param_trees:
        print('... Running random forest with ' + str(num_trees) + ' trees ...')
        k_fold_rf(X, y, num_trees, 5)

run_rf_diff_trees(train_x, train_y, 5)   
