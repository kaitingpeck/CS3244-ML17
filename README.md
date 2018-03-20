# ML-for-garbage-classification


## Installation
please follow https://www.tensorflow.org/get_started/ for Tensorflow installation guide

The installation option used in this project is ``virtual environment``

## Running the Programme
 1. activate the environment by
```$ source ./bin/activate```

2. Run the estimator ```$ python premade_estimator.py```


## Project Structure
1. Source code is found in ```/src```
2. training data is found in ```/data```


## Acknowledgement

The datasets are obtained from 
https://github.com/garythung/trashnet


## Progress

### 20th March
1. First attempt: convert ```premade_estimator.py``` to accept our own jpg. However the data loaded into the scripts are already vectors. Hence I need to find a way to convert jpg into vectors

2. ```cnn_mnist.py``` is the example of mnist. However their image format is unknow. This is what happens when you use untyped language. No one knows what's going on and the original developer has already left the company. 

3. I'm looking for the guide to convert jpg images to vectors that tf can accept. also need to know how to label them and load a bunch of them sequentially. 

4. coming very close to getting some image processed
