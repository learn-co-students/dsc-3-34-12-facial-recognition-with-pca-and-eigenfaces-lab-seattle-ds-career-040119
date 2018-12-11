
# Facial recognition with PCA and Eigenfaces - Lab

## Introduction
In this last lab of the section, we shall attempt to combine all the skills learnt around dimensionality reduction, running classification on reduced datasets and applying these in a image processing context. We shall look at a simple facial recognition experiment using PCA and Eigenfaces. As earlier, we shall try to run a classification task on the reduced dataset and check for the impact of PCA on classifier performance. Let's get on with it. 

## Objectives
You will be able to:
- Apply PCA to dataset containing images of human faces (or other visual objects)
- Calculate and visualize Eigenfaces from the given dataset
- Implement a machine learning pipeline in `scikit-learn` to perform preprocessing/classification and evaluation of facial a recognition task

## Facial Recognition

Face recognition is the challenge of classifying whose face is in an input image. This is different than face detection where the challenge is determining if there is a face in the input image. With face recognition, we need an existing database of faces. Given a new image of a face, we need to report the person’s name.

<img src="fr.jpg" width=400>

Formally, we can formulate face recognition as a classification task, where the inputs are images and the outputs are people’s names. We are going to perform a popular technique for face recognition called __eigenfaces__ which uses unsupervised dimensionality reduction with PCA.



__Eigenface__  is a name given to eigenvectors which are the components of the face itself. This technique has been used for face recognition where facial variations carry most importance. Although today there are better are more sophisticated approaches for performing this task e.g. deep networks, Eigenfaces still produce commendable results and with a little bit of tweaking, it can perform well where the focus is on data reduction and speed of execution. 

So let's get started , we shall first import required libraries below.

### Import Libraries
Import necessary libraries for above task 


```python
# Import necessary libraries 


# Code here 


```

## Olivetti Dataset

Image datasets can be huge in size. In order to create a facial dataset, a lot of preprocessing is generally required involving cropping, centering and color reductions etc to create data that contains JUST faces , which are comparable to each other. Sci-kit learn comes bundles with a couple of such datasets including `lfw` or "Labeled Faces in the Wild", containing images of 9 politicians, i.e. 9 classes. `Olivetti` faces dataset from AT&T contains a larger number of classes compared to lfw. So we let's run our experiment with this to really see the performance of classifier.


Before we start, lets analyze the Olivetti faces dataset to get a picture. The dataset consists of 400 images with greyscale 64×64 pixels . There are 10 images for each person, so there is 40 persons (target) which make it 40×100 equals 400 rows of data.  This means the dimensionality is huge with  64×64=4096 features. Lets import the dataset using scikit-learn built in function. The dataset is already normalized so you don't have to do preprocessing.

![](of.png)

Information on this dataset and how to import it can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html). 

### Import the Olivetti_faces dataset using scikit-learn and show the shape of dataset
We can `datasets` from `sklearn` to load the data. If the data isn't available on disk, scikit-learn will automatically download it. 


```python
# Importing olivetti dataset


# Code here 


```




    (400, 4096)



### Create X (features) and y (target) variabes from data and targets in the dataset. Check the shape. 
Create feature and target datasets from the `faces` dataset created above. 


```python
# Create feature and target set


# Code here 


```

    (400, 4096) (400,)


### Visualize first 20 images in the dataset 

Create subplots and run a loop to show first 20 images in the `faces` dataset. You can access an image by calling `faces.images[i]`, where `i` is the number of image we want to extract. 


```python


# Code here 


```


![png](index_files/index_10_0.png)


We see these faces have already been scaled to a common size. This is an important preprocessing piece for facial recognition, and is a process that can require a large collection of training data. This can be done in scikit-learn, but the challenge is gathering a sufficient amount of training data for the algorithm to work.

### Split the data into train and test sets using a 80/20 split. 
Split the features and  target variables for training and testing purpose and show the dimensions of both sets.


```python


# Code here 


```

    (320, 4096) (80, 4096)


We will now use scikit-learn’s PCA class to perform the dimensionality reduction. We have to select the number of components, i.e., the output dimensionality (the number of eigenvectors to project onto), that we want to reduce down to. We’ll use 150 components. Additionally, we’ll whiten our data, which is easy to do with a simple boolean flag! (Whitening just makes our resulting data have a unit variance, which has been shown to produce better results)

### Apply PCA to the training data with 150 components

4096 dimensions is a lot for any classifier. We can use PCA to reduce these features to a manageable size, while maintaining most of the information in the dataset.


```python


# Code here 


```




    PCA(copy=True, iterated_power='auto', n_components=150, random_state=None,
      svd_solver='auto', tol=0.0, whiten=True)



### Compute the mean face

One interesting part of PCA is that it computes the “mean” face, which can be interesting to examine. This can be computed with `pca.mean_`. This face will show you the mean for each dimension for all the images in the dataset. So it effectively shows you one MEAN face reflecting all the faces in the dataset. 


```python
# Show the mean face 


# Code here 


```




    <matplotlib.image.AxesImage at 0x1a1d211828>




![png](index_files/index_18_1.png)


### Visualize Principal Components for 25 Images - EigenFaces

The principal components measure deviations about this mean along orthogonal axes. Let's Visualize these principal components just like we visualized actual images above. These components can be accessed via `pca.components_[i]` where i is image you want to access. 


```python
# Visualize Principal Components



# Code here 


```


![png](index_files/index_20_0.png)


These components (“eigenfaces”) are ordered by their importance from top-left to bottom-right. We see that the first few components seem to primarily take care of lighting conditions; the remaining components pull out certain identifying features: the nose, eyes, eyebrows, etc.

With this projection computed, we can now project our original training and test data onto the PCA.

### Transform train and test datasets using trained PCA instance

We can apply the transform to bring our images down to a 150-dimensional space. 

Check the shape of PCA components and transform test and train sets with PCA. Show the shape of resulting datasets.


```python
# Transform train and test datasets using trained PCA algorithm



# Code here 


```

    (150, 4096)
    (320, 150) (80, 150)


These projected components correspond to factors in a linear combination of component images such that the combination approaches the original face.

### Run an SVM Classifier

After preprocessing our face data with PCA, we can run an SVM classifier to make predictions on the test set. For SVM, set C = 0.5 and gamma = 0.001. 


```python
# Create and train an instance of SVM classifier



# Code here 


```




    SVC(C=5.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)



### Pick up 30 images from the test data and predict their labels. 

Finally, we can evaluate how well this classification did. First, we might plot a few of the test-cases with the labels learned from the training set.

- Show the image, predicted value and weather it was a correct prediction or not. 


```python
# PRedict label for random test images and show the result


# Code here 


```


![png](index_files/index_28_0.png)


So out of all images, we see few wrong predictions, highlighted in red above. This doesn't sound so bad. IF you visually investigate this , you will realize that people predicted incorrectly are either not not looking straight at the camera , or smiling etc so their faces look quite different than other instances of same person. 

### Create a Classification Report

Predict the labels for complete test dataset and calculate the performance of classifier using `metrics.classification_report` to identify precision, recall, f1-score etc for each class in the dataset. Comment on the output. 


```python
# Create a classification report


# Code here 


```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00         2
              1       1.00      1.00      1.00         3
              2       1.00      1.00      1.00         3
              4       0.67      1.00      0.80         2
              5       1.00      1.00      1.00         2
              7       1.00      1.00      1.00         2
              8       1.00      1.00      1.00         4
              9       1.00      1.00      1.00         3
             10       1.00      1.00      1.00         4
             11       1.00      1.00      1.00         2
             12       1.00      1.00      1.00         1
             13       1.00      1.00      1.00         2
             14       1.00      1.00      1.00         2
             15       1.00      1.00      1.00         2
             16       1.00      1.00      1.00         3
             17       1.00      1.00      1.00         1
             18       1.00      1.00      1.00         2
             19       1.00      1.00      1.00         3
             20       0.50      1.00      0.67         2
             21       1.00      1.00      1.00         2
             23       1.00      1.00      1.00         1
             25       1.00      1.00      1.00         1
             26       1.00      1.00      1.00         1
             27       1.00      1.00      1.00         4
             28       1.00      1.00      1.00         5
             29       1.00      1.00      1.00         1
             30       1.00      1.00      1.00         2
             31       1.00      1.00      1.00         2
             33       1.00      1.00      1.00         1
             34       1.00      0.67      0.80         3
             35       1.00      1.00      1.00         2
             36       1.00      1.00      1.00         2
             37       1.00      1.00      1.00         1
             38       1.00      1.00      1.00         2
             39       1.00      0.60      0.75         5
    
    avg / total       0.98      0.96      0.96        80
    


Thats pretty good , we can identify the wrong predictions from this report and inspect the images independently to see why they were misclassified.  WE can visualize a confusion matrix to confirm this. 

### Create a confusion matrix from classification results


```python
# Create a confusion matrix


# Code here 


```




    <matplotlib.image.AxesImage at 0x1a21423438>




![png](index_files/index_34_1.png)


Above confirms our understanding around mis-classifications. 

Scikit-learn's ability to chain together different algorithms to create machine learning pipelines is a handy way to create complex computational architectures with just a few lines of code. Let's create a simple pipeline using skill we have already seen to perform above experiment in a single go. 

### Create an `scikit-learn` Pipeline to chain together PCA and SVM 


```python
# Chain PCA and SVM to run above experiment in a single execution. 


# Code here 


```

    [[1 0 0 ... 0 0 0]
     [0 3 0 ... 0 0 0]
     [0 0 3 ... 0 0 0]
     ...
     [0 0 0 ... 1 0 0]
     [0 0 0 ... 0 2 0]
     [0 0 0 ... 0 0 5]]
                 precision    recall  f1-score   support
    
              0       1.00      0.50      0.67         2
              1       1.00      1.00      1.00         3
              2       1.00      1.00      1.00         3
              4       0.67      1.00      0.80         2
              5       1.00      1.00      1.00         2
              7       1.00      1.00      1.00         2
              8       1.00      1.00      1.00         4
              9       0.75      1.00      0.86         3
             10       1.00      1.00      1.00         4
             11       1.00      1.00      1.00         2
             12       0.50      1.00      0.67         1
             13       1.00      1.00      1.00         2
             14       1.00      1.00      1.00         2
             15       1.00      0.50      0.67         2
             16       1.00      1.00      1.00         3
             17       1.00      1.00      1.00         1
             18       1.00      1.00      1.00         2
             19       1.00      1.00      1.00         3
             20       1.00      1.00      1.00         2
             21       1.00      1.00      1.00         2
             23       1.00      1.00      1.00         1
             25       1.00      1.00      1.00         1
             26       1.00      1.00      1.00         1
             27       1.00      1.00      1.00         4
             28       1.00      1.00      1.00         5
             29       1.00      1.00      1.00         1
             30       1.00      1.00      1.00         2
             31       1.00      1.00      1.00         2
             33       1.00      1.00      1.00         1
             34       1.00      1.00      1.00         3
             35       1.00      0.50      0.67         2
             36       1.00      1.00      1.00         2
             37       1.00      1.00      1.00         1
             38       1.00      1.00      1.00         2
             39       1.00      1.00      1.00         5
    
    avg / total       0.98      0.96      0.96        80
    


So here it is , our facial recognition system which performs quite well on this toy dataset that we saw. LAter on we shall look at more sophisticated approaches which might serve the same purpose. The focus here is on dimensionality reduction and developing an intuition around the idea that high dimensional datasets can be reduced to a large extent without too much compromise towards the predictive abilities of data. 

## Level Up - Optional 
- Use `lfw` dataset and run the above code again to understand the process better
- Create your own dataset! I

If you want to create your own face dataset, you’ll need several pictures of each person’s face (at different angles and lighting), along with the ground-truth labels. The wider variety of faces you use, the better the recognizer will do. The easiest way to create a dataset for face recognition is to create a folder for each person and put the face images in there. Make sure each are the same size and resize them so they aren’t large images! Remember that PCA will reduce the image’s dimensionality when we project onto that space anyways so using large, high-definition images won’t help and will slow down our algorithm. A good size is ~512×512 for each image. The images should all be the same size so you can store them in one numpy array with dimensions (num_examples, height, width) . (We’re assuming grayscale images). Then use the folder names to disambiguate classes. Using this approach, you can use your own images.

# Further Reading 
- [Maths behind computation of Eigenfaces](http://www.vision.jhu.edu/teaching/vision08/Handouts/case_study_pca1.pdf) 
- [PCA, Eigenfaces and all that](http://bugra.github.io/work/notes/2013-07-27/PCA-EigenFace-And-All-That/)

## Summary

Great! This was your first real introduction to analyzing images. You'll see much more on this in future lessons and labs!
