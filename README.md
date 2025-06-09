# ECS 111 Emotion Detection in Children's Drawings Final Project

This repository contains all the code that was used for the Emotion Detection in Children's Drawings final project for ECS 111, from preprocessing such as normalization, resizing, etc. to the final algorithm that was built for emotion detection, and the risk detection tasks. The main goal of this project is to build an algorithm that is capable of identifying the emotions present in a child's drawing, divided into four classes, anger, fear, happy, and sad, along with performing a risk detection analysis.

## Dataset: 
[Kaggle](https://www.kaggle.com/datasets/vishmiperera/children-drawings?select=data)

## Converted Data: 
[Google Drive](https://drive.google.com/drive/folders/1xkOsVxCkwTQJi3ruOoOHVbvzSn8kqRCS?q=sharedwith:public%20parent:1xkOsVxCkwTQJi3ruOoOHVbvzSn8kqRCS) \
Note: This Google Drive contains the converted images (preprocessed) along with the unconverted images and other files that were used throughout the duration of the project.


## Code Files Description
### Preprocessing: 
1. **conversion_split.py**: Converted all images to file type .jpg along with splits the dataset into training and testing sets. Creates csv files containing the relative paths to all images in the dataset along with the class they belong into defined by the name of the directory that the image belongs in.
2. **normalization.py**: Normalized pixel values via images/255 along with standardizing the images to be 224 x 224.
3. **data encoding for emotions.py**: Takes in the csv files created for the training and testing set and encodes the labels using both label encoding and one-hot emcoding methods.

### Algorithm:
1. **algorithm.py**: Contains the MobileNetV2 code along with the training algorithm on the training set.<br>
2. **analysis_pipeline.py**: Evaluates the emotion classification model by reporting accuracy, F1 score, ROC AUC, and RMSE, and generates various visualizations such as the confusion matrix, ROC curves, and confidence distribution to support performance analysis.

### Risk Detection: 
'please add code used here!'

## Instructions & how to run code
Go to the [Kaggle site](https://www.kaggle.com/datasets/vishmiperera/children-drawings/data) and download the images, also available on the Google Drive [Google Drive](https://drive.google.com/drive/folders/1PujZ1zqCYv2RPzIiG6jLSAZhP6mbpL2a?zx=dsjt3vfp82wq) under the not_converted folder. For most .py files, the paths would have to be updated to be reflective of your desktop because when working on the project, some group member's relative paths weren't working as expected, hence the use of absolute paths in the files.
### Order of running the scripts
1. Run conversion_split.py to obtain the converted file types along with the training and testing split (.csv files). Note: the converted files and .csv files can also be obained from the [Google Drive](https://drive.google.com/drive/folders/186FmT192KDe3QskJ-S7ILmuxSSRn_f6f?zx=dsjt3vfp82wq) under the converted folder. 
2. Run the normalization.py file to obtain standardized and normalized version of the images.
3. Run data encoding for emotions.py to obtain the one-hot encoded version of the training and testing sets. The csv files for this step can also be found on the Google Drive.
4. Run algorithm.py to get the trained model along with the hyperparameters that were set to obtain the highest accuracy and F1-scores
5. Run localized_model_algorithm.py to obtain model evaluation plots and other statistics
6. 'insert instructions to run the risk detection code!'

## File and Directory Structure
```
README.md
code/
data/
├───converted
│   ├───data_converted
│   │   ├───Angry
│   │   ├───Fear
│   │   ├───Happy
│   │   └───Sad
│   └───NewArts2_converted
│       ├───Angry
│       ├───Fear
│       ├───Happy
│       └───Sad
└───not_converted
    ├───data
    │   ├───Angry
    │   ├───Fear
    │   ├───Happy
    │   └───Sad
    └───NewArts2
        ├───Angry
        ├───Fear
        ├───Happy
        └───Sad
```





