# ECS 111 Emotion Detection in Children's Drawings Final Project

This repository will contains all the code that was used for the ECS 111 Emotion Detection in Children's Drawings project, from smaller tasks such as normalization, resizing, etc. to the final algorithm that is built for emotion detection, and the risk detection tasks. The main goal of this project is to build an algorithm that is capable of identifying the emotions present in a child's drawing, divided into four classes, anger, fear, happy, and sad, along with performing a risk detection analysis.

Project Description: 
[pdf](https://cdn-uploads.piazza.com/paste/m6slvq75i3k31k/d7030087025aca9f8d670bf7c3dc3e23df4e923c1545abb7a956953004bd352e/ECS111_SQ_2025_-_project.pdf)

Project Proposal: 
[Proposal](https://docs.google.com/document/d/1mfopRWyw--y7h06VD_z1D53k16H_82CRohDGWVvjiAk/edit?usp=sharing)

Project Presentation: 
[Presentation](https://docs.google.com/presentation/d/1iKzccb15EB8VY1FnFnWSoCPZVB1LiDXuvovfqDzCQkk/edit?usp=sharing)

Dataset: 
[Kaggle](https://www.kaggle.com/datasets/vishmiperera/children-drawings?select=data)

Converted Data: 
[Google Drive](https://drive.google.com/drive/folders/1xkOsVxCkwTQJi3ruOoOHVbvzSn8kqRCS?q=sharedwith:public%20parent:1xkOsVxCkwTQJi3ruOoOHVbvzSn8kqRCS) \
Note: This Google Drive contains the converted images (preprocessed) along with the unconverted images and other files that were used throughout the duration of the project.

Presentation: 
[Google Slides](https://docs.google.com/presentation/d/1iKzccb15EB8VY1FnFnWSoCPZVB1LiDXuvovfqDzCQkk/edit?usp=sharing)

Final Report: 
[Report](https://docs.google.com/document/d/1W-_J3VC6vMQZH5OmvAEEN8us_nsKwGjT3-p0BsSWbKo/edit?usp=sharing)


## Code Files Description
### Preprocessing: 
1. conversion_split.py: Converted all images to file type .jpg along with splits the dataset into training and testing sets. Creates csv files containing the relative paths to all images in the dataset along with the class they belong into defined by the name of the directory that the image belongs in.
2. normalization.py: Normalized pixel values via images/255 along with standardizing the images to be 224 x 224.
3. data encoding for emotions.py: Takes in the csv files created for the training and testing set and encodes the labels using both label encoding and one-hot emcoding methods./

<<<<<<< HEAD
### Algorithm:
algorithm.py: Contains the MobileNetV2 code along with the training algorithm on the training set.
analysis_pipeline.py: 'please add in description here!'
=======
- ~~Interim Checkup: 5/28/25 --> 6/7pm~~
- ~~Finished Algorithm Checkin: 6/1/25 --> 7pm~~
>>>>>>> 321a60f7148d6515c10e2f7f3c90aecc0e955b54

### Risk Detection: 
'please add code used here!'

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




