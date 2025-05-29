# ECS 111 Emotion Detection in Children's Drawings Project

This repository will contain all the code that is used for the ECS 111 Emotion Detection in Children's Drawings project, from smaller tasks such as normalization, resizing, etc. to the final algorithm that is built for emotion detection. The main goal of this project is to properly and accurately identify the emotion present in children's drawings.

Project Description: 
[pdf](https://cdn-uploads.piazza.com/paste/m6slvq75i3k31k/d7030087025aca9f8d670bf7c3dc3e23df4e923c1545abb7a956953004bd352e/ECS111_SQ_2025_-_project.pdf)

Project Proposal: 
[Proposal](https://docs.google.com/document/d/1mfopRWyw--y7h06VD_z1D53k16H_82CRohDGWVvjiAk/edit?usp=sharing)

Dataset: 
[Kaggle](https://www.kaggle.com/datasets/vishmiperera/children-drawings?select=data)

Converted Data: 
[Google Drive](https://drive.google.com/drive/folders/1xkOsVxCkwTQJi3ruOoOHVbvzSn8kqRCS?q=sharedwith:public%20parent:1xkOsVxCkwTQJi3ruOoOHVbvzSn8kqRCS) \
Note: Feel free to use the images here for any task and modify it for the algorithm too.

Task Plan: 
- Week 7:
  - ~~resizing + normalizing (sadia)~~
  - ~~conversion of file types to 1 type + division of testing and training data (jes)~~
  - ~~data encoding for the emotions + figure out algorithm to use (research & share with group on findings/algorithms to go for) (chuck)~~
  - extra:
    - technically everyone can do research on the algorithm :))
    - if tasks are done early and we have an algorithm in mind, we can start building our algorithm!

Week 8 Meeting Notes: 
- Algorithm: ~~ResNet50~~ Modified to use MobileNetV2 as it is better for small data sets such as ours with 1.1K images only
- Encoding: one-hot

- Week 8:
  - OVERALL GOAL: finish up algorithm by end of the week/before next meeting
  - ~~Build & fine-tuning ResNet50, compile model, add dropout if needed, train with callbacks (jes)~~
  - Create data augmentation pipeline, evaluate model, visualize predictions (chuck)
  - Research inference logic, run prediction script on test data, summarize model performance + charts if necessary (sadia) 

Additional Notes (5/25):
- Need to add more elements for expanding the project scope
  - Option 1: Prediction for what the Emotion of the next week might be 
    - Might use: Linear Regression
    - This option is just a thought can be modified to fit TA's comment 
  - Option 2: Simulation of student groups to predict which students are at risk
    - Might use: Simple RNN classifier
    - This option is also just a thought so feel free to modify

- ~~Interim Checkup: 5/28/25 --> 6/7pm?~~
- Finished Algorithm Checkin: 5/31/25 --> 2pm

- Week 9 + 10: 
  - Work on presentation + paper

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


