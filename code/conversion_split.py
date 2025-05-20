'''
Note: I moved the emotions folder from the NewArts2 (a subdirectory of NewArts2) folder into the first NewArts2
folder so if you directly download the emotions folder from Kaggle and run the following code, the NewArts2_converted
could be empty.

Use the Google Drive not_converted folder to run this code if you want to do the conversion yourself.
'''
from PIL import Image
import os
import random
import csv

os.chdir(r'C:\Users\jesly')
#replace for your respective device's path
directory = r'ECS-111-Emotion-Detection\data'
basedirectory = r'ECS-111-Emotion-Detection\data\not_converted'
#refer to readme file in github for file structure
target = r'ECS-111-Emotion-Detection\data\converted'
os.makedirs(target, exist_ok = True)

for group in os.listdir(basedirectory):
    combined = os.path.join(basedirectory, group)
    #joins to create a full path to the next subdirectory

    if os.path.isdir(combined):
        if not group.endswith('_converted'):
            newdir = f'{group}_converted'
            #name of the folder to append the converted files to
        else:
            newdir = group

        newpath = os.path.join(target, newdir)
        #the new path to the folder with the converted files
        os.makedirs(newpath, exist_ok = True)

        for classnm in os.listdir(combined):
            classpath = os.path.join(combined, classnm)
            #identifies the class subdirectory

            if os.path.isdir(classpath):
                newpathclass = os.path.join(newpath, classnm)
                os.makedirs(newpathclass, exist_ok = True)
                #creates the directory that stores all the converted files into

                for file in os.listdir(classpath):
                    filepath = os.path.join(classpath, file)
                    #identifies the path for the specific file

                    try:
                        with Image.open(filepath) as im:
                            rgb_im = im.convert('RGB')
                            base = os.path.splitext(file)[0]
                            newfilepath = os.path.join(newpathclass, f'{base}.jpg')
                            rgb_im.save(newfilepath, format = 'JPEG')
                        #performs the conversion
                    except Exception as e:
                        print(f'skipping {filepath}')
                        #in the event that the file is not an image

###Training-Testing Split###
#training-testing split: 80-20
images = []

for root, _, files in os.walk(target):
    for file in files:
        if file.endswith('.jpg'):
            fullpath = os.path.join(root, file)
            relpath = os.path.relpath(fullpath, target)
            label = os.path.basename(os.path.dirname(fullpath))
            images.append((relpath, label))
#obtains all relative image paths along with the labels for each image putting them together as

random.shuffle(images)
#shuffles the images

split = int(len(images) * 0.8)
training = images[:split]
testing = images[split:]
#splits the images into training and testing sets where training recieves 80% of the images while the remaining 20%
#is given to the testing set

trainingpathcsv = os.path.join(directory, 'training.csv')
with open(trainingpathcsv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['imagepath', 'label'])
    writer.writerows(training)
#saves the relative path and label of the training set as a csv file within the data folder

testingpathcsv = os.path.join(directory, 'testing.csv')
with open(testingpathcsv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['imagepath', 'label'])
    writer.writerows(testing)
#saves the relative path and label of the testing set as a csv file within the data folder