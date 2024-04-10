import os
from collections import Counter

import numpy as np
from sklearn.utils import shuffle
from PIL import Image, ImageEnhance

#set working directory
os.chdir('./src')


trainSetDirectory = './data/Training'
testSetDirectory =  './data/Testing'


def shuffleDataSet(directory):
    paths = []
    classifications = []
    for tumorType in os.listdir(directory):
        for img in os.listdir(f"{directory}/{tumorType}"):
            paths.append(f"{directory}/{tumorType}/{img}")
            classifications.append(tumorType)
    paths, classifications = shuffle(paths, classifications)
    return paths, classifications

trainingPaths, trainingLabels = [], []
testingPaths, testingLabels = [], []
trainingPaths, trainingLabels = shuffleDataSet(trainSetDirectory)
testingPaths, testingLabels, = shuffleDataSet(testSetDirectory)

#Count how many occurences of each tumor classification
tumorTypeCounts = Counter(trainingLabels)
totalTypes = len(trainingLabels)
percentages = {tumorType: count / totalTypes * 100 for tumorType, count in tumorTypeCounts.items()}
print(tumorTypeCounts)
for label, percentage in percentages.items():
    print(f"{label}: {percentage:.2f}%")
#Training set vs Testing Set
print(f"\nTraining Set: {len(trainingPaths)}\nTesting Set: {len(testingPaths)}")


    