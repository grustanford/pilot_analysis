import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pprint import pprint
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# dont need all the above libraries, I just pasted all of them in in case

# load in all gaze data csvs - zero preprocessing to remove noise
df = pd.read_csv('/Users/natashabanga/Documents/PsychSummer/pilot_analysis/all_conversation.csv')
patient = df.loc[df['SubjectID'] == 1]
control = df.loc[df['SubjectID'] == 0]


kf = KFold(n_splits=10)

# 100 patient samples, each with 1000 gaze angle values 
control_samples = [control.sample(n=1000, random_state = 25) for i in range(50)]
# total 200 patients, each with 1000 gaze values
patient_samples = [patient.sample(n=1000, random_state = 25) for i in range(50)]
allSamples = control_samples + patient_samples

# shuffle the data
random.seed(25)
random.shuffle(allSamples)
x = np.array([ch['Saccadic Amplitude'] for ch in allSamples])
y = np.array([ch['SubjectID'].iloc[0] for ch in allSamples])

# test train split (80-20), x values are each a series, y values are an int corresponding to control or patient
# train, test = train_test_split(allSamples, test_size=0.2, random_state=42)

# trainX = [ch['Outdoor Gaze Angle'] for ch in train]
# trainY = [ch['SubjectID'].iloc[0] for ch in train]

# testX = [ch['Outdoor Gaze Angle'] for ch in test]
# testY = [ch['SubjectID'].iloc[0] for ch in test]

# run svm!!
def svm(x_train, y_train, x_test, y_test):
    clf = SVC(kernel='linear') 
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    result = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(y_test, y_pred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(y_test, y_pred)
    print("Accuracy:",result2)

def crossVal():
    for train_index, test_index in kf.split(x):
        #print("TRAIN:", train_index, "TEST:", test_index)
        trainX, testX = x[train_index], x[test_index]
        trainY, testY = y[train_index], y[test_index]
        svm(trainX, trainY, testX, testY)


crossVal()