import numpy as np
import pandas as pd
import os
import subprocess, pickle

def main():
    os.chdir('/Users/jyeon/Documents/Github/pilot_analysis/')
    dataPath = os.path.join(os.getcwd(), 'data/feven/walking_outdoor')
    cleanupPath = os.path.join(os.getcwd(), 'code/cleanup.py')

    # run cleanup 
    command = ["python", cleanupPath, dataPath]
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    pupil_data = pickle.load(open(os.path.join(dataPath, 'eyedata.pkl'), 'rb'))

    # compute gaze angle
    getgazePath = os.path.join(os.getcwd(), 'code/get_gaze.py')
    command = ["python", getgazePath, dataPath]
    subprocess.run(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    gaze_angles = pd.read_csv(os.path.join(dataPath, 'gaze_angle_relative_to_gravity.csv'))

    
    "TODO: Now plot gaze_angles and see how it has moved when the subject is walking"

if __name__ == "__main__":
    main()