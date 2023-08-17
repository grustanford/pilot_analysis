import sys, os, json, pickle
import numpy as np
import pandas as pd

def main():
    # check if the data directory is given
    if len(sys.argv) < 2:
        print("Usage: python3 cleanup.py <data_dir>")
        sys.exit(1)

    # load the data
    data_dir = sys.argv[-1]
    
    # load info.json
    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    # read gaze frequency from info.json
    gaze_freq = info["gaze_frequency"]
    note = list(info["template_data"]["data"].values())
    recording_date = info["template_data"]["recording_name"]
    wearer = info["wearer_name"]

    # read worldtime
    time_path = os.path.join(data_dir, 'world_timestamps.csv')
    time_data = pd.read_csv(time_path)
    recording_start = time_data['timestamp [ns]'].iloc[0]
    recording_end = time_data['timestamp [ns]'].iloc[-1]

    # read events
    event_path = os.path.join(data_dir, 'events.csv')
    event_data = pd.read_csv(event_path)

    # remove events unuseful
    valid_events = (event_data['name'] != 'recording.begin') & (event_data['name'] != 'recording.end')
    event_data = event_data[valid_events]

    # read gaze data
    gaze_path = os.path.join(data_dir, "gaze.csv")
    gaze_data = pd.read_csv(gaze_path)

    # read fixation data
    fixation_path = os.path.join(data_dir, "fixations.csv")
    fixation_data = pd.read_csv(fixation_path)
    
    # read IMU data
    imu_path = os.path.join(data_dir, "imu.csv")
    imu_data = pd.read_csv(imu_path)

    # find when event happened in gaze 
    gaze_event = pd.DataFrame(columns = ['event index', 'event name'], index = range(len(gaze_data)))
    fixation_event = pd.DataFrame(columns = ['event index', 'event name'], index = range(len(fixation_data)))
    imu_event = pd.DataFrame(columns = ['event index', 'event name'], index = range(len(imu_data)))

    for i in range(len(event_data)):
        event_index = i+1
        grab_an_event = [event_data['timestamp [ns]'].iloc[i], event_data['name'].iloc[i]]

        # find gaze index closest to the event time
        gaze_timedifference = gaze_data['timestamp [ns]'].sub(grab_an_event[0]).abs()
        gaze_filter = gaze_data['timestamp [ns]'] <= grab_an_event[0]
        index = gaze_timedifference[gaze_filter].idxmin()
        gaze_event.iloc[index] = {'event index': event_index, 'event name': grab_an_event[1]}
        
        # find fixation index closest to the event time
        fixation_timedifference = fixation_data['start timestamp [ns]'].sub(grab_an_event[0]).abs()
        fixation_filter = fixation_data['start timestamp [ns]'] <= grab_an_event[0]
        index = fixation_timedifference[fixation_filter].idxmin()
        fixation_event.iloc[index] = {'event index': event_index, 'event name': grab_an_event[1]}

        # find imu index closest to the event time
        imu_timedifference = imu_data['timestamp [ns]'].sub(grab_an_event[0]).abs()
        imu_filter = imu_data['timestamp [ns]'] <= grab_an_event[0]        
        if all(not x for x in imu_filter):
            message = ['event index:' + str(event_index) + ', event name: "' + grab_an_event[1] +'" not found in imu data']
            message = '\n'.join(message)
            note.append([message])
        else:            
            index = imu_timedifference[imu_filter].idxmin()
            imu_event.iloc[index] = {'event index': event_index, 'event name': grab_an_event[1]}

    gaze = pd.concat([gaze_data, gaze_event], axis=1)
    fixation = pd.concat([fixation_data, fixation_event], axis=1)
    imu = pd.concat([imu_data, imu_event], axis=1)

    # drop unuseful columns
    gaze = gaze.drop(columns = ["section id", "recording id", "blink id"])
    fixation = fixation.drop(columns = ["section id", "recording id"])
    imu = imu.drop(columns = ["section id", "recording id"])
    
    # data all together
    out = {"recording start": recording_start,
           "recording end": recording_end,
           "recording date": recording_date,
           "gaze frequency": gaze_freq, 
           "note": note,
           "wearer": wearer,
           "gaze": gaze,
           "fixation": fixation,
           "imu": imu,
           "events": event_data[['timestamp [ns]', 'name']]}
    
    file_path = os.path.join(data_dir, 'eyedata.pkl')
    with open(file_path, "wb") as pickle_file:
        pickle.dump(out, pickle_file)
    
if __name__ == "__main__":
    main()
    