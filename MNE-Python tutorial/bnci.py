import numpy as np
import scipy.io as sio
from mne.filter import resample, filter_data
import mne
import matplotlib
import matplotlib.pyplot as plt
import copy

# Set this according to your dataset location.
DATASET_ROOT = "C:/Users/NMAIL/Desktop/BNCI dataset"


# 주어진 데이터셋을 EEG data & label로 처리하여 반환
# MI, 250Hz, C3,Cz,C4: 7,9,11 l/r hand: 0,1 foot, tongue: 2,3
# subj: 1-9, 0.5-100Hz, 10-20 system, returns 7.5s data, 2s before and 1.5s after the trial
def get_data_2a(subject, training, root_path=DATASET_ROOT+'/', sfreq=250):
    '''	Loads the dataset 2a of the BCI Competition IV
    available on http://bnci-horizon-2020.eu/database/data-sets

    Keyword arguments:
    subject -- number of subject in [1, .. ,9]
    training -- if True, load training data
                if False, load testing data

    Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1825
            class_return 	numpy matrix 	size = NO_valid_trial
    '''
    # two sessions per subject, trial: 4s, 250Hz
    sfreq = 250 if sfreq is None else sfreq

    # reference: left mastoid, ground: right mastoid
    # sampling rate: 250Hz
    # bandpassed: 0.5-100Hz
    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = int(7.5 * 250)

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, NO_channels, Window_Length))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(root_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(root_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0] # entire data (refer to comment before for loop)
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_fs = a_data3[3]
        a_classes = a_data3[4]
        a_artifacts = a_data3[5]
        a_gender = a_data3[6]
        a_age = a_data3[7]
        # a_trial.size > 0 means there is data in this run
        for trial in range(0, a_trial.size):
            # remove bad trials
            # if (a_artifacts[trial] == 0):
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + Window_Length), :NO_channels]) # 4s pre-trial and 4s post-trial
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    # index 500~1500 is the imagery time
    data = data_return[0:NO_valid_trial, :, :]
    if sfreq != 250:
        # low-pass filter
        data = filter_data(data, sfreq=250, l_freq=None, h_freq=sfreq / 2, verbose=False)
        # resample to sfreq
        data = resample(data, down=250 / sfreq, verbose=False)

    class_return = class_return[0:NO_valid_trial]-1

    return data, class_return


#EEG_to-epochs 함수에 사용되는 함수
def EEG_array_modifier(eeg_array, label_array):
    """
    Receives the EEG array and the label array to modify into a more suitable form to switch it into EpochsArray
    Used in EEG_to_epochs
    eeg_array and label_array elements should be in respective order

    :param eeg_array: (list) a list of lists that represent EEG of each trial
    :param label_array: (list) a list of labels that represent the label of each trial
    :return:
    """
    X,y,event_timepoints = [],[],[]
    for i,label in enumerate(label_array):
        X.append(np.array(eeg_array[i]))
        y.append(label)
        event_timepoints.append(i) # not sure what to do with this to be honest
    events_array = np.array([[event_timepoints[i],0,y[i]] for i in range(len(y))],int)
    return np.array(X), events_array

def EEG_to_epochs(eeg_array, label_array, sfreq=500, event_id = {'Left Hand':0,'Right Hand':1, 'Feet':2, 'Tongue':3}):
    
    channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'] # braindecode

    n_channels = len(channels)
    event_id = event_id
    ch_types = ['eeg'] * n_channels
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)
    info = info.set_montage(montage)
    data, events = EEG_array_modifier(eeg_array, label_array)
    print(data.shape)
    epochs = mne.EpochsArray(data, info, events, tmin=0, event_id=event_id)

    return epochs
