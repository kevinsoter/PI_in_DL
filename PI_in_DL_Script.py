"""
This script was written for the data analysis of the research manuscript:
    "Presynaptic Inhibition does not Contribute to Soleus 
     Ia Afferent Depression during Drop Landings" 
     by Soter, K., Hahn, D. & Grosprêtre, S.

Short Study description:
    The present study was the first to assess whether presynaptic inhibition 
    suppresses soleus H-reflex responses during drop landings. The results 
    demonstrate a change in recruitment gain between quiet stance and landing 
    task, with no discernible influence of presynaptic inhibition throughout, 
    which contradicts earlier hypotheses. These findings suggest the potential
    modulation of distinct spinal pathways by the descending drive, notably 
    the recurrent inhibition pathway. Further research is required to 
    investigate the specific role of recurrent inhibition in H-reflex 
    depression during drop landings, using appropriate assessment techniques.

Further information:
    - Some information has to filled in by hand, the placing is clearly stated
    - A 'connfigure file' with columns listing all participants pseudonyms, soleus
      H-reflex latencies and name sof trials is needed
    - Frequency used was 4000Hz (already filled in)
    - This also apllies to the kinematic data as it was automatically upsampled
      durinng data acquisition (actual sampling frequency was 148Hz)
    - Per subject and trial of trial three .txt files will be exported to 
      the working directory (mean values, SD values and results)

Familiarity with the manuscript is expected. Throughout different abbreviations
will be mentioned, sopme of these will be shortly reviewed here:
    - DT    = Drop Time (here synonymous with unstimulated trials)
    - QS    = Quiet Stance
    - H50   = Stimulation Intensity so H-reflex is at 50% of ascending curve
    - sol   = Soleus muscle
    - gm    = Gastrocnemiu smedials
    - ta    = Tibialis anterior
    - vm    = Vastus medialis
    - D1    = Conditioning to induce D1 inhibition in sol H-reflex
    - HF    = Conditioning to induce heteronymous facilitation in sol H-reflex

Custom-written scripts by Soter, K.
Created on Tue Jan  9 09:50:36 2024
"""

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import math
###############################################################################
""" ! Fill in by hand ! """

config_dir = 'C:/User/Study/ConfigureFile.txt' # Example diretory for the configuratory file
file_dir   = 'C:/User/Study/Data/'             # Example directory of all files
freq       = 4000                              # Already filled in

""" ! No filling in by hand from here on ! """
###############################################################################
""" Defining all functions necessary for the analyses """
# Correcting data offset (gyroscope)
def offset(frequency, data):
    data_offset = ( 
        data[(round(frequency * 0.25)):-1] - 
        np.mean(data[round(frequency * 0.2):round(frequency * 0.3)])
        )
    return pd.Series(data_offset)

# Avoiding gyroscope drift by setting no velocity to actual zero
def zero_velocity(frequency, data): 
    new_data = []
    [upper, lower] = max(data[0:round(frequency * 0.75)]), \
                     min(data[0:round(frequency * 0.75)])
    for val in data:
        if val < upper and val > lower:
            new_data.append(0)
        else:
            new_data.append(val)
    return pd.Series(new_data)

# Integrating gyroscope velocity to body rotation
def rotation(data, frequency): 
    rotation = pd.Series(np.zeros(len(data)))
    for i in range(1, len(rotation)):
        rotation[i] = (
            ((((data[i] + data[i - 1])
            * 1 / frequency)) * 0.5) 
            + rotation[i - 1]
        )
    return rotation

# Calculating root mean square (RMS) 
def rms(array, length): 
    square = 0
    mean = 0.0
    root = 0.0
    for i in range(0, length):
        square += (array[i] ** 2)
    mean = (square / (float)(length))
    root = math.sqrt(mean)
    return root

# Saving analysed data in .txt file to working directory        
def save(obj, new_name):
    obj.to_csv(str(new_name) + str('.txt'), sep='\t', index=False, na_rep='nan')

# Filter for gyroscope data
cutoff = 20  # 20Hz cutoff based on Catalfamo et al., 2010
w = cutoff / (freq / 2)  # normalize frequency
b, a = signal.butter(2, w, 'low') # Forward & backward for 4th order
 
"""
Analysing all files of one trial of measurement:
In the following a for-loop is build up to loop through all subjects and
analyse all files and all trials of measurement (e.g. 'H50 D1 pre' relates to
the measurements during landing with D1 conditioning, meaning stimulations to
tibial and common fibular nerve).
"""
configure = pd.read_table(config_dir, delimiter='\t', decimal=',', 
                          header=None, low_memory=False, 
                          names=['subjects', 'H_latency', 'measurement'])
subjects = configure.subjects
H_latency = configure.H_latency
all_trials = configure.measurement.dropna()

# Organizing txt files: Data coming from both DELSYS and POWERLAB are
# originally in the same txt file, this is for dividing these later
delsys_columns = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 30, 31] 
powerlab_columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for t in range(0, len(all_trials)):
    trial = all_trials[t]   
    for s in range(0, len(subjects)):
        subject = subjects[s]
        H_timing = H_latency[s]
        file_name = file_dir+subject+'\\'+trial+'.txt'    
        H_peak = H_timing * 4 # matching the frequency of 4000
    
        """ Loading the txt file specified earlier """
        with open(file_name) as fp:
            lines = []
            for l_no, line in enumerate(fp):
                # search string
                if 'StartOfBlock' in line:
                    lines.append(l_no)
    
        data = {}
        for i in range(0, len(lines)):
            if i < len(lines) - 1:
                skipr = lines[i] + 4 
                nrows = lines[i + 1] - skipr - 2
            else:
                skipr = lines[i] + 4 
                nrows = None
    
            data["trial" + str(i + 1)] = pd.read_table(
             file_name, delimiter='\t', decimal=',', skiprows=skipr, header=None, 
             nrows=nrows, low_memory=False, 
             names=['time', 'sol_EMG', 'vm_EMG', 'ta_EMG', 'gm_EMG', 'stim_TN',
                    'stim_marker', 'JO_platf', 'GN_platf', 'JO_marker',
                    'GN_marker', 'foot1_x', 'foot1_y', 'foot1_z', 'shank2_x', 
                    'shank2_y', 'shank2_z', 'thigh7_x', 'thigh7_y', 'thigh7_z', 
                    'mag_foot_x', 'mag_foot_y', 'mag_foot_z', 'mag_shank_x',
                    'mag_shank_y', 'mag_shank_z', 'hip6_x', 'hip6_y', 'hip6_z',
                    'platf_x', 'platf_y', 'platf_z', 'event_marker']
             )
            
        """ 
        Synchronising DELSYS and POWERLAB data:
        Due to the integration of DELSYS data into POWERLAB via bluetooth and 
        LABCHARTS inability to synchronize these, this had to be done by hand.
        How it works (assuming experimental setup is known):
        Usage of a cablebound goniometer and a bluetooth gyroscope (DELSYS) on the 
        same part of the jump-off-board. Maximal displacement of the goniometer 
        should logically happen at the same time as maximal velocity of the 
        gyroscope.
        """
        DT_cut_data = {} # Defining DataFrames
        final_data = {}
        btw_data = {}
        cut_data = {}
        cut_data_full = {}
        QS_data = {}        # Only needed for QS data
        QS_data_cut = {}    # Only needed for QS data
        sol_RMS = []
        gm_RMS = []
        ta_RMS = []
        vm_RMS = []
        mwave = []
            
        for i in range(0, len(lines)): # Preparing DataFrames
            DT_cut_data["trial" + str(i + 1)] = pd.DataFrame()
            final_data["trial" + str(i + 1)] = pd.DataFrame()
            btw_data["trial" + str(i + 1)] = pd.DataFrame()
            cut_data["trial" + str(i + 1)] = pd.DataFrame()
            cut_data_full["trial" + str(i + 1)] = pd.DataFrame()
            QS_data["trial" + str(i + 1)] = pd.DataFrame()      # Only needed for QS
            QS_data_cut["trial" + str(i + 1)] = pd.DataFrame()  # Only needed for QS
        
        # Treating the raw data
        for j in range(0, len(lines)):
            for k in range(0, len(powerlab_columns)):
                X = data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                   .columns[powerlab_columns[k]])]
                X = (
                    X[(round(freq * 0.25)):-1].reset_index(drop=True)
                    )
                btw_data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                   .columns[powerlab_columns[k]])] = X
            for k in range(0, len(delsys_columns)):
                X = data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                   .columns[delsys_columns[k]])]                
                if k < 18:  # Only filtering body part gyroscope data NOT platform
                    X = zero_velocity(
                        freq, offset(freq, (signal.filtfilt(b, a, X)))
                        )
                else:
                    X = zero_velocity(freq, offset(freq, X))
                btw_data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                   .columns[delsys_columns[k]])] = X
    
            # Calculating the time-shift between goniometer and gyroscope
            gonio = btw_data['trial' + str(j + 1)].JO_platf 
            gyro  = btw_data['trial' + str(j + 1)].platf_y
    
            gonio = gonio.reset_index(drop=True)
            gyro = gyro.reset_index(drop=True)
            JO_peaks = sp.signal.find_peaks((gonio * -1), distance=300, height=0.2)
            min_gonio = JO_peaks[0][0]
            max_vel = gyro.argmax()
            X = pd.Series(np.zeros(len(gyro)))
            intersec = (
                        np.argwhere(np.diff(np.sign(gyro[max_vel:-1]
                                            - X[max_vel:-1]))).flatten() + max_vel
                        )
            shift = min_gonio - intersec[0] # This var can now be used for synching
    
            # Since the time-shift was non-systemic a shift to the right (>0) 
            #                      but also to the left (<0) had to be considered.
            trial_sync = pd.DataFrame() # Defining DataFrame for synchronised data
            if shift < 0:
                for k in range(0, len(powerlab_columns)):
                    X = btw_data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                              .columns[powerlab_columns[k]])]
                    X = X[0:-1 - abs(shift)]
                    X = X.reset_index(drop=True)
                    trial_sync[str(data['trial' + str(j + 1)]
                                   .columns[powerlab_columns[k]])] = X
                for i in range(0, len(delsys_columns)):
                    X = btw_data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                           .columns[delsys_columns[i]])]
                    X = X[abs(shift):-1]
                    X = X.reset_index(drop=True)
                    trial_sync[str(data['trial' + str(j + 1)]
                                   .columns[delsys_columns[i]])] = X
            else:
                for k in range(0, len(powerlab_columns)):
                    X = btw_data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                           .columns[powerlab_columns[k]])]
                    X = X[abs(shift):-1]
                    X = X.reset_index(drop=True)
                    trial_sync[str(data['trial' + str(j + 1)]
                                   .columns[powerlab_columns[k]])] = X
                for i in range(0, len(delsys_columns)):
                    X = data['trial' + str(j + 1)][str(data['trial' + str(j + 1)]
                                                       .columns[delsys_columns[i]])]
                    X = X[0:-1 - abs(shift)]
                    X = X.reset_index(drop=True)
                    trial_sync[str(data['trial' + str(j + 1)]
                                   .columns[delsys_columns[i]])] = X
    
            """ 
            Calculation of joint rotations based on assumption of sagittal
            plane movement:
            1. Extracting the rotational velocity of each body part
            2. Calculating resultant velocity from x, y & z directions
            3. Calculation of rotation of each body segment
            4. Calculation of joint angles by subtracting the rotation of the
               body segments making up this joint 
               (e.g. knee rotation = shank rotation - thigh rotation)
            """
            # 1.
            gyro_hip_x = trial_sync.hip6_x  # hip
            gyro_hip_y = trial_sync.hip6_y  # hip
            gyro_hip_z = trial_sync.hip6_z  # hip
            gyro_thigh_x = trial_sync.thigh7_x  # thigh
            gyro_thigh_y = trial_sync.thigh7_y  # thigh
            gyro_thigh_z = trial_sync.thigh7_z  # thigh
            gyro_shank_x = trial_sync.shank2_x  # shank
            gyro_shank_y = trial_sync.shank2_y  # shank
            gyro_shank_z = trial_sync.shank2_z  # shank
            gyro_foot_x = trial_sync.foot1_x  # foot
            gyro_foot_y = trial_sync.foot1_y  # foot
            gyro_foot_z = trial_sync.foot1_z  # foot
            # 2.
            res_vel_hip = np.sqrt(
                (gyro_hip_x ** 2) + (gyro_hip_y ** 2) + (gyro_hip_z ** 2)
                ) * np.sign(gyro_hip_x)
            res_vel_thigh = np.sqrt(
                (gyro_thigh_x ** 2) + (gyro_thigh_y ** 2) + (gyro_thigh_z ** 2)
                ) * np.sign(gyro_thigh_x)
            res_vel_shank = np.sqrt(
                (gyro_shank_x ** 2) + (gyro_shank_y ** 2) + (gyro_shank_z ** 2)
                ) * np.sign(gyro_shank_x)
            res_vel_foot = np.sqrt(
                (gyro_foot_x ** 2) + (gyro_foot_y ** 2) + (gyro_foot_z ** 2)
                ) * np.sign(gyro_foot_x)
            # 3.
            res_rot_hip = rotation(res_vel_hip, freq)
            res_rot_thigh = rotation(res_vel_thigh, freq)
            res_rot_shank = rotation(res_vel_shank, freq)
            res_rot_foot = rotation(res_vel_foot, freq)
            # 4.
            final_data['trial' + str(j + 1)][str('rotation_hip')] = (res_rot_thigh - res_rot_hip)
            final_data['trial' + str(j + 1)][str('rotation_knee')] = (res_rot_shank - res_rot_thigh)
            final_data['trial' + str(j + 1)][str('rotation_ankle')] = (res_rot_foot - res_rot_shank)
            # Adding all other data into DataFrame "final_data"
            final_data['trial' + str(j + 1)][str('JO_platf')] = trial_sync.JO_platf
            final_data['trial' + str(j + 1)][str('GN_platf')] = trial_sync.GN_platf
            final_data['trial' + str(j + 1)][str('stim_TN')] = trial_sync.stim_TN
            final_data['trial' + str(j + 1)][str('sol_EMG')] = trial_sync.sol_EMG
            final_data['trial' + str(j + 1)][str('vm_EMG')] = trial_sync.vm_EMG
            final_data['trial' + str(j + 1)][str('ta_EMG')] = trial_sync.ta_EMG
            final_data['trial' + str(j + 1)][str('gm_EMG')] = trial_sync.gm_EMG
            final_data['trial' + str(j + 1)][str('vel_hip')] = (res_vel_thigh - res_vel_hip)
            final_data['trial' + str(j + 1)][str('vel_knee')] = (res_vel_shank - res_vel_thigh)
            final_data['trial' + str(j + 1)][str('vel_ankle')] = (res_vel_foot - res_vel_shank)
            
            """ 
            Sorting data into DataFrames fitting for further analyses:
            1. Centered around the ground contact (GC) with -1.5s & +1s (cut_data_full)
            2. Centered around the GC with +/- 100ms (cut_data)
            3. Specifically for further calculations for Background EMG in DT trials:
                30ms window around timepoint of H-reflex occurrence in stimulated trials
            """
            if trial == 'H50 QS' or trial == 'D1' or trial == 'HF':
                QS_data['trial' + str(j + 1)][str('sol_EMG')] = data['trial' + str(j + 1)].sol_EMG
                QS_data['trial' + str(j + 1)][str('vm_EMG')] = data['trial' + str(j + 1)].vm_EMG
                QS_data['trial' + str(j + 1)][str('ta_EMG')] = data['trial' + str(j + 1)].ta_EMG
                QS_data['trial' + str(j + 1)][str('gm_EMG')] = data['trial' + str(j + 1)].gm_EMG
                QS_data['trial' + str(j + 1)][str('stim_TN')] = data['trial' + str(j + 1)].stim_TN
            
                stim_QS = next(x for x, val in enumerate(QS_data['trial' + str(j + 1)].stim_TN) if val > 0.1)
            
                QS_data_cut['trial' + str(j + 1)][str('sol_EMG')] = QS_data['trial' + str(j + 1)].sol_EMG[
                                                                       stim_QS - 200:stim_QS + 440].reset_index(drop=True) #before stim-220:stim-20
                QS_data_cut['trial' + str(j + 1)][str('vm_EMG')] = QS_data['trial' + str(j + 1)].vm_EMG[
                                                                      stim_QS - 320:stim_QS - 280].reset_index(drop=True)
                QS_data_cut['trial' + str(j + 1)][str('ta_EMG')] = QS_data['trial' + str(j + 1)].ta_EMG[
                                                                      stim_QS - 320:stim_QS - 280].reset_index(drop=True)
                QS_data_cut['trial' + str(j + 1)][str('gm_EMG')] = QS_data['trial' + str(j + 1)].gm_EMG[
                                                                      stim_QS - 320:stim_QS - 280].reset_index(drop=True)
            else:
                GC = next(x for x, val in enumerate(final_data['trial' + str(j + 1)].GN_platf) if val > 0.01)
                ### 1. Full data (all trials chosen trial), meaning +1s & -1.5s ###
                cut_data_full['trial' + str(j + 1)][str('rotation_hip')] = final_data['trial' + str(j + 1)].rotation_hip[
                                                                           GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('rotation_knee')] = final_data['trial' + str(j + 1)].rotation_knee[
                                                                            GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('rotation_ankle')] = final_data['trial' + str(j + 1)].rotation_ankle[
                                                                             GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('sol_EMG')] = final_data['trial' + str(j + 1)].sol_EMG[
                                                                      GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('vm_EMG')] = final_data['trial' + str(j + 1)].vm_EMG[
                                                                     GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('ta_EMG')] = final_data['trial' + str(j + 1)].ta_EMG[
                                                                     GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('gm_EMG')] = final_data['trial' + str(j + 1)].gm_EMG[
                                                                     GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('JO_platf')] = final_data['trial' + str(j + 1)].JO_platf[
                                                                     GC - 6000:GC + 4000].reset_index(drop=True)
                cut_data_full['trial' + str(j + 1)][str('stim_TN')] = final_data['trial' + str(j + 1)].stim_TN[
                                                                       GC - 6000:GC + 4000].reset_index(drop=True)
                ### 2. Cut data (all trials chosen trial), meaning +/- 100ms ###
                cut_data['trial' + str(j + 1)][str('rotation_hip')] = final_data['trial' + str(j + 1)].rotation_hip[
                                                                      GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('rotation_knee')] = final_data['trial' + str(j + 1)].rotation_knee[
                                                                       GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('rotation_ankle')] = final_data['trial' + str(j + 1)].rotation_ankle[
                                                                        GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('sol_EMG')] = final_data['trial' + str(j + 1)].sol_EMG[
                                                                 GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('vm_EMG')] = final_data['trial' + str(j + 1)].vm_EMG[
                                                                GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('ta_EMG')] = final_data['trial' + str(j + 1)].ta_EMG[
                                                                GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('gm_EMG')] = final_data['trial' + str(j + 1)].gm_EMG[
                                                                GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('vel_hip')] = final_data['trial' + str(j + 1)].vel_hip[
                                                                 GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('vel_knee')] = final_data['trial' + str(j + 1)].vel_knee[
                                                                  GC - 400:GC + 400].reset_index(drop=True)
                cut_data['trial' + str(j + 1)][str('vel_ankle')] = final_data['trial' + str(j + 1)].vel_ankle[
                                                                   GC - 400:GC + 400].reset_index(drop=True)        
                ### 3. DataFrame specifically for bEMG RMS calculations in DT trials ###
                DT_cut_data['trial' + str(j + 1)][str('sol_EMG_pre')] = cut_data['trial' + str(j + 1)].sol_EMG[
                                                                    279-H_peak:399-H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('vm_EMG_pre')] = cut_data['trial' + str(j + 1)].vm_EMG[
                                                                    279 - H_peak:399 - H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('ta_EMG_pre')] = cut_data['trial' + str(j + 1)].ta_EMG[
                                                                    279 - H_peak:399 - H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('gm_EMG_pre')] = cut_data['trial' + str(j + 1)].gm_EMG[
                                                                    279 - H_peak:399 - H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('sol_EMG_post')] = cut_data['trial' + str(j + 1)].sol_EMG[
                                                                        519 - H_peak:639 - H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('vm_EMG_post')] = cut_data['trial' + str(j + 1)].vm_EMG[
                                                                        519 - H_peak:639 - H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('ta_EMG_post')] = cut_data['trial' + str(j + 1)].ta_EMG[
                                                                        519 - H_peak:639 - H_peak].reset_index(drop=True)
                DT_cut_data['trial' + str(j + 1)][str('gm_EMG_post')] = cut_data['trial' + str(j + 1)].gm_EMG[
                                                                        519 - H_peak:639 - H_peak].reset_index(drop=True)
                    
                """
                EMG and M-Wave (due to CFN or FN stimulation) in a 50ms window 
                ending 5ms prior to stimulation instant during stimulated trials 
                (D1 Pre & POST, HF Pre & Post and Unconditioned).
                In HF & unconditioned trials this means 55ms to 5ms before stimulation 
                of TN; In D1 trials this means 80ms to 30ms before stimulation of TN as
                the FN stiumulation happened in the 30-0ms period and also elicits an
                EMG spike in SOL.
                """
                if trial == 'Drop Time' or trial == 'H50 QS':
                   pass # To ensure unstimulated measurements don't get analysed here         
                elif trial == 'D1':
                    stim = next(x for x, val in enumerate(QS_data['trial' + str(j + 1)].stim_TN) if val > 0.1)
                    mwave.append(np.ptp(QS_data['trial' + str(j + 1)].ta_EMG[stim-52:stim-4]))
                elif trial == 'HF':
                    stim = next(x for x, val in enumerate(QS_data['trial' + str(j + 1)].stim_TN) if val > 0.1)
                    mwave.append(np.ptp(QS_data['trial' + str(j + 1)].vm_EMG[stim+40:stim+240]))
                elif trial == 'H50 D1 pre' or trial == 'H50 D1 post':
                    stim = next(x for x, val in enumerate(cut_data_full['trial' + str(j + 1)].stim_TN) if val > 0.1)
                    sol_RMS.append(rms(cut_data_full['trial' + str(j + 1)].sol_EMG[stim - 320:stim - 120].reset_index(drop=True), 200))
                    gm_RMS.append(rms(cut_data_full['trial' + str(j + 1)].gm_EMG[stim - 320:stim - 120].reset_index(drop=True), 200))
                    ta_RMS.append(rms(cut_data_full['trial' + str(j + 1)].ta_EMG[stim - 320:stim - 120].reset_index(drop=True), 200))
                    vm_RMS.append(rms(cut_data_full['trial' + str(j + 1)].vm_EMG[stim - 320:stim - 120].reset_index(drop=True), 200))
                    mwave.append(np.ptp(cut_data_full['trial' + str(j + 1)].ta_EMG[stim-52:stim-4]))
                else: # This also calculates M-wave in uncon.; Can be deleted later on
                    stim = next(x for x, val in enumerate(cut_data_full['trial' + str(j + 1)].stim_TN) if val > 0.1)
                    sol_RMS.append(rms(cut_data_full['trial' + str(j + 1)].sol_EMG[stim - 220:stim - 20].reset_index(drop=True), 200))
                    gm_RMS.append(rms(cut_data_full['trial' + str(j + 1)].gm_EMG[stim - 220:stim - 20].reset_index(drop=True), 200))
                    ta_RMS.append(rms(cut_data_full['trial' + str(j + 1)].ta_EMG[stim - 220:stim - 20].reset_index(drop=True), 200))
                    vm_RMS.append(rms(cut_data_full['trial' + str(j + 1)].vm_EMG[stim - 220:stim - 20].reset_index(drop=True), 200))
                    mwave.append(np.ptp(cut_data_full['trial' + str(j + 1)].vm_EMG[stim+40:stim+240]))
                
        """
        Calculating mean values and standard deviation of the whole dataset, 
        meaning -1.5s & +1s around GC from cut_data_full
        """
        gathered_data_full = {}
        for i in range(0, len(cut_data_full['trial1'].columns)):
            gathered_data_full[cut_data_full['trial1'].columns[i]] = pd.DataFrame()
        for i in range(0, len(cut_data_full)):
            gathered_data_full['rotation_hip']['rot_hip' + str(i + 1)] = cut_data_full['trial' + str(i + 1)].loc[:,
                                                                      'rotation_hip']  # first (0) column is hip
            gathered_data_full['rotation_knee']['rot_knee' + str(i + 1)] = cut_data_full['trial' + str(i + 1)].loc[:,
                                                                        'rotation_knee']
            gathered_data_full['rotation_ankle']['rot_ankle' + str(i + 1)] = cut_data_full['trial' + str(i + 1)].loc[:,
                                                                          'rotation_ankle']
            gathered_data_full['sol_EMG']['sol_EMG' + str(i + 1)] = rms(cut_data_full['trial' + str(i + 1)].loc[:, 'sol_EMG'], len(cut_data_full['trial' + str(i + 1)].loc[:, 'sol_EMG']))
            gathered_data_full['vm_EMG']['vm_EMG' + str(i + 1)] = rms(cut_data_full['trial' + str(i + 1)].loc[:, 'vm_EMG'], len(cut_data_full['trial' + str(i + 1)].loc[:, 'vm_EMG']))
            gathered_data_full['ta_EMG']['ta_EMG' + str(i + 1)] = rms(cut_data_full['trial' + str(i + 1)].loc[:, 'ta_EMG'], len(cut_data_full['trial' + str(i + 1)].loc[:, 'ta_EMG']))
            gathered_data_full['gm_EMG']['gm_EMG' + str(i + 1)] = rms(cut_data_full['trial' + str(i + 1)].loc[:, 'gm_EMG'], len(cut_data_full['trial' + str(i + 1)].loc[:, 'gm_EMG']))
    
        mean_values_full = pd.DataFrame()
        std_values_full = pd.DataFrame()
        for i in range(0, len(cut_data_full['trial1'].columns)):
            mean_values_full[cut_data_full['trial1'].columns[i]] = np.mean(gathered_data_full[cut_data_full['trial1'].columns[i]],axis=1)
            std_values_full[cut_data_full['trial1'].columns[i]] = np.std(gathered_data_full[cut_data_full['trial1'].columns[i]],axis=1)        
    
        """
        Gathering and forming mean values of bEMG RMS values and kinematic data 
        (ankle, knee & hip joint) for quiet stance, unstimulated and 
        stimulated trials:
            1st Part for quiet stance trials
            2nd Part for unstimulated trials
            3rd Part for stimulated trials
        """
        ### 1st Part ###
        if trial == 'H50 QS':
            gathered_QS = {}
            for i in range(0, len(QS_data_cut['trial1'].columns)):
                gathered_QS[QS_data_cut['trial1'].columns[i]] = []
            for i in range(0, len(QS_data_cut)):
                gathered_QS['sol_EMG'].append(rms(QS_data_cut['trial' + str(i + 1)].sol_EMG, 200))
                gathered_QS['vm_EMG'].append(rms(QS_data_cut['trial' + str(i + 1)].vm_EMG, 200))
                gathered_QS['ta_EMG'].append(rms(QS_data_cut['trial' + str(i + 1)].ta_EMG, 200))
                gathered_QS['gm_EMG'].append(rms(QS_data_cut['trial' + str(i + 1)].gm_EMG, 200))
            
            mean_values_QS = pd.DataFrame()
            std_values_QS = pd.DataFrame()
            for i in range(0, len(QS_data_cut['trial1'].columns)):
                mean_values_QS[QS_data_cut['trial1'].columns[i]] = np.mean(gathered_QS[QS_data_cut['trial1'].columns[i]], axis=1)
                std_values_QS[QS_data_cut['trial1'].columns[i]] = np.std(gathered_QS[QS_data_cut['trial1'].columns[i]], axis=1)     
        ### 2nd Part ###
        if trial == 'Drop Time':
            gathered_data_EMG = {}
            for i in range(0, len(DT_cut_data['trial1'].columns)):
                gathered_data_EMG[DT_cut_data['trial1'].columns[i]] = []
            for i in range(0, len(DT_cut_data)):
                gathered_data_EMG['sol_EMG_pre'].append(rms(DT_cut_data['trial' + str(i + 1)].sol_EMG_pre, 120))
                gathered_data_EMG['vm_EMG_pre'].append(rms(DT_cut_data['trial' + str(i + 1)].vm_EMG_pre, 120))
                gathered_data_EMG['ta_EMG_pre'].append(rms(DT_cut_data['trial' + str(i + 1)].ta_EMG_pre, 120))
                gathered_data_EMG['gm_EMG_pre'].append(rms(DT_cut_data['trial' + str(i + 1)].gm_EMG_pre, 120))
                gathered_data_EMG['sol_EMG_post'].append(rms(DT_cut_data['trial' + str(i + 1)].sol_EMG_post, 120))
                gathered_data_EMG['vm_EMG_post'].append(rms(DT_cut_data['trial' + str(i + 1)].vm_EMG_post, 120))
                gathered_data_EMG['ta_EMG_post'].append(rms(DT_cut_data['trial' + str(i + 1)].ta_EMG_post, 120))
                gathered_data_EMG['gm_EMG_post'].append(rms(DT_cut_data['trial' + str(i + 1)].gm_EMG_post, 120))
        
            gathered_data_kin = {}
            for i in range(0, len(cut_data['trial1'].columns)):
                gathered_data_kin[cut_data['trial1'].columns[i]] = pd.DataFrame()
            for i in range(0, len(cut_data)):
                gathered_data_kin['rotation_hip']['rot_hip' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:,'rotation_hip']  # first (0) column is hip
                gathered_data_kin['rotation_knee']['rot_knee' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'rotation_knee']
                gathered_data_kin['rotation_ankle']['rot_ankle' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'rotation_ankle']
                gathered_data_kin['vel_hip']['vel_hip' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'vel_hip']
                gathered_data_kin['vel_knee']['vel_knee' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'vel_knee']
                gathered_data_kin['vel_ankle']['vel_ankle' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'vel_ankle']
    
            mean_values = pd.DataFrame()
            std_values = pd.DataFrame()
            for i in range(0, len(cut_data['trial1'].columns)):
                mean_values[cut_data['trial1'].columns[i]] = np.mean(gathered_data_kin[cut_data['trial1'].columns[i]], axis=1)
                std_values[cut_data['trial1'].columns[i]] = np.std(gathered_data_kin[cut_data['trial1'].columns[i]], axis=1)    
        ### 3rd Part ###
        else:
            gathered_data_kin = {}
            for i in range(0, len(cut_data['trial1'].columns)):
                gathered_data_kin[cut_data['trial1'].columns[i]] = pd.DataFrame()
            for i in range(0, len(cut_data)):
                gathered_data_kin['rotation_hip']['rot_hip' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:,'rotation_hip']  # first (0) column is hip
                gathered_data_kin['rotation_knee']['rot_knee' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'rotation_knee']
                gathered_data_kin['rotation_ankle']['rot_ankle' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'rotation_ankle']
                gathered_data_kin['vel_hip']['vel_hip' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'vel_hip']
                gathered_data_kin['vel_knee']['vel_knee' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'vel_knee']
                gathered_data_kin['vel_ankle']['vel_ankle' + str(i + 1)] = cut_data['trial' + str(i + 1)].loc[:, 'vel_ankle']
    
            mean_values = pd.DataFrame()
            std_values = pd.DataFrame()
            for i in range(0, len(cut_data['trial1'].columns)):
                mean_values[cut_data['trial1'].columns[i]] = np.mean(gathered_data_kin[cut_data['trial1'].columns[i]], axis=1)
                std_values[cut_data['trial1'].columns[i]] = np.std(gathered_data_kin[cut_data['trial1'].columns[i]], axis=1)
                
        """
        Results:
        Gathering all results of one subjects specific trial of trials (e.g. drop 
        landings with stimulation to TN and CFN prior to GC [meaning D1 pre]) in
        one DataFrame for later export.
        """
        if trial == 'H50 QS':
            results_QS = pd.DataFrame()
            results_QS['variable'] = (
                'sol_EMG', 'gm_EMG', 'ta_EMG', 'vm_EMG'
                )
            results_QS['mean'] = (
                np.mean(gathered_QS['sol_EMG'], np.mean(gathered_QS['gm_EMG']),
                np.mean(gathered_QS['ta_EMG']), np.mean(gathered_QS['vm_EMG']))
                )
        elif trial == 'D1' or trial == 'HF':
            results_QS = pd.DataFrame()
            results_QS['variable'] = ('Mwave')
            results_QS['mean'] = np.mean(mwave)
        if trial == 'Drop Time': # Only for unstimulated trials
            results_unstim = pd.DataFrame()
            results_unstim['variable'] = (
                'sol_RMS_pre', 'sol_RMS_post', 'gm_RMS_pre', 'gm_RMS_post', 
                'ta_RMS_pre', 'ta_RMS_post', 'vm_RMS_pre', 'vm_RMS_post', 'vel_hip',
                'vel_knee', 'vel_ankle', 'hip_GC', 'knee_GC', 'ankle_GC',
                )
            results_unstim['mean'] = (
                np.mean(gathered_data_EMG['sol_EMG_pre']), 
                np.mean(gathered_data_EMG['sol_EMG_post']), 
                np.mean(gathered_data_EMG['gm_EMG_pre']), 
                np.mean(gathered_data_EMG['gm_EMG_post']),
                np.mean(gathered_data_EMG['ta_EMG_pre']), 
                np.mean(gathered_data_EMG['ta_EMG_post']), 
                np.mean(gathered_data_EMG['vm_EMG_pre']), 
                np.mean(gathered_data_EMG['vm_EMG_post']),
                np.max(abs(mean_values['vel_hip'][400:-1])), 
                np.max(abs(mean_values['vel_knee'][400:-1])),
                np.max(abs(mean_values['vel_ankle'][400:-1])),
                (mean_values['rotation_hip'][400] * -1),
                (mean_values['rotation_knee'][400]),
                (mean_values['rotation_ankle'][400] * -1)
                )
        else: # Only for stimulated trials (incl. joint angle change around stim.)
            results_stim = pd.DataFrame()
            results_stim['variable'] = (
                'sol_RMS', 'gm_RMS', 'ta_RMS', 'vm_RMS', 'Mwave', 'vel_hip',
                'vel_knee', 'vel_ankle', 'hip_GC', 'knee_GC', 'ankle_GC',
                'hip_diff(stim1)', 'knee_diff(stim1)', 'ankle_diff(stim1)',
                'hip_diff(stim2)', 'knee_diff(stim2)', 'ankle_diff(stim2)'
                )
            results_stim['mean'] = (
                np.mean(sol_RMS), np.mean(gm_RMS), np.mean(ta_RMS), 
                np.mean(vm_RMS), np.mean(mwave),
                np.max(abs(mean_values['vel_hip'][400:-1])), 
                np.max(abs(mean_values['vel_knee'][400:-1])),
                np.max(abs(mean_values['vel_ankle'][400:-1])),
                (mean_values['rotation_hip'][400] * -1),
                (mean_values['rotation_knee'][400]),
                (mean_values['rotation_ankle'][400] * -1),
                (mean_values['rotation_hip'][400 - H_peak] * -1) - (mean_values['rotation_hip'][280 - H_peak] * -1),
                (mean_values['rotation_knee'][400 - H_peak]) - (mean_values['rotation_knee'][280 - H_peak]),
                (mean_values['rotation_ankle'][400 - H_peak] * -1) - (mean_values['rotation_ankle'][280 - H_peak] * -1),
                (mean_values['rotation_hip'][640 - H_peak] * -1) - (mean_values['rotation_hip'][520 - H_peak] * -1),
                (mean_values['rotation_knee'][640 - H_peak]) - (mean_values['rotation_knee'][520 - H_peak]),
                (mean_values['rotation_ankle'][640 - H_peak] * -1) - (mean_values['rotation_ankle'][520 - H_peak] * -1)
                )
    
        """" Saving """
        if trial == 'H50 QS' or trial == 'D1' or trial == 'HF':
            save(results_QS, subject + '_' + trial + '_results')
            save(mean_values_QS, subject + '_' + trial + '_mean')
            save(std_values_QS, subject + '_' + trial + '_SD')
        else:
            save(mean_values_full, subject + '_' + trial + '_full_mean')
            save(std_values_full, subject + '_' + trial + '_full_SD')
            if trial == 'Drop Time':
                save(results_unstim, subject + '_' + trial + '_results')
            else:
                save(results_stim, subject + '_' + trial + '_results')