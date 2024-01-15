# PI_in_DL
Data analysis script for the research manuscript "Presynaptic Inhibition does not Contribute to Soleus Ia Afferent Depression during Drop Landings " by Soter, K., Hahn, D., and Grosprêtre, S.

The script provided here was written for the data analysis of the research manuscript:
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
    - The file directories have to be inserted by hand at the appropriate placing
      as indicated in the script
	- The files directory should include a folder for each participant which
	  comprises all trials (example in the script)
	- Each trial has to be fully within one file (example for two participants
	  is provided in the github repository)
    - A 'connfigure file' with columns listing all participants pseudonyms, soleus
      H-reflex latencies and names of trials is needed
	- The columns should be seperated by a tab file
	- An example is provided in the github repository
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
