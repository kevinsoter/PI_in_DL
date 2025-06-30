# PI_in_DL
The script provided here was written for the data analysis of the research manuscript:
    "Presynaptic inhibition does not mediate 
    reduced soleus H-reflex amplitudes during drop landings" 
     by Soter, K., Hahn, D. & Grosprêtre, S.

Abstract:
During drop landings, shortly after ground contact, spinal excitability is decreased. This decrease, as measured by soleus H-reflex, has been presumed, but not proven, to originate from presynaptic inhibition, facilitated by the descending drive from supraspinal centres. Therefore, the aim of this study was to examine presynaptic inhibition during the flight and landing phases of drop landings.
Fifteen participants received peripheral nerve stimulations during quiet stance and pre (PRE), and post (POST) ground contact of 40cm drop landings. Stimulations during drop landings were timed to elicit soleus H-reflexes 30-0ms PRE and 30-60ms POST landings, respectively. Presynaptic inhibition was assessed by conditioning the soleus H-reflex with femoral nerve stimulations, eliciting H-reflex heteronymous facilitation (HHF) and common fibular nerve stimulations, eliciting H-reflex D1 inhibition (HD1). Conditioned soleus H-reflex amplitudes were normalised to maximal M-waves (Mmax) and compared with the unconditioned H-reflexes (HTest) during quiet stance, PRE, and POST. EMG of soleus, medial gastrocnemius, tibialis anterior, and vastus medialis as well as hip, knee, and ankle joint angles were measured throughout drop landings and quiet stance. 
HTest POST was significantly smaller than PRE (-8.5% Mmax,  p = 0.016). Facilitation and inhibition were observed in quiet stance (HHF-HTest: +7.8% Mmax,  p < 0.001; HD1-HTest: -9.5%Mmax,  p = 0.003), but not during PRE or POST (all p = 1.000). 
 Both paradigms were effective in quiet stance, but not during drop landings, suggesting that the decreased soleus H-reflex POST is not due to ongoing presynaptic inhibition. Instead, reduced motoneuron excitability may indicate other underlying mechanisms.


Further information:
    - The file directories have to be inserted by hand at the appropriate placing
      as indicated in the script
	- The files directory should include a folder for each participant which
	  comprises all trials (example in the script)
	- Each trial has to be fully within one file (example for two participants
	  is provided in the github repository)
    - A 'configure file' with columns listing all participants pseudonyms, soleus
      H-reflex latencies and names of trials is needed
	- The columns should be seperated by a tab file
	- An example is provided in the github repository
    - Frequency used was 4000Hz (already filled in)
    - This also apllies to the kinematic data as it was automatically upsampled
      during data acquisition (actual sampling frequency was 148Hz)
    - Per subject and trial type three .txt files will be exported to 
      the working directory (mean values, SD values and results)

Familiarity with the manuscript is expected. Throughout different abbreviations
will be mentioned, sopme of these will be shortly reviewed here:
    - DT    = Drop Time (here synonymous with unstimulated trials)
    - QS    = Quiet Stance
    - H50   = Stimulation intensity so H-reflex is at 50% of ascending curve
    - sol   = Soleus muscle
    - gm    = Gastrocnemius medials
    - ta    = Tibialis anterior
    - vm    = Vastus medialis
    - D1    = Conditioning to induce D1 inhibition in sol H-reflex
    - HF    = Conditioning to induce heteronymous facilitation in sol H-reflex

Custom-written script by Soter, K.
Created on Tue Jan  9 09:50:36 2024 in Python (version 3.10.4) 
with Spyder IDE (version 5.4.3 standalone) 
