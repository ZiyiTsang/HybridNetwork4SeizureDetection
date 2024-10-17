# Implementation of Hybrid Network Design for Seizure Detection
This is the code-repo of my bachelor thesis.
## Abstract
Automatic seizure detection algorithms have become a research hotspot these years. As a key metric, detection latency is often neglected, which directly affects the
timeliness of medical intervention. This work developed three types of Feature
Extractors (FE), namely convolution (Conv), recurrent (Rec) and transformer (Transf)
for extracting EEG information, with two different time scales. Experiments show
that increasing the scale of the Conv and Rec FEs only marginally improves the
detection latency, with no such benefit seen for the Transf FE. Subsequently, our
work further introduced hybrid FE based on 2 individual FE. The result shows the
both the feature fusion and the shorter slicing windows will significantly improve
detection latency. The present method achieves 1.82 seconds in epilepsy detection
latency, which is the current SOTA score. Moreover, a closed-loop neuroimaging monitoring system (C-RNMS) is
implemented to solve the urgent need of epilepsy patients. Based on Arduino
hardware, this system can receive and process electrical signals in real time via a
mobile application, which meets the multiple needs of people with epilepsy and
lowers the threshold for them and their caregivers to access and monitor brain signals. The findings of this research are expected to provide new insights and ideas for
epilepsy detection.

## Code Structure
- Code: the code for hybrid model design
- Data: the download file will store at here. please download CHB-MIT dataset and place at here.
- Thesis: the proposal and thesis


