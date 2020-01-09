# Implementation TrafficSignRecognition for JetsonTX2
### Note: this repo is WIP!

This part splits into two sections: 
- one for the training and engine creation of the CNN-Classifier (MobileNetV1)
- the other one is the runtime code for the Nvidia Jetson TX2

## Current ToDo's
An overview of the current open points. 

!Complete scripts etc.!

### CNN Classifier
- implement EfficientNet (https://arxiv.org/abs/1905.11946)

### Jetson Runtime Code 
- improve ROI-Filtering 
- refactor inference for bigger batch size in order to detect faster more signs
- check if there is an possible benefit of using the shared memory between CPU/GPU

