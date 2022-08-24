# Classification NN MindWave Wheelchair

Neural Network for classify EEG signal to control wheelchair. Using Muse as EEG sensor with gamma and beta signal. Process EEG data then input to the NN below

![NN arch](https://user-images.githubusercontent.com/52401633/186486781-b40666a6-df82-4074-94dc-9fc21c3ed473.png)

Outputs are 5 classes (forward, turn left, turn right, backward, stop). The output class is sent to microcontroller to control the wheelchair
