# 2nd-year
This repository contains codes and pre-trained checkpoints of a place recognition model for Friends video.  

# Requirements
- Ubuntu (tested on v14.04)
- Python (tested on v2.7)
- Pytorch (tested on v0.4.1)
- Numpy
- PIL
- opencv-python
- matplotlib
- jsonl

# Download pre-trained checkpoint
Pre-trained checkpoint should be placed in the root directory.

You can download the pre-trained checkpoint from this [link](https://drive.google.com/file/d/17hNeJG5SV4NkLMO_GQRkx_fcE7MT1nxO/view?usp=sharing)


# How to use
Input: video file (e.g. *.avi, *.mkv) 

Output: jsonl file (predicted class of video frames for every second, 1 fps)

Below is an example of output jsonl flle. 

    {"second": 0.0, "type": "location", "class": "none"}
    {"second": 0.0, "type": "location", "class": "none"}
    ...
    {"second": 52.0, "type": "location", "class": "cafe"}
    {"second": 53.0, "type": "location", "class": "cafe"}
    ...
    {"second": 314.0, "type": "location", "class": "home-livingroom-Monica"}
    {"second": 315.0, "type": "location", "class": "home-livingroom-Monica"}
    ...

You can run following command on a terminal,

    python demo.py <video-file> <output-file-name>
    
For example 

    python demo.py input.avi output
    
Then output.jsonl file will be saved in the root directory


# Acknowledgements
This project was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
