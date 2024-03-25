# pollinator_detective: 
# detect and count pollinator visits from video frames

<img src="asserts/demo.png" /> </div>

<br>

## Contributors
- Dr. Erik Solhaug: project leader
- Ursina Baselgia: experiments and image annotation
- Hongyuan Zhang: code implementation

\*\*We are from Dr. Diana Santelia's Lab at ETH Zurich
<br>

## Working flow

### 1. Select region of interest (ROI) and track it with [CSRT tracker](https://docs.opencv.org/3.4/d2/da2/classcv_1_1TrackerCSRT.html)
This was used since the flower may dance with wind, and to reduce the field of view
<img src="asserts/csrt_tracker.gif" /> </div>
<br>


### 2. Detect pollinator within the ROI with [Swin-L-DINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/dino)
The pretrained catergories: Bumblebees, Flies, Honeybees, Hoverfly_A, Hoverfly_B, and Wildbees

<br>

 
## Data preparation
### 1. Data annotation was done with [ISAT_with_segment_anything](https://github.com/yatengLG/ISAT_with_segment_anything)
Consider trying our tool for our image annotation



### 2. Data clustering and sampling with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan)
To form a more balanced traning set by sampling each cluster
<img src="asserts/hdbcan.png" /> </div>



## Current bottleneck
- Unblanced datasets in terms of number of pollinators in each catergoreis and pixel sizes of them
- Hard to confirm if a pollinator is trully landed on the flower
- Misidentified pollinators such as ants and beetles
