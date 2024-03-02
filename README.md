# Pose Estimation
This repository is used for our research on human pose estimation and tracking.
Human detection and pose estimation are implemented using [MMLab](https://github.com/open-mmlab) projects.
Tracking is implemented using [Unitrack](https://github.com/Zhongdao/UniTrack.git).

# Installation
```
bash setup.sh
```

# Usage
For videos stored in the VIDEO_DIR directory, the script performs pose estimation and tracking. The output results are saved in the following location: ```[VIDEO_DIR]/[FILE_NAME]/json/pose.json```, where FILE_NAME represents the video file name.

```
pose_estimation.py -vd VIDEO_DIR -od OUTPUT_DIR [-c CFG_PATH] [-g GPU] [-v]
```

optional arguments:
  - -vd VIDEO_DIR, --video_dir VIDEO_DIR
    path of input video directory
  - -od OUTPUT_DIR, --output_dir OUTPUT_DIR
    path of output data directory
  - -c CFG_PATH, --cfg_path CFG_PATH
  - -g GPU, --gpu GPU     gpu number
  - -v, --video           with writing video

# Output Format
The output format of the JSON file is as follows:
```json
[
  {
    "frame": frame number,
    "id": tracking id,
    "bbox": bounding box,
    "keypoints": keypoints,
  }
  ...
]
```
