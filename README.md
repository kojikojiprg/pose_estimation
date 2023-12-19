# Pose Estimation
This project can human detection, pose estimation, and tracking.
Human detection and pose estimation are implemented using [MMLab](https://github.com/open-mmlab) projects.
Tracking is implemented using [Unitrack](https://github.com/Zhongdao/UniTrack.git).

# Installation
```
bash setup.sh
```

# Inference
```
pose_estimation.py [-h] -vd VIDEO_DIR -od OUTPUT_DIR [-c CFG_PATH] [-g GPU] [-v]
```

optional arguments:  
  -h, --help            show this help message and exit  
  -vd VIDEO_DIR, --video_dir VIDEO_DIR  
                        path of input video directory  
  -od OUTPUT_DIR, --output_dir OUTPUT_DIR  
                        path of output data directory  
  -c CFG_PATH, --cfg_path CFG_PATH  
  -g GPU, --gpu GPU     gpu number  
  -v, --video           with writing video  
  
