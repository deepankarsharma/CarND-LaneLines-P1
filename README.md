#**Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

### Instructions on running the project

Usage: lane_finder.py [options]

Options:

  -h, --help                   show this help message and exit
  
  -i IFNAME, --ifname=IFNAME   Input filename to process
  
  -o OFNAME, --ofname=OFNAME   Output filename
  
  -v, --process-video          Incoming file is a video
  
  -d, --debug                  Enable debug output


Example invocations:
python lane_finder.py -i test_images/whiteCarLaneSwitch.jpg -o test_images_output/whiteCarLaneSwitch.jpg -d
python lane_finder.py -i test_videos/challenge.mp4 -o test_videos_output/challenge.mp4 -v

Writeup for this project is present at writeup.md

