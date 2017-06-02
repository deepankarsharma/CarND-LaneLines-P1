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

Pre generated images and videos are present under the test_images_output and test_videos_output folders.


### Examples

![Alt text](/test_images/solidWhiteCurve.jpg?raw=true "")
![Alt text](/test_images_output/solidWhiteCurve.jpg?raw=true "")
![Alt text](/test_images/solidWhiteRight.jpg?raw=true "")
![Alt text](/test_images_output/solidWhiteRight.jpg?raw=true "")
![Alt text](/test_images/solidYellowCurve.jpg?raw=true "")
![Alt text](/test_images_output/solidYellowCurve.jpg?raw=true "")
![Alt text](/test_images/solidYellowCurve2.jpg?raw=true "")
![Alt text](/test_images_output/solidYellowCurve2.jpg?raw=true "")
![Alt text](/test_images/solidYellowLeft.jpg?raw=true "")
![Alt text](/test_images_output/solidYellowLeft.jpg?raw=true "")
![Alt text](/test_images/whiteCarLaneSwitch.jpg?raw=true "")
![Alt text](/test_images_output/whiteCarLaneSwitch.jpg?raw=true "")

### Pipeline visualization

![Alt text](/debug/debug_solidWhiteCurve.jpg?raw=true "")
![Alt text](/debug/debug_solidWhiteRight.jpg?raw=true "")
![Alt text](/debug/debug_solidYellowCurve.jpg?raw=true "")
![Alt text](/debug/debug_solidYellowCurve2.jpg?raw=true "")
![Alt text](/debug/debug_solidYellowLeft.jpg?raw=true "")
![Alt text](/debug/debug_whiteCarLaneSwitch.jpg?raw=true "")