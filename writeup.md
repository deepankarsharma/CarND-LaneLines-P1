# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

Pipeline contains the following steps

1. Image converted to hsv
2. Mask created for detecting yellow parts of the image using hsv image.
3. Mask created for detecting white parts of the image using hsv image.
4. Combined mask from 2 and 3 created
5. Grayscale image created from original image
6. Mask from 4 applied to the grayscale image
7. Gaussian blur applied to masked image
8. Canny filter applied
9. Region of interest used to narrow down the part of the image to be examined
10. Hough transform used to detect lines
11. Slope is calculated for lines. Lines that have an absolute slope less than a threshold are discarded. Lines with infinite slope discarded as well.
11. Lines are divided into lines with positive slope and lines with negative slope.
12. Use numpy polyfit to fit a single line from all lines with positive slope and another one from lines with negative slope.
13. Draw the lines to mark the lanes


### Examples

![Alt text](/test_images/solidWhiteCurve.jpg?raw=true "solidWhiteCurve")
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


### 2. Identify potential shortcomings with your current pipeline

1. Current pipeline uses slope filtering of lines, this works well when the car is in the normal state of driving sensibly in a lane. But will break down when the car is driving at near perpendicular angles to the lanes.
2. Detecting lanes solely based on colors opens the door for random white / yellow objects to throw the pipeline off
3. If car reaches an alignment where both left and right lane have a slope that is either both positive or both negative, the pipeline will break down. 

### 3. Suggest possible improvements to your pipeline

1. Using moving average or some other filtering mechanism to smooth out detected lanes across different frames
2. Have a more sophisticated mechanism of filtering out unlikely lines after the hough transform
3. Incorporate shape of lanes during detection so that random white / yellow objects dont throw the pipeline off
4. Do a partitioning of the lines by left half of screen and right half of screen. This will mitigate shortcoming number 3 listed above.
