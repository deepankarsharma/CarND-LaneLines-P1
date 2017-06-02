import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import os
from optparse import OptionParser


PARAMS_DICT = {
    'debug_set': set("*"),
    'hsv_yellow_lower': [16, 100, 100],
    'hsv_yellow_upper': [46, 255, 255],
    'hsv_white_threshold': 40,
    'draw_line_width': 8,
    'draw_line_color': (255, 0, 0),
    'hough_slope_threshold': 0.5,
    'hough_rho': 1,
    'hough_theta': np.pi / 180,
    'hough_threshold': 15,
    'hough_min_line_length': 10,
    'hough_max_line_gap': 20,
    'canny_low': 50,
    'canny_high': 150,
    'gaussian_blur_size': 7
}

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.

    Improvements over the standard version are
    1. Filters away lines whose slope is flat below a certain threshold
    2. Seperates out lines with positive and negative slopes into different streams
    3. Does combination of lines into a single line
    """
    lines = cv2.HoughLinesP(img, rho, theta,
                            threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (float(y2) - float(y1)) / (float(x2) - float(x1))
        if abs(slope) < PARAMS_DICT['hough_slope_threshold']:
            continue
        if slope >= 0:
            pos_x.append(x1)
            pos_x.append(x2)
            pos_y.append(y1)
            pos_y.append(y2)
        else:
            neg_x.append(x1)
            neg_x.append(x2)
            neg_y.append(y1)
            neg_y.append(y2)

    y1 = img.shape[0]
    y2 = img.shape[0] * 0.6

    if pos_x:
        pos_m, pos_b = np.polyfit(pos_x, pos_y, 1)
        pos_x1 = (y1 - pos_b) / pos_m
        pos_x2 = (y2 - pos_b) / pos_m

    if neg_x:
        neg_m, neg_b = np.polyfit(neg_x, neg_y, 1)
        neg_x1 = (y1 - neg_b) / neg_m
        neg_x2 = (y2 - neg_b) / neg_m

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if pos_x:
        cv2.line(line_img, (int(pos_x1), int(y1)), (int(pos_x2), int(y2)),
            PARAMS_DICT['draw_line_color'], PARAMS_DICT['draw_line_width'])

    if neg_x:
        cv2.line(line_img, (int(neg_x1), int(y1)), (int(neg_x2), int(y2)),
            PARAMS_DICT['draw_line_color'], PARAMS_DICT['draw_line_width'])

    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


class ImageResult(object):
    @classmethod
    def from_file(klass, fname, debug_set=None):
        orig = mpimg.imread(fname)
        return klass(orig, fname, debug_set)

    @classmethod
    def from_image(klass, img, debug_set=None):
        return klass(img, '<unknown>', debug_set)

    def __init__(self, orig, fname, debug_set=None):
        self.fname = fname
        self.orig = orig
        self.debug_set = set() if debug_set is None else debug_set
        self.proc = self.process_image()

    def process_image(self):

        def dbg(tag, img, gather):
            if tag in self.debug_set or "*" in self.debug_set:
                gather.append(img)

        dbg_list = []
        orig_image = self.orig
        dbg("orig", orig_image, dbg_list)

        # Convert to hsv so that we can build masks for yellow
        # and white lane markers
        hsv_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2HSV)
        dbg("hsv", hsv_image, dbg_list)

        # define range of yellow color in HSV
        lower_yellow = np.array(PARAMS_DICT['hsv_yellow_lower'])
        upper_yellow = np.array(PARAMS_DICT['hsv_yellow_upper'])
        # Threshold the HSV image to get only yellow color
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 255 - PARAMS_DICT['hsv_white_threshold']])
        upper_white = np.array([255, PARAMS_DICT['hsv_white_threshold'], 255])
        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        # Build a combined mask to detect yellow and white lines
        combined_mask = yellow_mask | white_mask
        dbg("combined_mask", np.dstack((combined_mask, combined_mask, combined_mask)), dbg_list)

        image = grayscale(orig_image)
        dbg("gray", np.dstack((image, image, image)), dbg_list)

        image = cv2.bitwise_and(image, image, mask=combined_mask)
        dbg("masked", np.dstack((image, image, image)), dbg_list)

        image = gaussian_blur(image, PARAMS_DICT['gaussian_blur_size'])
        dbg("blur1", np.dstack((image, image, image)), dbg_list)

        shp = image.shape
        vertices = np.array(
            [[
                (0, shp[0]),
                (shp[1] / 2 - 75, shp[0] * 0.6),
                (shp[1] / 2 + 75, shp[0] * 0.6),
                (shp[1], shp[0])
            ]], dtype=np.int32)

        image = canny(image, PARAMS_DICT['canny_low'], PARAMS_DICT['canny_high'])
        dbg("canny", np.dstack((image, image, image)), dbg_list)

        image = region_of_interest(image, vertices)
        dbg("region", np.dstack((image, image, image)), dbg_list)

        image = hough_lines(image,
                            PARAMS_DICT['hough_rho'],
                            PARAMS_DICT['hough_theta'],
                            PARAMS_DICT['hough_threshold'],
                            PARAMS_DICT['hough_min_line_length'],
                            PARAMS_DICT['hough_max_line_gap'])
        dbg("hough", image, dbg_list)
        image = weighted_img(image, orig_image, 0.2, 1.0)
        dbg("final", image, dbg_list)

        if dbg_list:
            mpimg.imsave('debug_' + os.path.basename(self.fname), np.vstack(dbg_list))
        return image


def process_image(image):
    """ Process image and mark any lanes found in it """
    ir = ImageResult.from_image(image)
    return ir.proc


def process_input(ifname, ofname, process_video=False, debug=False):
    """
    Perform lane finding on a media file
    :param ifname: Filename of input file to process
    :param ofname: Filename of target output file to create to store processed data
    :param process_video: Set true if incoming media file is a video
    :return: None
    """
    if process_video:
        video = VideoFileClip(ifname)
        processed = video.fl_image(process_image)
        processed.write_videofile(ofname, audio=False)
        return
    debug_set = PARAMS_DICT['debug_set'] if debug else set()
    ir = ImageResult.from_file(ifname, debug_set)
    mpimg.imsave(ofname, ir.proc)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--ifname", dest="ifname", help="Input filename to process")
    parser.add_option("-o", "--ofname", dest="ofname", help="Output filename")
    parser.add_option(
        "-v", "--process-video", action="store_true",
        dest="process_video", default=False,
        help="Incoming file is a video")
    parser.add_option(
        "-d", "--debug", action="store_true",
        dest="debug", default=False,
        help="Enable debug output")
    options, args = parser.parse_args()

    ifname = options.ifname
    ofname = options.ofname
    process_video = options.process_video
    debug = options.debug
    print(ifname)
    process_input(ifname, ofname, process_video, debug)
