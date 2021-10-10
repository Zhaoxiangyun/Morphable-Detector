"""
Module for cv2 utility functions and maintaining version compatibility
between 3.x and 4.x
"""
import cv2


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

class LiveImagePlotter(object):
    """Live/interactive image plotting utility. You give an image,
    this class will show the image and wait for user/key input to
    continue or break.
    """
    def __init__(self):
        self.cnt = 0
        self.print_help_interval = 25
        self.help_msg = "=> Press <space> to continue; <Esc> to stop script;"

    def __del__(self):
        cv2.destroyAllWindows()

    def __call__(self, image, name=""):
        if self.cnt % self.print_help_interval == 0:
            print(self.help_msg)
        cv2.imshow(name, image)
        next_image = True
        while True:
            k = cv2.waitKey(-1) # Wait for a key
            if k == 27: # Esc: stop
                next_image = False
                break
            elif k == 32:
                break
            else:
                print("Unknown key: {:d}".format(k))
                print(self.help_msg)
        self.cnt += 1
        return next_image
