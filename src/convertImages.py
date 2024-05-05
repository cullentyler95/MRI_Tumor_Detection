import cv2
import numpy as np
import matplotlib.pyplot as plt
import ants

from antspynet.utilities import brain_extraction

brain_mask =  brain_extraction('.\\data\\yes\\Y11.jpg', verbose=True)