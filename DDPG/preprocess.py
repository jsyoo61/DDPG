import numpy as np
from collections import deque

class Preprocessor():

    def __init__(self, m=4):
        self.m = m
        self.recent_frame = None
        self.processed_frames = deque(maxlen = self.m)

    def preprocess(self, image):

        # 1. Max pixel value from current & previous frame
        if self.recent_frame == None:
            processed_1 = image
        else:
            processed_1 = np.maximum(self.recent_frame, image)

        # 2. Extract Y channel (luminance)
        processed_2 = f1(processed_1)

        # 3. Rescale to 84 * 84
        processed_final = f2(processed_2)

        # 4. Enqueue result
        self.processed_frames.append(processed_final)

        # If processed_frames is not full, (meaning it's the initial phase)
        while len(self.processed_frames) < self.m:
            # Copy 1st frame
            self.processed_frames.append(processed_final)

        # Discard previous frame, substitue with current frame
        self.recent_frame = image

        return np.array(self.processed_frames)

    def reset(self):

        # New game
        # No recent frame, No processed frames
        self.recent_frame = None
        self.processed_frames = deque(maxlen = self.m)
