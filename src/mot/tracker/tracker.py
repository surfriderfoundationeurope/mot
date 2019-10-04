import cv2
import numpy as np
from numpy.linalg import inv

class CameraFlow():
    def __init__(self):
        self.margin_w = 200
        self.margin_h = 20
        self.img_shape = (768,1024)
        self.features_mask = np.zeros(self.img_shape, dtype="uint8")
        self.features_mask[self.margin_h:-self.margin_h, self.margin_w:-self.margin_w] = 1


    def compute_transform_matrix(self, im_prev, im_next):
        prev_pts = cv2.goodFeaturesToTrack(im_prev,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3,
                                           mask=self.features_mask)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(im_prev, im_next, prev_pts, None)
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        next_pts = next_pts[idx]
        m = cv2.estimateAffine2D(prev_pts, next_pts)
        return m[0]

    def warp_image(self, im, matrix):
        return cv2.warpAffine(im, matrix, self.img_shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    def warp_coords(self, coords, matrix):
        '''
        Assure coords is a np array of shape (X, 2)
        '''
        matrix = cv2.invertAffineTransform(matrix)
        linear = matrix[:, 0:2]
        transl = matrix[:, 2]
        return np.dot(coords, linear) + transl

    def compute_transform_matrices(self, images_stack):
        im_next = images_stack[0]
        matrices = []
        for i in range(len(images_stack) - 1):
            im_prev = im_next
            im_next = images_stack[i+1]
            matrices.append(self.compute_transform_matrix(im_prev, im_next))
        return matrices
