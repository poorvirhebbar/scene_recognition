import cv2
import numpy as np

def computeSIFT(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        step_size = 20
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        dense_feat = sift.compute(img, kp)
        x.append(dense_feat[1])
        
    return x

def feature_extraction(img, feature):
	if feature == 'HoG':
		# HoG parameters
		win_size = (32, 32)
		block_size = (32, 32)
		block_stride = (16, 16)
		cell_size = (16, 16)
		nbins = 9
		deriv_aperture = 1
		win_sigma = 4
		histogram_norm_type = 0
		l2_hys_threshold = 2.0000000000000001e-01
		gamma_correction = 0
		nlevels = 64

		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
					histogramNormType,L2HysThreshold,gammaCorrection,nlevels)		
		hist = hog.compute(img)
		# print(hist)
		return hist 


	elif feature == 'SIFT':
		sift = cv2.xfeatures2d_SIFT.create()
		step_size = 20
		kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
		dense_feat = sift.compute(img, kp)
		return dense_feat[1]


		# s = computeSIFT(img) 
		# # print(s)
		# return s


		# Your code here. You should also change the return value.

        # return np.zeros((1500, 128))




