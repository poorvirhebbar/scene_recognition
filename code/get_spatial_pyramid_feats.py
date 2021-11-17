import cv2
import numpy as np
from numpy import linalg
import math
from distance import pdist
from feature_extraction import feature_extraction
from sklearn.cluster import KMeans

def extract_denseSIFT(img):
	DSIFT_STEP_SIZE = 2
	sift = cv2.xfeatures2d.SIFT_create()
	disft_step_size = DSIFT_STEP_SIZE
	keypoints = [cv2.KeyPoint(x, y, disft_step_size)
		for y in range(0, img.shape[0], disft_step_size)
			for x in range(0, img.shape[1], disft_step_size)]

	descriptors = sift.compute(img, keypoints)[1]

	#keypoints, descriptors = sift.detectAndCompute(gray, None)
	return descriptors


# form histogram with Spatial Pyramid Matching upto level L with codebook kmeans and k codewords
def getImageFeaturesSPM(L, img, kmeans, k):
	W = img.shape[1]
	H = img.shape[0]   
	h = []
	for l in range(L+1):
		w_step = math.floor(W/(2**l))
		h_step = math.floor(H/(2**l))
		x, y = 0, 0
		for i in range(1,2**l + 1):
			x = 0
			for j in range(1, 2**l + 1):                
				desc = extract_denseSIFT(img[y:y+h_step, x:x+w_step])                
				#print("type:",desc is None, "x:",x,"y:",y, "desc_size:",desc is None)
				predict = kmeans.predict(desc)
				histo = np.bincount(predict, minlength=k).reshape(1,-1).ravel()
				weight = 2**(l-L)
				h.append(weight*histo)
				x = x + w_step
			y = y + h_step
            
	hist = np.array(h).ravel()
	# normalize hist
	dev = np.std(hist)
	hist -= np.mean(hist)
	hist /= dev
	return hist


# # get histogram representation for training/testing data
# def getHistogramSPM(L, data, kmeans, k):    
# 	x = []
# 	for i in range(len(data)):        
# 		hist = getImageFeaturesSPM(L, data[i], kmeans, k)        
# 		x.append(hist)
# 	return np.array(x)

def computeSIFT(data):
	x = []
	for i in range(0, len(data)):
		sift = cv2.xfeatures2d.SIFT_create()
		img = data[i]
		step_size = 15
		kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
		dense_feat = sift.compute(img, kp)
		x.append(dense_feat[1])
	    
	return x

# extract dense sift features from training images


def clusterFeatures(all_train_desc, k):
	kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
	return kmeans

def get_spatial_pyramid_feats(image_paths, max_level, feature):
	if feature == 'HoG':
		vocab = np.load('vocab_hog.npy')

	elif feature == 'SIFT':
		vocab = np.load('vocab_sift.npy')

	vocab_size = vocab.shape[0]
	x = []
	k = 200
	x_train = computeSIFT(train_data)
	x_test = computeSIFT(test_data)

	all_train_desc = []

	for i in range(len(x_train)):
		for j in range(x_train[i].shape[0]):
			all_train_desc.append(x_train[i][j,:])
	all_train_desc = np.array(all_train_desc)

	kmeans = clusterFeatures(all_train_desc, k)
	for path in image_paths:
		img = cv2.imread(path)[:, :, ::-1]
		hist = getImageFeaturesSPM(max_level, img, kmeans, k)
		x.append(hist)
	print(np.array(x))
	return np.array(x)

