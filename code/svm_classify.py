import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC

# def computeSIFT(data):
# 	x = []
# 	for i in range(0, len(data)):
# 		sift = cv2.xfeatures2d.SIFT_create()
# 		img = data[i]
# 		step_size = 15
# 		kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
# 		dense_feat = sift.compute(img, kp)
# 		x.append(dense_feat[1])
	    
# 	return x

# # extract dense sift features from training images


# def clusterFeatures(all_train_desc, k):
# 	kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
# 	return kmeans

# def formTrainingSetHistogram(x_train, kmeans, k):
# 	train_hist = []
# 	for i in range(len(x_train)):
# 		data = copy.deepcopy(x_train[i])
# 		predict = kmeans.predict(data)
# 		train_hist.append(np.bincount(predict, minlength=k).reshape(1,-1).ravel())

# 	return np.array(train_hist)

def svm_classify(train_image_feats, train_labels, test_image_feats, kernel_type):
	categories = np.unique(train_labels)
	new_labels = train_labels
	cfd_matrix = np.zeros((test_image_feats.shape[0],len(categories)))
	i =0
	for categ in categories:
		new_train_labels1 = np.zeros((len(train_labels)))
		new_train_labels2 = np.where(train_labels==categ, 1, new_train_labels1)
		cl = svm.SVC(random_state=0,C=0.025, kernel= kernel_type)
		cl.fit(train_image_feats, new_train_labels2)
		cfd_arr = cl.decision_function(test_image_feats)
		cfd_matrix[:,i]= cfd_arr
		i +=1
	j=0
	for vec in cfd_matrix:
		index = np.argmax(vec)
		new_labels[j]= categories[index]
		j +=1
	return new_labels










	# k = 60

	# x_train = computeSIFT(train_image_feats)
	# x_test = computeSIFT(test_image_feats)

	# all_train_desc = []

	# for i in range(len(x_train)):
	# 	for j in range(x_train[i].shape[0]):
	# 		all_train_desc.append(x_train[i][j,:])
	# all_train_desc = np.array(all_train_desc)

	# kmeans = clusterFeatures(all_train_desc, k)

	# train_hist = formTrainingSetHistogram(x_train, kmeans, k)
	# test_hist = formTrainingSetHistogram(x_test, kmeans, k)


	# # normalize histograms
	# scaler = preprocessing.StandardScaler().fit(train_hist)
	# train_hist = scaler.transform(train_hist)
	# test_hist = scaler.transform(test_hist)


	# for c in np.arange(0.0001, 0.1, 0.00198):
	# 	clf = LinearSVC(random_state=0, C=c)
	# 	clf.fit(train_hist, train_labels)
	# 	predict = clf.predict(test_hist)
	# 	print ("C =", c, ",\t Accuracy:", np.mean(predict == test_label)*100, "%")

	# return test_label





































	"""
	This function should train a linear SVM for every category (i.e., one vs all)
	and then use the learned linear cls to predict the category of every
	test image. Every test feature will be evaluated with all 15 SVMs and the
	most confident SVM will 'win'.

	:param train_image_feats: an N x d matrix, where d is the dimensionality of the feature representation.
	:param train_labels: an N array, where each entry is a string indicating the ground truth category
	    for each training image.
	:param test_image_feats: an M x d matrix, where d is the dimensionality of the feature representation.
	    You can assume M = N unless you've modified the starter code.
	:param kernel_type: SVM kernel type. 'linear' or 'RBF'

	:return:
	    an M array, where each entry is a string indicating the predicted
	    category for each test image.
	"""

	# categories = np.unique(train_labels)
	# clf = SVC(kernel = kernel_type)
	# clf.fit(train_image_feats, train_labels)
	# SVMResults = clf.predict(test_image_feats)

	# correct = sum(1.0 * (SVMResults == testLabels))
	# accuracy = correct / len(testLabels)
	# print "SVM: " +str(accuracy)+ " (" +str(int(correct))+ "/" +str(len(testLabels))+ ")"


	# svc = SVC(random_state=0)
 #    param_C = [0.001 , 0.01 , 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
 #    param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
 #    param_grid = [{'C': param_C,
 #                   'gamma': param_gamma,
 #                   'kernel': ['rbf']}]

 #    gs = GridSearchCV(estimator = svc,
 #                      param_grid= param_grid,
 #                      scoring='accuracy',
 #                     )
    
 #    gs = gs.fit(train_image_feats, train_labels)
    
 #    print(f'Best Training Score = {gs.best_score_:.3f} with parameters {gs.best_params_}')
    
 #    cl = gs.best_estimator_
 #    cl.fit(train_image_feats, train_labels)
    
    
    # pred_label = cl.predict(test_image_feats)
  
	# return pred_label

    # Your code here. You should also change the return value.
