import numpy as np
from distance import pdist

# def dist(vec1, vec2):
# 	sum_of_squares=0
# 	for i in range(len(vec1)):
# 		sum_of_squares = (vec1[i] - vec2[i])*(vec1[i] - vec2[i])
# 	return math.sqrt(sum_of_squares)

def kmeans_clustering(all_features, vocab_size, epsilon, max_iter):

	vocab_matrix = np.zeros((vocab_size, all_features.shape[1]))
	

	total_pts = len(all_features[:,0])
	class_arr = np.zeros(total_pts)

	random_array = np.random.choice(total_pts, vocab_size, replace=False)
	random_array.sort()
	

	for i in range(vocab_size):
		vocab_matrix[i]= all_features[random_array[i]]

	for iteration in range(max_iter):
		dist_mat = pdist(all_features, vocab_matrix)
		c1=0
		for vec in dist_mat:
			index= np.argmin(vec)
			class_arr[c1]= index
			c1 +=1

		sum_in_class = np.zeros((vocab_size, all_features.shape[1]))
		new_vocab_matrix = np.zeros((vocab_size, all_features.shape[1]))
		class_num = np.zeros((vocab_size))
		

		for i in range(total_pts):
			curr_class = class_arr[i]
			sum_in_class[int(curr_class)] += all_features[i]
			class_num[int(curr_class)] +=1

		for i in range(vocab_size):
			if class_num[i]==0:
				new_vocab_matrix[i]= vocab_matrix[i]
			else:
				new_vocab_matrix[i]= (sum_in_class[i])/ (class_num[i])

		dist_mat1 = pdist(vocab_matrix, new_vocab_matrix)

		for n in range(vocab_size):
			if dist_mat1[n][n] > epsilon:
				break
			if n == vocab_size -1:
				break

		vocab_matrix = new_vocab_matrix
		
	# print(vocab_matrix)

	return vocab_matrix
