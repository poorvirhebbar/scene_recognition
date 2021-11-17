import numpy as np


def get_features_from_pca(feat_num, feature):
    """
    This function loads 'vocab_sift.npy' or 'vocab_hog.npg' file and
    returns dimension-reduced vocab into 2D or 3D.

    :param feat_num: 2 when we want 2D plot, 3 when we want 3D plot
    :param feature: 'Hog' or 'SIFT'

    :return: an N x feat_num matrix
    """

    if feature == 'HoG':
        vocab = np.load('vocab_hog.npy')
    elif feature == 'SIFT':
        vocab = np.load('vocab_sift.npy')

    # Your code here. You should also change the return value.

    mean = np.mean(vocab.T, axis =1)
    centered = vocab -mean
    covar = np.cov(centered.T)

    val, vec = np.linalg.eig(covar)

    l = val.tolist()

    tuple1 = [l.index(x) for x in sorted(l, reverse=True)[:feat_num]]

    ft_size = vocab.shape[1]

    vector1 = np.zeros((ft_size, feat_num))

    for i in range(feat_num):
    	vector1[:,i] = vec[:,tuple1[i]]

    final = vector1.T.dot(centered.T)
    print(final)
    return final.T

    # return np.zeros((vocab.shape[0],2))


