import pandas as pd
from pylab import *
import cv2
from sklearn.decomposition import PCA, RandomizedPCA
import os
import util.feature_util as feat
import cvxpy as cvx


'''
## Load data and features
'''
df = pd.read_csv("../../data/images/features/CUB/BVLCref_fc8-CUB.csv")

CNNfeatures = array(df.iloc[:,1:])
img_paths = list(df.iloc[:,0])
img_paths = ["../../data/images/imgs/" + img_path.split("data/")[-1] for img_path in img_paths]

print 'data and features loaded'


'''
## Rescale imgs
'''
width = 200
height = 200

DT = feat.DimTransformer(w = width, h = height)
imgs = DT.transform(img_paths)

BW = feat.BWTransformer()
imgs_bw = BW.transform(imgs)


img_arrays = zeros((len(imgs), width*height))
for i, img in enumerate(imgs_bw):
    img_arrays[i,:] = reshape(img, (1,width*height))

print 'images rescaled'




'''
# ## Find pixel weights for randomly selected images
'''

def sample_diffs(sample, np_arr):
    diffs = np.sqrt((np_arr - sample)**2)
    return diffs

def optimize_weights(X_diffs, y_diffs):
    sc = np.linalg.norm(X_diffs)
    A = X_diffs/sc
    b = y_diffs/sc
    w = cvx.Variable(X_diffs.shape[1])
    #objective = cvx.Minimize(cvx.sum_entries(cvx.huber(A*w - b,1000)))
    objective = cvx.Minimize(cvx.norm(A*w - b,2))
    constraints = [0 <= w]

    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.SCS)
    return prob.status, w.value


random_samples = random_integers(0, len(img_paths)-1, 100)

for i, sample_id in enumerate(random_samples):
    print 'working on sample', i
    '''
    # ## Compute Pairwise differences
    '''
    indx_list = [range(0,sample_id), range(sample_id+1, img_arrays.shape[0])]
    indx = [item for sublist in indx_list for item in sublist]

    X_diffs = sample_diffs(img_arrays[sample_id,:], img_arrays[indx,:])
    y_diffs = sum(sample_diffs(CNNfeatures[sample_id,], CNNfeatures[indx,:]), axis=1)

    '''
    # ## Learn Optimal Weights
    '''
    statusprob, weights = optimize_weights(X_diffs, y_diffs)


    weights_thresh = np.array(weights).copy()
    weights_thresh[weights_thresh < 1e-4] = 0
    weights_thresh = 255.*weights_thresh/max(weights_thresh)
    #weights_reshape = reshape(weights_thresh, (height, width, 3))
    weights_reshape = reshape(weights_thresh, (height, width))


    figure(figsize=(16, 10))
    subplot(2,2,1)
    imshow(imgs[sample_id])
    subplot(2,2,2)
    imshow(weights_reshape, cmap='Greys_r')
    savefig('../../results/CUB_' + str(sample_id) + '.png', bbox_inches='tight')

