# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:43:43 2021

@author: Admin
"""

'''

input a data. (histdata of the bone.)  (n,)

1. use a cluster algorithm to find the threshold, noraml bone and bme bone.

2. calculate the bme proportion.(optional)

3. draw the histogram with line.(optional)



'''


import numpy as np
from sklearn import mixture



class cluster_method(object):
    
    def __init__(self, data, quan_data, K, histfig_savepath=None):
        """
        K: the number of clusters
        N: the number of samples
        
        """
        
        self.data = data
        self.quan_data = quan_data
        self.K = K
        self.centers = 0
        self.labels = 0
        
        #bme propertities
        self.bme_threshold = 0
        self.bme_proportion = 0
        self.bme_mean = 0
        self.bme_std = 0
        
        #model parameters
        self.alpha_normal_gmm = 0
        self.alpha_bme_gmm = 0        
        self.normal_mean = 0
        self.normal_std = 0
        self.bme_gmm_mu = 0
        self.bme_gmm_sigma = 0        

        self.bme_upper_threshold = max(data)
        #savepath
        self.histfig_savepath = histfig_savepath
        

    def GMM_fit_2th(self):
        #bme_threshold = mu_(normal) + sigm(normal) * 1.96
        X = self.data.reshape(-1,1)
        clf = mixture.GaussianMixture(n_components=self.K, covariance_type='spherical',max_iter=1000,tol = 1e-5)
        clf.fit(X)
        mu_ = clf.means_.reshape(2,)
        sigma2_ = np.sqrt(clf.covariances_)
        alpha_ = clf.weights_
        
        nor_arg = np.argmin(mu_)
        bme_arg = np.argmax(mu_)
        
        self.alpha_normal_gmm = alpha_[nor_arg]
        self.alpha_bme_gmm = alpha_[bme_arg]
        self.normal_mean = mu_[nor_arg]
        self.normal_std = sigma2_[nor_arg]
        self.bme_gmm_mu = mu_[bme_arg]
        self.bme_gmm_sigma = sigma2_[bme_arg]
        
        self.bme_threshold = self.normal_mean + self.normal_std * 1.96
        self.bme_proportion, self.bme_mean, self.bme_std = bme_information(self.quan_data,self.bme_threshold, self.bme_upper_threshold)

def bme_information(data,bme_down_threshold,bme_up_threshold=None):
    if not bme_up_threshold:
        bme_up_threshold = max(data)
    if bme_down_threshold >= bme_up_threshold:
        return 'None', 'None', 'None'
    else:
        bme_proportion = np.sum((data>bme_down_threshold)&(data<bme_up_threshold))/data.shape[0]
        mean = np.mean(data[(data>bme_down_threshold)&(data<bme_up_threshold)])
        std = np.std(data[(data>bme_down_threshold)&(data<bme_up_threshold)])
        return bme_proportion, mean, std


# return the histogram data for quantificatino.
def quan_data(img, seg, quan_bone_num):
    # img = img.reshape(20,448,448)
    # seg = seg.reshape(20,448,448)
    if quan_bone_num == None:
        # 1.Chose overall bone
        bone_seg = np.zeros(seg.shape)
        bone_seg[seg >= 1] = 1
        bone_img = img * bone_seg
        bone_img[seg < 1] = -1
    if quan_bone_num != None:
        # 1.Chose a particular bone
        bone_seg = np.zeros(seg.shape)
        for i in quan_bone_num:
            bone_seg[seg == i] = 1
        bone_img = img * bone_seg
        for i in range(16):
            if i not in quan_bone_num:
                bone_img[seg == i] = -1

    # 2.calculate overall bone histogram
    histogram = bone_img.flatten()
    histogram = np.delete(histogram, np.where(histogram == -1))
    # hist,bin_edges = np.histogram(histogram,bins=np.arange(0,2000,10), density=True)
    return histogram