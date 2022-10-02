# 0: 'airplane'
# 1: 'automobile'
# 2: 'bird'
# 3: 'cat'
# 4: 'deer'
# 5: 'dog'
# 6: 'frog'
# 7: 'horse'
# 8: 'ship'
# 9: 'truck'

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.transform import resize
from scipy.stats import norm
from PIL import Image

tic = time.process_time()

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

datadict = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')

all_data = unpickle('./cifar-10-python/cifar-10-batches-py/data_batch_1')
for i in range(2, 6):
    temp = unpickle(f'./cifar-10-python/cifar-10-batches-py/data_batch_{str(i)}')
    all_data['data'] = np.concatenate((all_data['data'], temp['data']))
    all_data['labels'] = np.concatenate((all_data['labels'], temp['labels']))

X = datadict["data"]

Y = datadict["labels"]

labeldict = unpickle('./cifar-10-python/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

testdatadict = unpickle('./cifar-10-python/cifar-10-batches-py/test_batch')

test_X = testdatadict['data']
test_Y = testdatadict['labels']
test_Y = np.array(test_Y)

# X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

def cifar10_color(X, num_of_images):
    new_X = np.reshape(X, (num_of_images, 3, -1))
    return resize(new_X, (num_of_images, 3, 1))
    # outputs Xp

######################## QUESTION 1 ########################

def compute_mean_colors_and_variances(data, labels):
    output = [[] for i in range(10)]
    for index, label in enumerate(labels):
        output[label].append(data[index])
    # print(temp)
    mu_output = np.empty((10, 3, 1))
    sigma_output = np.empty((10, 3, 1))
    for i, label in enumerate(output):
        # print('label ',label)
        # print('variance',np.var(label,axis=0))
        # try:
        #     print('mean ', np.mean(label, axis=0).reshape(1,3))
        # except:
        #     pass
        if not label:
            continue
        mu_output[i] = np.mean(label, axis=0)
        sigma_output[i] = np.var(label,axis=0)
    # print(output2)
    return mu_output, sigma_output
    
    
def cifar_10_naivebayes_learn(Xp, Y):
    mu, sigma = compute_mean_colors_and_variances(Xp, Y)
    temp = np.zeros((10, 1))
    for label in Y:
        temp[label] += 1
    priors = np.zeros((10, 1))
    for index, val in enumerate(temp):
        priors[index] = val / Y.shape[0]
    # print(priors)
    return mu, sigma, priors

def cifar10_classifier_naivebayes(x, mu, sigma, p):
    new_x = np.reshape(x, (3, -1))
    resized = resize(new_x, (3, 1))
    # print(resized)
    check = float('-inf')
    for i in range(10):
        value = np.prod(norm.pdf(resized, loc=mu[i], scale=sigma[i]), axis=0).mean()
        value *= p[i]
        if value > check:
            check = value
            output = i
    return output

######################## QUESTION 2 ########################

def cifar_10_bayes_learn(Xf, Y):
    mu_output = np.empty((10, 3))
    sigma_output = np.empty((10, 3, 3))
    # print(sigma_output.shape)
    output = [[] for i in range(10)]
    for index, label in enumerate(Y):
        output[label].append(Xf[index])
    # print('output', output)
    for i, label in enumerate(output):
        # print('label ',label)
        # print('cov',np.cov(np.asarray(label).T))
        # try:
        #     print('mean ', np.mean(label, axis=0).reshape(1,3))
        # except:
        #     pass
        if not label:
            continue
        data = np.asarray(label).squeeze()
        mu_output[i] = np.mean(data, axis=0)
        sigma_output[i] = np.cov(data.T)
    temp = np.zeros((10, ))
    for label in Y:
        temp[label] += 1
    priors = np.zeros((10,))
    for index, val in enumerate(temp):
        priors[index] = val / Y.shape[0]
    return mu_output, sigma_output, priors

def cifar10_classifier_bayes(x, mu, sigma, p):
    new_x = np.reshape(x, (3, -1))
    resized = resize(new_x, (3, 1))
    check = float('-inf')
    for i in range(10):
        value = np.random.multivariate_normal(mu[i], sigma[i]).mean()
        value *= p[i]
        if value > check:
            check = value
            output = i
    return output

######################## QUESTION 3 ########################

def cifar10_2x2_color(X, num_of_images):
    new_X = np.reshape(X, (num_of_images, 3, -1))
    print(new_X)
    return resize(new_X, (num_of_images, 3, 2, 2))

def cifar_10_bayes_learn2(Xf, Y):
    mu_output = np.empty((10, 12))
    sigma_output = np.empty((10, 12, 12))
    # print(sigma_output.shape)
    output = [[] for i in range(10)]
    for index, label in enumerate(Y):
        output[label].append(Xf[index])
    # print('output', output)
    for i, label in enumerate(output):
        # print('label ',label)
        # print('cov',np.cov(np.asarray(label).T))
        # try:
        #     print('mean ', np.mean(label, axis=0).reshape(1,3))
        # except:
        #     pass
        if not label:
            continue
        val = np.mean(label)
        mu_output[i] = np.full((12,), val)
        sigma_output[i] = np.cov(np.asarray(label).T)
    temp = np.zeros((10, ))
    for label in Y:
        temp[label] += 1
    priors = np.zeros((10,))
    for index, val in enumerate(temp):
        priors[index] = val / Y.shape[0]
    return mu_output, sigma_output, priors

# Xf = cifar10_2x2_color(X[:10], 10)
# mu, sigma, p = cifar_10_bayes_learn2(Xf, Y[:2])

# print('original', X[0])
# new_image = np.reshape(X[0],(3, -1))
# print('reshape', new_image)
# print('resize', resize(new_image, (3, 2, 2)))
# print('resized', cifar10_color(X[:1000], 1000))

def class_acc(pred, gt):
    # if pred.shape[0] == gt.shape[0]:
    num_of_pts = pred.shape[0]
    # else:
    #     print('arrays do not match')
    #     return
    counter = 0
    for i in range(num_of_pts):
        if pred[i] == gt[i]:
            counter += 1
    return (counter / num_of_pts) * 100

# # Change training_num to set how many datasets to train model
# training_num = 200
# print(X.shape)
# Xp = cifar10_color(X[:training_num], training_num)

# For question 1
def qn1(Xp, test_num):
    # mu, sigma, p = cifar_10_naivebayes_learn(Xp, Y[:training_num])
    mu, sigma, p = cifar_10_naivebayes_learn(Xp, all_data['labels'])
    result = np.empty((test_num,))
    for i in range(test_num):
        result[i] = cifar10_classifier_naivebayes(test_X[i], mu, sigma, p)
    print('qn1: ',class_acc(result, test_Y[:test_num]),'%')

# For question 2
def qn2(Xp, test_num):
    # mu2, sigma2, p2 = cifar_10_bayes_learn(Xp, Y[:training_num])
    mu, sigma, p = cifar_10_bayes_learn(Xp, all_data['labels'])
    result2 = np.empty((test_num,))
    for i in range(test_num):
    # For question 1
        result2[i] = cifar10_classifier_naivebayes(test_X[i], mu, sigma, p)
    print('qn2: ',class_acc(result2, test_Y[:test_num]),'%')

# For question 3
def qn3():
    Xf = cifar10_2x2_color(X[:10], 10)
    mu, sigma, p = cifar_10_bayes_learn2(Xf, Y[:2])
    

# Change test_num to set how many datasets to predict labels for
test_num = 10000
Xp = cifar10_color(all_data['data'], 50000)
qn1(Xp, test_num)
qn2(Xp, test_num)

Xf = cifar10_2x2_color(X[:10], 10)


toc = time.process_time()
print(toc - tic)


# def cifar10_classifer_random(x):
#     return np.random.randint(0,9,x.shape[0])

# def cifar10_classifier_1nn(x, trdata, trlabels):
#     dist = np.zeros(len(trdata))
#     for i in range(0, len(trdata)):
#         dist[i] = np.sum(np.subtract(trdata[i], x) ** 2)
#     test_label = trlabels[dist.argmin()]
#     return test_label



# testX = testdatadict['data'][:1000]
# # output = np.zeros(len(testX))
# # for i in range(len(testX)):
# #     output[i] = cifar10_classifier_1nn(testX[i], X[:1000], Y[:1000])
# # print(output)
# output = cifar10_classifier_1nn(testX, X[:1000], Y[:1000])
# print(class_acc(output, Y))
# toc = time.process_time()
# print(toc - tic)

# # for i in range(X.shape[0]):
# #     # Show some images randomly
# #     if random() > 0.999:
# #         plt.figure(1)
# #         plt.clf()
# #         plt.imshow(X[i])
# #         plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
# #         plt.pause(1)