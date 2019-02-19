"""
Created on Sun Jan 10 21:10:19 2019

@author: Tae Hoon Jun
"""
from numpy.linalg import norm
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from random import randrange
import glob

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

      
def part1(faces_file_name, act):
  #Note: you need to create the uncropped folder first in order 
  #for this to work
  try:
      if not os.path.exists("uncropped/"):
          os.makedirs("uncropped/")
      if not os.path.exists("cropped/"):
          os.makedirs("cropped/")
  except OSError:
      print ("Error: Creating directory. " + "uncropped/ and cropped/")
  

  
  #act = list(set([a.split("\t")[0] for a in open(faces_file_name).readlines()]))
  
  testfile = urllib.URLopener()            
  for a in act:
      name = a.split()[1].lower()
      i = 0
      for line in open(faces_file_name):
          if a in line:
              filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
              #A version without timeout (uncomment in case you need to 
              #unsupress exceptions, which timeout() does)
              #testfile.retrieve(line.split()[4], "uncropped/"+filename)
              #timeout is used to stop downloading images which take too long to download
              #Continue downloading if download wasn't completed last time
              if not os.path.isfile("uncropped/"+filename):
                  timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                  if not os.path.isfile("uncropped/"+filename):
                      continue
                  #select 3d images
                  try:
                      #read in the images                  
                      img = imread("uncropped/" + filename)
                      #Get the area of the faces
                      face = line.split()[5].split(',')
                      cropped = img[int(face[1]):int(face[3]), int(face[0]):int(face[2]), :]
                      #Convert image to grayscale and resize image to 32X32
                      result = rgb2gray(imresize(cropped, (32,32)))    
                      
                      #save cropped image to the cropped folder, change heatmap version to grayscale
                      plt.imsave("cropped/"+filename, result, cmap="gray")
                  except:
                      print("Unable to load image: "+ filename)
                      continue
              i += 1
              
def part2(act, training, validation, test, seed=0):
    '''Return the randomly assigned training, validation, test data dictionary
    Arguments:
    act -- array of an actors' real names
    training -- range of training set
    validation -- range of validation set
    test -- range of test set
    '''
    training_set = {}
    validation_set = {}
    test_set = {}
    np.random.seed(0)
    for a in act:
        training_set[a] = []
        validation_set[a] = []
        test_set[a] = []
        name = a.split()[1].lower()
        #get the all filenames for the actor
        files = glob.glob('cropped/' + name + '*')
        i = 0
        
        while (len(files) > 0 and i < training + validation + test):
            random_index = randrange(len(files))
            random_data = files.pop(random_index)
            if len(training_set[a]) < training:
                training_set[a].append(random_data)
            elif len(validation_set[a]) < validation:
                validation_set[a].append(random_data)
            elif len(test_set[a]) < test:
                test_set[a].append(random_data)
            i += 1
            
    return training_set, validation_set, test_set

def grad_descent(f, gradf, x, y, init_t, alpha, max_iter, EPS = 1e-5):
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    while norm(t-prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*gradf(x, y, t)
        iter += 1
        
    return t

"""
Quadratic loss function as linear regression
"""
def f(x, y, theta):
    return sum((y - np.dot(theta.T, x))**2)

"""
Derivative of the loss function respect to theta
"""
def gradf(x, y, theta):
    #sum start from 1
    return -2*sum((y-np.dot(theta.T, x))*x, 1)

def output(x, theta):
    result = []
    for h in np.dot(theta.T, x):
        if h < 0:
            result.append(-1)
        else:
            result.append(1)
    return result

def create_matrix(classifier, data_set):
    
    feature = np.array([])
    label = np.array([])
    feature = feature.reshape(1025, 0)
    for actor in classifier.keys():
        for image in data_set[actor]:
            #gather the score for the images(always correct score)
            label = np.hstack((label, np.array([classifier[actor]])))
            im = imread(image)
            #divide image input by 255 and arrange it by 1024x1
            im = im[:,:,0]
            im = (im/255.).reshape(1024,1)
            #x_0 = 1
            im = np.vstack((np.array([1]), im))
            #gather the images pixels
            feature = np.hstack((feature, im))
            
    return feature, label

def record_grad_descent(f, gradf, x, y, init_t, alpha, max_iter, recording_step, EPS = 1e-5):
    iter_num = []
    f_values = []
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    init_f = f(x, y, init_t)
    while norm(t-prev_t) > EPS and iter < max_iter and f(x, y, t) <= init_f*2:
        prev_t = t.copy()
        t -= alpha*gradf(x, y, t)
        if iter % recording_step == 0:
            f_values.append(f(x, y, t))
            iter_num.append(iter)
        iter += 1
        
    return t, f_values, iter_num

def performance(feature, theta, label):
    num_correct = 0.0
    classification = output(feature, theta)
    for i in range(label.size):
        if classification[i] == label[i]:
            num_correct += 1
    
    return num_correct/label.size

def part3():
  try:
      if not os.path.exists("figures/"):
          os.makedirs("figures/")
  except OSError:
      print ("Error: Creating directory. " + "figures/")
          
  classifier = {'Alec Baldwin': -1, 'Steve Carell' : 1}
  
  act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
  training_set, validation_set, test_set = part2(act, 70, 10, 10)
  data, data_label = create_matrix(classifier, training_set)
  
  print("Learning rate #1:")
  plt.figure(1)
  theta0 = np.random.normal(0, 0., 1025)
  legend = []
  max_iter = 100
  recording_step = 20
  for x in range(16, 26, 2):
      alpha = 0.000001*x
      theta, f_value, iter_num = record_grad_descent(f, gradf, data, data_label, theta0, alpha, max_iter, recording_step)
      plt.plot(iter_num, f_value, '^-')
      legend.append("alpha = "+str(alpha))
  plt.legend(legend)
  plt.xlabel("Number of iterations")
  plt.ylabel("Cost function value on Training Set")
  plt.savefig("figures/p3f1.jpg")
  plt.show()
  
  print("Learning rate #2:")
  plt.figure(2)
  theta0 = np.random.normal(0, 0., 1025)
  legend = []
  max_iter = 1000
  recording_step = 200
  for x in range(12, 24, 2):
      alpha = 0.000001*x
      theta, f_value, iter_num = record_grad_descent(f, gradf, data, data_label, theta0, alpha, max_iter, recording_step)
      plt.plot(iter_num, f_value, '^-')
      legend.append("alpha = "+str(alpha))
  plt.legend(legend)
  plt.xlabel("Number of iterations")
  plt.ylabel("Cost function value on Training Set")
  plt.savefig("figures/p3f2.jpg")
  plt.show()
  
  print("Theta Initialization")
  np.random.seed(0)
  for deviation in [0., 0.0001, 0.001, 0.01, 0.1, 1]:
      f_value = []
      #Try 5 times
      for j in range(5):
          theta0 = np.random.normal(0,deviation,1025)
          f_value.append(f(data, data_label,theta0))
      print "Initial value of cost function on the training set with standard deviation of theta to be %.4f : Mean: f(x) = %.2f  Standard Error: %.2f" % (deviation, np.mean(f_value), np.std(f_value))
      
  v_data, v_label = create_matrix(classifier, validation_set)
  print "\nReport the values of the cost function on the training and the validation sets"
  print "Report performance on the training and the validation sets\n"
  alpha = 0.000015
  max_iter = 100000
  cost_t = []
  cost_v = []
  perf_t = []
  perf_v = []
  for seed in range(5):
      training_set, validation_set, test_set = part2(act, 70,10,10,seed)
      theta0 = np.random.normal(0,0.,1025)
      data, dlabel = create_matrix(classifier, training_set)
      test, tlabel = create_matrix(classifier, test_set)
      validation, vlabel = create_matrix(classifier, validation_set)
      theta = grad_descent(f, gradf, data, dlabel, theta0, alpha, max_iter)
      cost_t.append(f(data, dlabel, theta)/(2*dlabel.size))
      cost_v.append(f(validation, vlabel, theta)/(2*vlabel.size))
      perf_t.append(performance(data, theta, dlabel, classifier)*100)
      perf_v.append(performance(validation, theta, vlabel, classifier)*100)
  print "Value of cost function on the training set: f(x) = %f +/- %f" % (np.mean(cost_t),np.std(cost_t))
  print "Value of cost function on the validation set: f(x) = %f +/- %f" % (np.mean(cost_v),np.std(cost_v))
  print "Performance of the classifier on the training set:  %.2f%% +/- %f%%" % (np.mean(perf_t),np.std(perf_t))
  print "Performance of the classifier on the validation set:  %.2f%% +/- %f%% \n" % (np.mean(perf_v),np.std(perf_v))


def part4_a():
    try:
      if not os.path.exists("figures/"):
          os.makedirs("figures/")
    except OSError:
      print ("Error: Creating directory. " + "figures/")
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
    classifier = {'Alec Baldwin': -1, 'Steve Carell' : 1}
    
    print("Display thetas obtained by training using the full dataset")
    training_set, validation_set, test_set = part2(act, 70,10,10)
    feature, label = create_matrix(classifier, training_set)
    
    init_t = np.random.normal(0, 0., 1025)
    alpha = 0.000015
    max_iter = 100000
    theta = grad_descent(f, gradf, feature, label, init_t, alpha, max_iter)
    plt.imshow((theta[1:]).reshape(32,32), cmap = "gray")
    plt.savefig("figures/part4af1.jpg")
    plt.show()
    
    print("Display thetas obtained by training using a training set that" 
          + "contains only two images of each actor")
    for i in training_set:
        training_set[i] = random.sample(training_set[i], 2)
        
    feature, label = create_matrix(classifier, training_set)
    
    init_t = np.random.normal(0, 0., 1025)
    alpha = 0.000015
    max_iter = 100000
    theta = grad_descent(f, gradf, feature, label, init_t, alpha, max_iter)
    plt.imshow((theta[1:]).reshape(32,32), cmap = "gray")
    plt.savefig("figures/part4af2.jpg")
    plt.show()
    
def part4_b():
    try:
      if not os.path.exists("figures/"):
          os.makedirs("figures/")
    except OSError:
      print ("Error: Creating directory. " + "figures/")
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
    classifier = {'Alec Baldwin': -1, 'Steve Carell' : 1}
    training_set, validation_set, test_set = part2(act, 70,10,10)
    feature, label = create_matrix(classifier, training_set)
    
    print("Stopping gradient descent process earlier and later")
    init_t = np.random.normal(0, 0., 1025)
    alpha = 0.000015
    max_iter = [100, 1000000]
    figure = 1
    for i in max_iter:
        print("Maximum iteration is: " + str(i))
        theta = grad_descent(f, gradf, feature, label, init_t, alpha, i)
        plt.imshow((theta[1:]).reshape(32,32), cmap = "gray")
        plt.savefig("figures/part4bf" + str(figure) + ".jpg")
        plt.show()
        figure += 1
    max_iter = 1000
    print("Initializing thetas with different standard deviation")
    for i in [0., 0.1]:
        init_t = np.random.normal(0, i, 1025)
        print("Standard deviation is :" + str(i))
        theta = grad_descent(f, gradf, feature, label, init_t, alpha, max_iter)
        plt.imshow((theta[1:]).reshape(32,32), cmap = "gray")
        plt.savefig("figures/part4bf" + str(figure) + ".jpg")
        plt.show()
        figure += 1
    
def part5():
    act = ['Kristin Chenoweth', 'Fran Drescher', 'America Ferrera', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan']
    classifier = {'Lorraine Bracco': -1, 'Peri Gilpin': -1, 'Angie Harmon':-1, 'Alec Baldwin': 1, 'Bill Hader': 1, 'Steve Carell' : 1}
    classifier2 = {'Kristin Chenoweth':-1, 'Fran Drescher':-1, 'America Ferrera':-1, 'Daniel Radcliffe':1, 'Gerard Butler':1, 'Michael Vartan':1}
    training_set_NA, validation_set_NA, test_set_NA = part2(act, 10, 0, 10)
    test, tlabel = create_matrix(classifier2, test_set_NA)
    
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
    alpha = 0.000005
    max_iter = 10000
    
    training_set_size = []
    perform_test_mean = []
    perform_validation_mean = []
    perform_training_mean = []
    perform_test_err = []
    perform_validation_err = []
    perform_training_err = []
    
    print("Report performance on the test set:")
    
    for i in range(10,80,10):
        perform_test = []
        perform_validation = []
        perform_training = []
        training_set_size.append(i*6)
        for j in range(10):
            training_set,validation_set,test_set_not_used = part2(act,i+10,10, 0,j)
            validation,vlabel = create_matrix(classifier, validation_set)
            data,data_label = create_matrix(classifier, training_set)

            theta0 = np.random.normal(0,0.0,1025)
            theta = grad_descent(f, gradf, data, data_label, theta0, alpha,max_iter)
            perform_test.append(performance(test, theta, tlabel))
            perform_training.append(performance(data, theta, data_label))
            perform_validation.append(performance(validation, theta, vlabel))
        
        perform_test_mean.append(np.mean(perform_test))
        perform_validation_mean.append(np.mean(perform_validation))
        perform_training_mean.append(np.mean(perform_training))
        
        perform_test_err.append(np.std(perform_test))
        perform_validation_err.append(np.std(perform_validation))
        perform_training_err.append(np.std(perform_training))
        
        print "Performance on the test set is %f+/-%f when training set size is %d." %(np.mean(perform_test),np.std(perform_test),i*6)
    
    plt.errorbar(training_set_size,perform_test_mean,yerr=perform_test_err,fmt='r.-')
    plt.errorbar(training_set_size,perform_validation_mean,yerr=perform_validation_err,fmt='g.-')
    plt.errorbar(training_set_size,perform_training_mean,yerr=perform_training_err,fmt='b.-')
    plt.xlabel('Training Set Size')
    plt.ylabel('Performance')
    plt.legend(['test', 'validation','training'])
    plt.savefig('figures/part5f1.jpg')
    plt.show()
    
def part6_f(x, y, theta):
    cost = np.matmul(theta.T, x) - y
    return np.trace(np.matmul(cost, cost.T))

def part6_df(x, y, theta):
    cost = np.matmul(theta.T, x) - y
    
    return 2*np.matmul(x, cost.T)

def finite_diff(x, y, theta, h, p, q):
    theta_h = theta.copy()
    theta_h[p][q] += h
    return (part6_f(x, y, theta_h) - part6_f(x, y, theta))/h

'''
Compare gradient components between finite-differences vs my function
'''
def part6_grad_error(x, y, theta, h, p, q):
    approx = finite_diff(x, y, theta, h, p, q)
    actual = part6_df(x, y, theta)[p][q]
    #error(a,b) = |a-b|/(|a|+|b|)
    error = np.abs(approx - actual)/(np.abs(approx) + np.abs(actual))
    return error

def part6_create_matrix(classifier, data_set):
    
    feature = np.array([])
    label = np.array([])
    feature = feature.reshape(1025, 0)
    label = label.reshape(6,0)
    for actor in classifier.keys():
        for image in data_set[actor]:
            #gather the score for the images(always correct score)
            label = np.hstack((label, np.array(classifier[actor])))
            im = imread(image)
            #divide image input by 255 and arrange it by 1024x1
            im = im[:,:,0]
            im = (im/255.).reshape(1024,1)
            #x_0 = 1
            im = np.vstack((np.array([1]), im))
            #gather the images pixels
            feature = np.hstack((feature, im))
            
    return feature, label

def part6():
    try:
      if not os.path.exists("figures/"):
          os.makedirs("figures/")
    except OSError:
      print ("Error: Creating directory. " + "figures/")
    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
    training_set, validation_set, test_set = part2(act, 80, 10, 10)
    
    classifier = {'Lorraine Bracco':[[1],[0],[0],[0],[0],[0]], 'Peri Gilpin':[[0],[1],[0],[0],[0],[0]], 'Angie Harmon':[[0],[0],[1],[0],[0],[0]], 'Alec Baldwin':[[0],[0],[0],[1],[0],[0]], 'Bill Hader':[[0],[0],[0],[0],[1],[0]], 'Steve Carell':[[0],[0],[0],[0],[0],[1]]}
    
    data, dlabel = part6_create_matrix(classifier, training_set)
    #validation, vlabel = part6_create_matrix(classifier, validation_set)
    #test, tlabel = part6_create_matrix(classifier, test_set)
    
    np.random.seed(0)
    #6x1 vector for classifier
    theta0 = np.random.normal(0, 0.0, (1025, 6))    
    mean_error = []
    std_error = []
    h = 1e-13
    for h in [10**x for x in range(-13, -5)]:
        error = []
        for i in range(10):
            #random 5 coordinates
            p = int(round((theta0.shape[0]-1)*np.random.rand()))
            q = int(round((theta0.shape[1]-1)*np.random.rand()))
            error.append(part6_grad_error(data, dlabel, theta0, h, p, q))
        mean_error.append(np.mean(error))
        std_error.append(np.std(error))
        
    f, ax = plt.subplots()
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    ax.errorbar([10**x for x in range(-13, -5)], mean_error, yerr = std_error, fmt='.-')
    plt.xlabel("Value of h")
    plt.ylabel("Average Relative Error over 5 coordinates")
    plt.savefig("figures/part6.jpg")
    plt.show()

if __name__ == "__main__":
    part6()
  #part1
  #act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
  #act = ['Kristin Chenoweth', 'Fran Drescher', 'America Ferrera', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan']
  #part1("facescrub_actors.txt", act)
'''
  act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
  a,b,c = part2(act, 30, 10, 10)
  print(a)
  print('\n')
  print(b)
  print('\n')
  print(c)
'''
  #part3
'''
  print("part3")
  
  part3()
'''
  #part4a
  #part4_a()  
  #part4b
  #part4_b()
  #part5()
