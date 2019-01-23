
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

      
def part1(faces_file_name):
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
  act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
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
                      cropped = rgb2gray(cropped)    
                      #Convert image to grayscale and resize image to 32X32
                      result = imresize(cropped, (32,32))
                      #save cropped image to the cropped folder, change heatmap version to grayscale
                      plt.imsave("cropped/"+filename, result, cmap="gray")
                  except:
                      print("Unable to load image: "+ filename)
                      continue
              i += 1
def part2(act, training, validation, test):
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
    
    for a in act:
        training_set[a] = []
        validation_set[a] = []
        test_set[a] = []
        name = a.split()[1].lower()
        #get the all filenames for the actor
        files = glob.glob('uncropped/' + name + '*')
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

if __name__ == "__main__":
  #part1("facescrub_actresses.txt")
  act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Alec Baldwin', 'Bill Hader', 'Steve Carell']
  a,b,c = part2(act, 30, 10, 10)
  print(a)
  print('\n')
  print(b)
  print('\n')
  print(c)