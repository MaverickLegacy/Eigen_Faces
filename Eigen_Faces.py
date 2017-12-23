import numpy as np
import cv2
from skimage import data
import os,sys
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
#from ipywidgets import interact, fixed, FloatSlider, IntSlider, Label, Checkbox, FloatRangeSlider
from numpy.linalg import svd 

def read_one_image_and_convert_to_vector(file_name):
    image = cv2.imread(file_name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

class EigenFaces(object):
    def __init__(self, training_dir, test_dir):
        self.filedir = os.getcwd()
        self.train_folder = training_dir
        self.test_folder = test_dir
        self.load_data(training_dir)
        self.calc_eigenvec()
        self.threshold = 6000
    
    def load_data(self,directory,test = False):
     if test != True:
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        self.train_file_names = files
        #print(files)
        os.chdir(directory)        
        self.img_size = read_one_image_and_convert_to_vector(files[0]).shape
        self.no_of_samples = len(files)
        self.train_data = np.zeros((self.no_of_samples,self.img_size[0],self.img_size[1]))
        self.train_img_vec_form = np.zeros((np.product(self.img_size),self.no_of_samples))
        for i in range(0,self.no_of_samples):
            self.train_data[i,:,:] = read_one_image_and_convert_to_vector(files[i])
            self.train_img_vec_form[:,i] = np.ravel(self.train_data[i,:,:])
     else:
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        self.test_file_names = files
        os.chdir(directory)        
        del(files[len(files)-1])
        self.test_data = np.zeros((np.product(self.img_size),len(files)))
        for i in range(0,len(files)):
            self.test_data[:,i] = np.ravel(read_one_image_and_convert_to_vector(files[i]))
     os.chdir(self.filedir)   
     
    def calc_eigenvec(self):
        self.mean = np.sum(self.train_img_vec_form,axis = 1) / self.no_of_samples 
        self.mean = np.reshape(self.mean,(self.mean.shape[0],1))
        self.mat = self.train_img_vec_form-self.mean
        self.eig_vec, self.eig_val, V = svd(self.mat, full_matrices=False)
        
    def display_eigen_faces(self):
        fig, axes_array= plt.subplots(1,1)
        fig.set_size_inches(5,5)
        img_plot = axes_array.imshow(np.reshape(self.mean,self.img_size),cmap=plt.cm.gray)
        plt.show()
        fig, axes_array = plt.subplots(5,5)
        fig.set_size_inches(5,5)
        c = 0
        for i in range(5):
            for j in range(5):
                image = np.reshape(self.eig_vec[:,c],(425,425))
                img_plt = axes_array[i][j].imshow(image,cmap = plt.cm.gray)
                axes_array[i][j].axis('off')
                c+=1         
        plt.show()                           
    
    def select_top_k_display(self,k=2):    
        size = (100,100)
        fig, axes_array = plt.subplots(5,5)
        fig.set_size_inches(5,5)
        self.weights = np.dot(self.mat.T, self.eig_vec)
        recon = self.mean + np.dot(self.eig_vec[:,:k],self.weights[:,:k].T)
        c=0
        for i in range(5):
            for j in range(5):
                image = np.reshape(recon[:,c],(425,425))
                img_plt = axes_array[i][j].imshow(image,cmap = plt.cm.gray)
                axes_array[i][j].axis('off')  
                c+=1
        plt.show()
    # input image is vectorised 
    def classify_face(self,k):
        self.load_data(self.test_folder,True)
        #fig, axes_array = plt.subplots(self.test_data.shape[1],2)#self.test_data.shape[1]
        eig_weights = np.dot(self.eig_vec[:,:k].T , self.mat)    
        for i in range(self.test_data.shape[1]):#
            fig, axes_array = plt.subplots(1,2)
            fig.set_size_inches(5,5)
            ip_wvec = np.dot(self.eig_vec[:,:k].T, self.test_data[:,i:i+1] - self.mean)
            ed = np.sum((eig_weights - ip_wvec)**2,axis=0)
            nearest_img = np.argmin(np.sqrt(ed))
            axes_array[0].imshow(np.reshape(self.test_data[:,i],self.img_size),cmap = plt.cm.gray,title='Test Image')
            axes_array[0].axis('off')
            #axes_array[0].title('Test Image')
            if(ed[nearest_img]<self.threshold):
                axes_array[1].imshow(self.train_data[nearest_img,:,:],cmap = plt.cm.gray,title = 'Nearest Image')
            #axes_array[1].title('Nearest Image')
            axes_array[1].axis('off')
        plt.show()
    
    def plot_error_rate(self):
        self.load_data(self.test_folder,True)
        x = range(1,self.no_of_samples+1)
        error_rates =[] 
        #tvals = range(2000,30000,500)
        #thresholds = range(20000,5000,-500)
        for k in range(self.no_of_samples):
            eig_weights = np.dot(self.eig_vec[:,:k].T , self.train_img_vec_form)
            no_of_errors = 0
            for i in range(self.test_data.shape[1]):
                ip_wvec = np.dot(self.eig_vec[:,:k].T, self.test_data[:,i:i+1] - self.mean)
                ed = np.sum((eig_weights - ip_wvec)**2,axis=0)
                img_num = int(self.test_file_names[i][1:4])
                nearest_img = np.argmin(ed)    
                pred_num = int(self.train_file_names[nearest_img][1:4])
                #print(img_num,' ', pred_num)
                if(ed[nearest_img] < self.threshold):
                    if(img_num >81 or img_num != pred_num): 
                        no_of_errors += 1
                elif(ed[nearest_img] > self.threshold):    
                    if(img_num < 81):
                        no_of_errors += 1
            error_rates += [no_of_errors]                
        error_rates = np.array(error_rates) * 100 / self.test_data.shape[1]
        plt.plot(x,error_rates)
        plt.xlabel("K")
        plt.ylabel("Error Rate")
        plt.show()           
if __name__ == "__main__":
    cwd = os.getcwd()
    test_dir = cwd + os.path.sep +'Eigenfaces'+os.path.sep + "Test"
    training_dir = cwd + os.path.sep +'Eigenfaces'+os.path.sep +"Train"
    ef = EigenFaces(training_dir, test_dir)
    ef.classify_face(5)
