import numpy as np
import imageio
import glob
import os
from scipy.misc import imresize
import matplotlib.pyplot as plt
'''
This is framework for multiple Neural Networks. 
The first is Fully connected Node with 3 layers

The DataLoader  is used to import images save weights and so on..
Author @ Ambrose Mcduffy
'''
class Fully_connected(object):
# This is a fully connected MLP with 3 layers
    def __init__(self,input=None,test_input=None,targets=None,test_targets=None,lr=0.01,epochs=8,n_hidden=[1,1], num_class=1,lambd = .7):
        self.input = input
        self.targets = targets
        self.test_input = test_input
        self.test_targets = test_targets
        self.lr = lr
        self.epochs = epochs
        self.n_hidden = n_hidden
        self.num_class = num_class
        self.lambd = lambd
    def sigmoid(self,x):
        return 1./(1.+np.exp(-x))
    def relu(self,x):
        return np.maximum(0,x)
    def softmax(self,x):
        esp = np.exp(x-max(x))
        return esp/(sum(esp))
    def flat_vec(self,x,y):
        return (self.input.reshape(x.shape[0],-1).T, self.input.reshape(x.shape[0],-1).T)
    def init_parameters(self,):
        # Initializing the parameters with random values & Using L2 Regularization
        w1 = np.random.randn(self.n_hidden[0], self.input.shape[0]) * np.sqrt(1./self.input.shape[0])
        w2 = np.random.randn(self.n_hidden[1], self.n_hidden[0]) * np.sqrt(1./self.n_hidden[0])
        w3 = np.random.randn(self.num_class,self.n_hidden[1]) * np.sqrt(1./self.n_hidden[1])
        b1 = np.zeros((self.n_hidden[0],1))
        b2 = np.zeros((self.n_hidden[1],1))
        b3 =  np.zeros((1,1))
        parameters = {"w1":w1,
        "w2":w2,
        "w3":w3,"b1":b1,"b2":b2,"b3":b3}
        return parameters
    def initialization_momentum(self,parameters):
        # This is a setup for Adam optimizer
        L = 3
        v = {}
        s = {}
        l2_dims = [self.input.shape[0], self.n_hidden[0], self.n_hidden[1]]
        for l in range(L):
            v["dw"+str(l+1)] = np.zeros_like(parameters["w"+str(l+1)]) * np.sqrt(1./l2_dims[l])
            s["dw"+str(l+1)] = np.zeros_like(parameters["w"+str(l+1)])* np.sqrt(1./l2_dims[l])
        return v, s
    def optimize_parameters(self,parameters, grads,v,s,beta=.9, beta2=.99):
        # Updating the weights
        L = 3
        parameters = parameters
        for l in range(L):
            v["dw"+str(l+1)] = beta * v["dw"+str(l+1)] + (1-beta) * grads["dw"+str(l+1)]
            s["dw"+str(l+1)] =  beta2 * s["dw"+str(l+1)] +(1-beta2) * (grads["dw"+str(l+1)] **2)
        for l in range(L):
            parameters["w"+str(l+1)] = parameters["w"+str(l+1)] - self.lr * v["dw"+str(l+1)]/np.sqrt(s["dw"+str(l+1)] + 1e-8)
        return parameters
    def loss(self,yhat,y, parameters):
        # Cross entropy loss with L2 Regularization
        xentropy = (-1./self.input.shape[1]) * np.sum(y*np.log(yhat) + (1.-y) * np.log(1.-yhat))
        l2_ = self.lambd * (np.sum(np.square(parameters["w1"])) +  np.sum(np.square(parameters["w2"])) +  np.sum(np.square(parameters["w3"]))) / (2*self.input.shape[1])
        cost = xentropy + l2_
        return cost 
    
    def forward(self,x,parameters):
        # Forward pass.. l1,l2,l3 are the logits..
        l1 = np.dot(parameters["w1"],x) + parameters["b1"]
        h1 = np.tanh(l1)
        l2 = np.dot(parameters["w2"], h1) + parameters["b2"]
        h2 = np.tanh(l2)
        l3 = np.dot(parameters["w3"], h2) + parameters["b3"]
        yhat = self.sigmoid(l3)
        cache = (h1,h2,yhat,l1,l2,l3)
        return yhat, cache
    def backprop(self,yhat,parameters,cache):
        h1,h2,_,l1,l2,l3 = cache
        error = yhat - self.targets
        db3 =  1.0/self.input.shape[1] * np.sum(error, axis=1, keepdims=True)
        dw3 = 1.0/self.input.shape[1] * np.dot(error,h2.T) + (self.lambd*parameters["w3"])
        dh3 = np.dot(parameters["w3"].T,error) * (1-(h2**2))
        db2 =  1.0/self.input.shape[1] * np.sum(dh3, axis=1, keepdims=True)
        dw2 = 1.0/self.input.shape[1] * np.dot(dh3, h1.T) + (self.lambd*parameters["w2"])
        dh2 = np.dot(parameters["w2"].T, dh3) * (1-(h1**2))
        db1 =  1.0/self.input.shape[1] * np.sum(dh2, axis=1, keepdims=True)
        dw1 = 1.0/self.input.shape[1] * np.dot(dh2, self.input.T) + (self.lambd*parameters["w1"])
        grads = {"dw1":dw1, "dw2":dw2,"dw3":dw3, "db1":db1,"db2":db2,"db3":db3}
        return grads
    
    def optimization(self,grads, parameters):
        parameters["w1"]= parameters["w1"] - self.lr * grads["dw1"]
        parameters["w2"]= parameters["w2"] - self.lr * grads["dw2"]
        parameters["w3"]= parameters["w3"] - self.lr * grads["dw3"]
        parameters["b1"]= parameters["b1"] - self.lr * grads["db1"]
        parameters["b2"]= parameters["b2"] - self.lr * grads["db2"]
        parameters["b3"]= parameters["b3"] - self.lr * grads["db3"]
        return parameters
    
    def train(self, print_every=100):
        parameters = self.init_parameters()
        v,s = self.initialization_momentum(parameters)
        l_epochs = []
        l_loss = []
        l_loss_test =[]
        for e in range(self.epochs):
            yhat, cache = self.forward(self.input,parameters)
            grads = self.backprop(yhat, parameters,cache)
            #parameters = self.optimization(grads, parameters)
            parameters= self.optimize_parameters(parameters, grads,v,s,beta=.9, beta2=.99)
            yhat_test, cache_test = self.forward(self.test_input,parameters)
            if e % print_every == 0:
                train_accuracy = 100-(np.mean(abs(yhat-self.targets)) * 100)
                test_accuracy = 100-(np.mean(abs(yhat_test-self.test_targets)) * 100)
                l_epochs.append(e)
                l_loss.append(self.loss(yhat, self.targets, parameters))
                l_loss_test.append(self.loss(yhat_test, self.test_targets, parameters))
                print("Epoch: {} ---> training loss --> {} training accuracy ---> {} test_loss ---> {} test_accuracy ---> {}".format(e,self.loss(yhat, self.targets,parameters),train_accuracy, self.loss(yhat_test, self.test_targets, parameters), test_accuracy))
        plt.xlabel("Epochs")
        plt.ylabel("ERROR")
        plt.title("Overall performance")
        plt.plot(l_epochs, l_loss,l_epochs,l_loss_test)
        return yhat, yhat_test, parameters
class DataLoader(object):
    '''
    this class loads input data from images..
    if one wants class types add class name at the beginning of an image
    example: 0_cat.jpg or 1_dog.jpg
    
    the images keeps aspect ratio
    
    '''
    def __init__(self,path_,resize_image = False, dims = (None,None)):
        self.path_ = path_
        self.resize_image = resize_image
        self.dims = dims
    def import_image(self):
        '''
        this module imports the images, if resize ==True it will resize to designated
        dimensions 
        '''
        # looking for jpg files
        jpgfiles_end = len([name for name in os.listdir(self.path_) if name.endswith(".jpg")])
        # grabbing the names of the files.. is the directory..
        names_ = [name for name in os.listdir(self.path_) if name.endswith(".jpg")]
        print(jpgfiles_end)
        try:
            if os.path.exists(self.path_) == True:
                print("Importing in images...")
                label = []
                dataset = []
                for i in range(jpgfiles_end):
                    paths = glob.glob(self.path_+names_[i])[0]
                    filename = names_[i]
                    label.append(int(filename[0]))
                    data = imageio.imread(paths)
                    # resizing the image data
                    if self.resize_image == True:
                          w,h,c = data.shape
                          if h > w:
                                new_h, new_w = (int(self.dims[0]*h/w),self.dims[0])
                          elif h < w:
                                new_h, new_w = (self.dims[0],int(self.dims[0]*w/h))
                          else:
                                new_h, new_w = (self.dims[0],self.dims[0])
                          data = imresize(data,  size= (new_w,new_h))
                          data = data[:self.dims[0],:self.dims[1]]
                          dataset.append(data)
                    else:
                          dataset.append(data)
                print("\n\tImage import is complete!")
            return np.array(dataset), np.array(label)
        except:
            print("Path incorrect or do not exists. \nPlease check the file path!")
            return None
    def load_weights(self,):
        return np.load(self.path).item()
    def save_weights(self,name,weights):
        return np.save(self.path+name,weights)
