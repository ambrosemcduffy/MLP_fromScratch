from ambrose_flow import Fully_connected
from ambrose_flow import DataLoader
import numpy as np
# Image files
train_path = "C:\\Users\\ambro\\Desktop\\Subi_dataset\\train_set\\"
test_path = "C:\\Users\\ambro\\Desktop\\Subi_dataset\\test_set\\"
# Loading in the data & reformatting
x_train, y_train = DataLoader(train_path,resize_image=True,dims=(100,100)).import_image()
x_test, y_test = DataLoader(test_path,resize_image=True,dims=(100,100)).import_image()

# Flattening to feed into networkm and normalizing..
x_train = x_train.reshape(x_train.shape[0],-1).T/255
x_test = x_test.reshape(x_test.shape[0],-1).T/255
# Reformatting the targets so its m x nx
y_train = y_train.reshape(y_train.shape[0],-1).T
y_test = y_test.reshape(y_test.shape[0],-1).T 
# Training the network
model = Fully_connected(x_train, x_test, y_train, y_test,lr = .001, epochs = 600, n_hidden = [80,20], lambd=.2)
pred, pred_test, weights = model.train(print_every = 100)
# saving out the weights
np.save(train_path+"weight02",weights)