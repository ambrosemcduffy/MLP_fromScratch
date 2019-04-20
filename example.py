from ambrose_flow import Fully_connected
from ambrose_flow import DataLoader
import matplotlib.pyplot as plt
import numpy as np
# Image files
train_path = "data/train_set/"
test_path = "data/test_set/"
dims = 48
# Loading in the data & reformatting
x_train, y_train = DataLoader(train_path,resize_image=True,dims=(dims,dims)).import_image()
x_test, y_test = DataLoader(test_path,resize_image=True,dims=(dims,dims)).import_image()

# Flattening to feed into networkm and normalizing..
x_train_flatten = x_train.reshape(x_train.shape[0],-1).T/255
x_test_flatten = x_test.reshape(x_test.shape[0],-1).T/255
# Reformatting the targets so its m x nx
y_train_flat = y_train.reshape(y_train.shape[0],-1).T
y_test_flat = y_test.reshape(y_test.shape[0],-1).T
# Training the network
model = Fully_connected(x_train_flatten, x_test_flatten, y_train_flat, y_test_flat,lr = .0001, epochs = 1550, n_hidden = [30,8], lambd=0)
pred, pred_test, weights = model.train(print_every = 10)
# saving out the weights
np.save(train_path+"weight02",weights)

# setting plotting
idx = 13
row = 1
column = 6
fig = plt.figure(figsize=(13,8))
predictions = []
height = []
# predicting on test..
# images I want to produce..
index = [1,8,20,25,5,27]
for i in range(1,row*column+1):
      # obtaining the index number
      idx = index[i-1]
      # grabbing associated image
      img = x_test[idx]
      # appending the image number
      height.append(i-1)
      # setting up plotting
      fig.add_subplot(row,column,i)
      plt.xlabel("image"+" " + str(i-1))
      plt.imshow(img)
      # normalizing the inputs
      img = img/255.
      # expanding the dimensions to be m*nh*nw*nc
      img = np.expand_dims(img, axis = 0)
      # flattening the inputs so it can be passed in the network
      img_flatten = img.reshape(img.shape[0],-1).T
      # obtaining the predictions from saved weights
      pred_val,_ = model.forward(img_flatten,weights)
      if pred_val >= 0.6:
            print("\n\nimage"+" " + str(i-1) + " " +" is the rock!")
      predictions.append(pred_val[0][0])
#Plotting Predictions..
plt.show()
plt.bar(height,predictions)
plt.xlabel("images")
plt.ylabel("predictions")
plt.title("image predictions plot")
