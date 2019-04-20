# Fully connected layer
 I created an **MLP** after taking **Andrew-Ng** course which could be found [**_here_**](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning). 
 We learned how to build our own deep learning model from scratch, this could is an example of that. 

## Packages that needs to be installed:
There's a few packages one needs to install to get my **MLP** to work.
* numpy
* imageio
* scipy
* matplotlib
## How to use:
There's an example file on how I used it, but here's a quick run down.
**Loading in the Data & Reformatting**
~~~
x_train, y_train = DataLoader(train_path,resize_image=True,dims=(dims,dims)).import_image()
x_test, y_test = DataLoader(test_path,resize_image=True,dims=(dims,dims)).import_image()
~~~
**Flattening to feed into network,and normalizing..**
~~~
x_train_flatten = x_train.reshape(x_train.shape[0],-1).T/255
x_test_flatten = x_test.reshape(x_test.shape[0],-1).T/255
~~~

**Reshape the targets so its M x NX**
~~~
y_train_flat = y_train.reshape(y_train.shape[0],-1).T
y_test_flat = y_test.reshape(y_test.shape[0],-1).T
~~~
**Training the network**
~~~
model = Fully_connected(x_train_flatten, x_test_flatten, y_train_flat, y_test_flat,lr = .0001, epochs = 1550, n_hidden = [30,8], lambd=0)
pred, pred_test, weights = model.train(print_every = 10)
~~~
**Saving out the tuned Weights**
~~~
np.save(train_path+"weight02",weights)
~~~