import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


 # Define Net
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 (1,36) input
        self.fc1 = nn.Linear(784,100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100,10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.dropout(F.relu(self.fc1(x)),p=0.6,training=True) # with True we activate dropout for predictions
        x = F.dropout(F.relu(self.fc2(x)),p=0.6,training=True)
        x = F.log_softmax(self.fc3(x), dim=1) #softmax
        return x

def train_epoch(model, opt, X, Y, epoch, batch_size=64):
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch).long()
        opt.zero_grad()
        # Forward
        y_hat = net(x_batch)
        # Compute loss
        loss = F.nll_loss(y_hat, y_batch)
        # Compute gradients
        loss.backward()
        # update weights
        opt.step()
        losses.append(loss.data.numpy())

    print("epoch: ", epoch,"--- loss: ",np.mean(losses) )
    return np.mean(losses)

# Function to return mean, std and plot of noisy images
def noisy_predict(net, X_train,y_train, X_tensor, noise_std=100, n_images = 5, n_pred = 100 ):

    # add Gaussian noise to observations
    noise_ = noise_std*np.random.randn(X_tensor.shape[0],X_tensor.shape[1])
    noise = torch.from_numpy(noise_.astype('float')).float()
    X_noise = X_trainT + noise

    # Get prediction from noisy images
    pred_noise = []
    for i in range(0,n_pred):
        pred_noise.append(net(X_noise[0:5]).detach().numpy())

    # Transform log_softmax n softmax predictions
    pred_list = np.array(pred_noise)
    pred_list_ = np.exp(pred_list)*100

    # Calculate mean and std of softmax predicitons
    mean_pred = np.round(np.mean(pred_list_,axis=0),1)
    std_pred = np.round(np.std(pred_list_,axis=0),1)
    pred_values = np.argmax(mean_pred,1)
    pred_prob = np.round(np.max(mean_pred,1),0)
    pred_std_values = [std_pred[i,j] for i,j in zip(np.arange(0,5),np.argmax(mean_pred,1))]

    # Plot results
    plt.figure(figsize=(20,4))
    for index, (image, label) in enumerate(zip(X_train[0:5]+noise_[0:5], y_train[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
        plt.title('True: %i\n' % label + "Pred: "+str(pred_values[index])+" \n"+
                  "softmax: "+ str(pred_prob[index])+" +/- "+str(pred_std_values[index]) , fontsize = 15)
    plt.show()

if __name__ == "__main__":

    # Import MNIST dataset and split in train/test
    mnist = fetch_mldata('MNIST original')
    X_train, X_test, y_train, y_test = train_test_split(
     mnist.data, mnist.target, test_size=1/7.0, random_state=0)

     # Convert from numpy to torch tensors
    y_trainT = torch.from_numpy(y_train.astype('float')).float()
    X_trainT = torch.from_numpy(X_train.astype('float')).float()

    # Define Net
    net = Net()
    opt = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # Train
    print("training model...")
    e_losses = []
    num_epochs = 30
    for e in range(num_epochs):
        e_losses.append( train_epoch(net, opt, X_trainT, y_trainT, e, batch_size = 64) )

    print("get prediction...")
    noisy_predict(net, X_train,y_train, X_tensor=X_trainT, noise_std=100, n_images = 5, n_pred = 100 )
