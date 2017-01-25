#based on tensorflow v12
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math

#load dataset
data = np.load('ORL_faces.npz') #assumption is that the data is unzipped
trainX = data['trainX']
testX = data['testX']
trainY= data['trainY']
testY= data['testY']

classes = 20 #Number of classes
#Reshape train and test labels to one hot vector
trainY = np.eye(classes)[trainY]
testY  = np.eye(classes)[testY]

# Parameters
learning_rate = 1e-4
max_epochs = 100
display_step_size = 10 # Number of iterations before checking on the perfomance of network (validation)
prob_keep = 0.5

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16
mx_pooling_size = 2        #max pooling size 2 *2

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36

# Fully-connected layer.
fc_neurons = 1024         # Number of neurons in fully-connected layer.

#Image properties
img_shape = (112, 92)
channels = 1

#create place holders
X = tf.placeholder(tf.float32, shape=[None, img_shape[0]*img_shape[1]], name='X')
X_image = tf.reshape(X, [-1, img_shape[0], img_shape[1], channels])
Y = tf.placeholder(tf.float32, shape=[None, classes], name='Y')
Y_classes = tf.argmax(Y, dimension=1)


def generate_weights(shape):
    # Create new matrix
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

def generate_biases(size):
    #create baises
    return tf.Variable(tf.constant(0.01, shape=[size]))


def convolution(input_data,num_channels, filter_size, num_filters): # compute convolutions with relu output
    #shape for weights
    shape = [filter_size, filter_size, num_channels, num_filters]
    # Generate new weights
    W = generate_weights(shape=shape)
    # generate new biases, one for each filter.
    b= generate_biases(size=num_filters)
    #tensorflow convolution
    out = tf.nn.conv2d(input=input_data, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    # Add the biases
    out= tf.nn.bias_add(out,b)
    #relu activation
    out = tf.nn.relu(out)
    return out, W

def max_pooling(input_data,size): #max pooling layer
    out = tf.nn.max_pool(value=input_data, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')
    return out

def reduce(tensor):
    #reduce the 4-dim tensor, the output from the
    #conv/maxpooling to 2-dim as input to the fully-connected layer
    features = tensor.get_shape()[1:4].num_elements() # The volume
    reduced = tf.reshape(tensor, [-1, features])
    return reduced, features

def compute_fc_layer(input_data,input_size, output_size, use_relu=True, user_dropout=False):
    # generate new weights and biases.
    W = generate_weights(shape=[input_size, output_size])
    b = generate_biases(size=output_size)
    #compute the out
    out = tf.matmul(input_data, W) + b
    # Use ReLU?
    if use_relu:
        out = tf.nn.relu(out)
    #Add dropout regularisation if its not the out layer
    if user_dropout:
         out = tf.nn.dropout(out, prob_keep)
    return out


#CNN architecture
#layer 1
conv_layer1, conv_W1 = convolution(input_data=X_image,num_channels=channels,
                                   filter_size=filter_size1,num_filters=num_filters1)

max_pooling_layer_1 = max_pooling(input_data=conv_layer1,size=mx_pooling_size)

#Layer 2
conv_layer2, conv_W2 =convolution(input_data=max_pooling_layer_1,num_channels=num_filters1,
                                  filter_size=filter_size2,num_filters=num_filters2)

max_pooling_layer_2 = max_pooling(input_data=conv_layer2,size=mx_pooling_size)

#reshape the out from covolution layers for input into fully connected layers
Xi, features = reduce(max_pooling_layer_2)

#Fully connected layers
FC1 = compute_fc_layer(input_data=Xi,input_size=features,
                       output_size=fc_neurons,use_relu=True,user_dropout=True)

FC2 = compute_fc_layer(input_data=FC1,input_size=fc_neurons,
                       output_size=classes, use_relu=False, user_dropout=False)

output = tf.nn.softmax(FC2) #softmax output
pred = tf.argmax(output, dimension=1) # predictions
#compute cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=FC2, labels=Y))
#optimse the cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#Compute Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y_classes), tf.float32))


#tensorflow session
init = tf.global_variables_initializer()
training_loss = []  # training loss
test_loss =[[],[]]  #Keep test loses at evey epoch
training_accuracy =[0] #Keep training accuracy
test_accurracy = [0]  # accuracy on test data



#run session
with tf.Session() as sess:
    sess.run(init)
    epoch =0

    while epoch < max_epochs+1:
       #get a mini batch of size batchsize
        training_data = {X: trainX, Y:trainY }
        # run the optimser
        _, train_loss,train_acc = sess.run([optimizer, cost, accuracy], feed_dict=training_data)
        training_loss.append(train_loss)
       # Print status every 10 iterations.
        if epoch % display_step_size == 0:
            training_accuracy.append(train_acc * 100)
            # After every 10 epochs, check the accuracy on the test data
            test_data = {X: testX, Y: testY}
            test_acc,tst_loss = sess.run([accuracy,cost], feed_dict=test_data)
            test_accurracy.append(test_acc*100)
            test_loss[0].append(epoch)
            test_loss[1].append(tst_loss)

            print ("Epoch: {0:>3}, Training Loss:{1:>6.8f}, Training Accuracy:{2:>6.1%}, Test loss: {3:>6.8f}," \
                  "Test accuracy: {4:>6.1%}".format(epoch, train_loss, train_acc, tst_loss,test_acc))

        epoch +=1
        
    print ('Optimisation done. Accuracy attained on test data:{0:.2f}'.format(max(test_accurracy)))

    # Plots training loss

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(15, 6))
    ax1.plot(training_loss, color='red', label='Training loss')
    ax1.plot(test_loss[0], test_loss[1],color ='blue',label='Test loss')
    ax1.set_title('Loss: Training vs Test')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.margins(.05)
    ax1.legend(loc='best')
    ax1.grid()

    # plot test accuracy
    ax2.plot(np.array(range(len(training_accuracy))) * display_step_size, training_accuracy,
             color="red", label='Training: Acc ={0:.1f}%'.format(np.max(training_accuracy)))
    ax2.plot(np.array(range(len(test_accurracy))) * display_step_size, test_accurracy,
             color="blue", label='Test: Acc = {0:.1f}%'.format(np.max(test_accurracy)))
    ax2.set_title('Accuracy: Training vs Test')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy %')
    ax2.axis([0, max_epochs + 5, 0, 105])
    ax2.legend(loc='best')
    ax2.grid()

    def visualise_conv_filter(weights, session): 
        #Visualise filters

        conv_filter = session.run(weights) # get weights
        # sub plots gridlayout
        sub_plots_grids = int(math.ceil(math.sqrt(conv_filter.shape[3])))

        # plot figure with sub-plots an grids.
        fig, axs = plt.subplots(sub_plots_grids, sub_plots_grids)

        for i, ax in enumerate(axs.flat): #loop through all the filters

            if i < conv_filter.shape[3]: #check if a filter is valid

                img = conv_filter[:, :, 0, i] #format image

                ax.imshow(img,vmin=np.min(conv_filter), vmax=np.max(conv_filter))#plot image

            # Remove marks from the axis
            ax.set_yticks([])
            ax.set_xticks([])

    visualise_conv_filter(conv_W1, sess)
    visualise_conv_filter(conv_W2, sess)
    plt.show()


