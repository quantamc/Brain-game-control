import numpy as np   # matrix operations
import pandas as pd # for moving the data around
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import tensorflow as tf

# loading all the data files
path = "./"
filenames = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in filenames:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

# drop the duplicates if any
emo_df = frame.drop_duplicates()
instances = emo_df.shape[0]
print emo_df
# grabbing the sensor values
x = emo_df.ix[:,2:-1].values
y = emo_df['state']
size = y.shape[0]

# encoding the text labels for math operations
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# one hot ecode the labels
label_seq = np.unique(y)

onehot_encoded = list()
for value in y:
    label_ = [0 for _ in range(len(label_seq))]
    label_[value] = 1
    onehot_encoded.append(label_)

y_ = np.asarray(onehot_encoded)
labels = np.asmatrix(y_)

# split the data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.4, random_state = 0)

# Scale shift the data to standardize the dataset
scaler = StandardScaler()

# fitiing with the training data
scaler.fit(X_train)
# now for transforming the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


tf.reset_default_graph

batch_size = 10

X = tf.placeholder(tf.float32, [None, 14])  #placeholder for the input layer
W = tf.Variable(tf.zeros([14, 2]))  # the weights of the linear layer
b = tf.Variable(tf.zeros([2]))   # bias of the linear layer

init = tf.global_variables_initializer()

# model
Y = tf.nn.softmax(tf.matmul(X, W) + b)  # compute the model predictions

# Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 2])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1)) # Is the model's prediction correct?
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # compute teh average accuracy

optimizer = tf.train.GradientDescentOptimizer(0.003) #learning rate
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session() #create a session
sess.run(init) # Initialize the variables


avg_train_cost = 0. 
avg_train_accuracy = 0.
avg_test_cost = 0.
avg_test_accuracy = 0.


# initialize slice parameters for batching
start = 0
end = batch_size-1
num_steps = size/batch_size   # How many training steps do we want?
for i in range(num_steps):
    # load of batch of images and correct answers
    batch_X = X_train[start:end]
    batch_Y = y_train[start:end] # get a batch of signal instances
    train_data = {X:batch_X, Y_:batch_Y}
    # train
    sess.run(train_step, feed_dict=train_data) # Run the training
    
    # sucess ?
    a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
    
    
    
    if i % 10 == 0:
        print "Step", i, "Current train cost and accuracy: ", c, a
    
    avg_train_cost += c
    avg_train_accuracy += a
    # update the slice params
    start += 10
    end += 10
    
print("Completed Training")

avg_train_cost /= num_steps
avg_train_accuracy /= num_steps

print("Average train cost: ", avg_train_cost)
print("Average train accuracy: ", avg_train_accuracy)

# How well did we do on the test data?
test_data = {X:X_test, Y_:y_test}
a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

print("Test cost: ", c)
print("Test accuracy: ", a)