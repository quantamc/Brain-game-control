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
emo_df = frame
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

starter_learning_rate = 0.01
training_iteration = 10000
batch_size = 10
display_step = 100

x = tf.placeholder(tf.float32, [None, 14]) # siganl data with shape 1*16
y = tf.placeholder(tf.float32, [None, 2]) # the 2 emotional states

# create a model

# layer values
K = 200
L = 100
M = 60
N = 30

# Set the model weights
W1 = tf.Variable(tf.truncated_normal([14, K], stddev=0.1)) # weigths of the liniear model
B1 = tf.Variable(tf.zeros([K])) # bias for layer K

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=0.1))
B5 = tf.Variable(tf.zeros([2]))

with tf.name_scope("Wx_b") as scope:
    # Constuct a model
    Y1 = tf.nn.relu(tf.matmul(x, W1) + B1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    model = tf.nn.softmax(tf.matmul(Y4, W5) + B5) # the softmax
    
# decaying the learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                          100000, 0.96, staircase=True)

    
# Summary opps to collect the data
w_h = tf.summary.histogram("weights", W1)
b_h = tf.summary.histogram("biases", B1)

# more namescopes to clean up graph representation
with tf.name_scope("cost_function") as scope:
    # Minimize error using cross entropy
    #cross entropy
    cost_function = -tf. reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)
    
with tf.name_scope("train") as scope:
    # Gradient descent and specify the global step to increment it
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function, global_step=global_step)
    
# % of correct answers found in batch

with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1)) # Is the model's prediction correct?
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # compute teh average accuracy
    
# inintalizing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single opearator
merged_summary_op = tf.summary.merge_all()

sess = tf.Session() #create a session
sess.run(init) # Initialize the variables
with tf.Session() as sess:
    sess.run(init)
    
    # set the logs writer to the folder /tmp/tensorflow-logs
    summary_writer = tf.summary.FileWriter('data/logs')

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        avg_train_accuracy = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        # initialize slice parameters for batching
        start = 0
        end = batch_size-1
        ## loop over all the batches
        for i in range(total_batch):
            batch_xs = X_train[start:end]
            batch_ys = y_train[start:end]
            # fit training using the batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            avg_train_accuracy += sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # write logs for each iteartion
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)

            # update the slice params
            start += 10
            end += 10
        # Display logs per iteration step
        if iteration % display_step == 0:
            print "iteration", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost), "accuracy=", "{:.9f}".format(avg_train_accuracy)
            

    print "Tuning complete"

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
    print "Accuracy:", accuracy.eval({x: X_test, y: y_test})

def plot_confusion_matrix(cm, classes, normalize=False,
                        title='Confusion matrix', 
                        cmap=plt.cm.Blues):

    """
    Prints the plots of the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    pass

from sklearn.metrics import confusion_matrix
import itertools
# Compute confusion matrix for SVM
y_pred = predictions
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#plot non-nomalized confusion matrix 
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_labels,
                        title='Confusion matrix, withot normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_labels, normalize=True,
                        title='Nomalized confusion matrix')

plt.show()