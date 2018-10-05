import numpy as np   # matrix operations
import pandas as pd # for moving the data around
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import pywt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV



def load_data(frame):

    # drop the duplicates if any
    frame = frame.drop_duplicates()
    instances = frame.shape[0]
    # grabbing the sensor values
    x = frame.ix[:,2:-1].values
    y = frame['state']
    size = y.shape[0]
    print y
    class_label = np.unique(y)

    # align the xs
    x = x.T
    print x.shape
    # Descete wavelet transform

    labels = list(frame)
    print labels
    # extract the table labels
    print x.shape
    t = np.arange(0.0, instances, 2.6)
    #t = np.arange(0.0, instances, 1)

    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(20,10))
    for i in x:
        ax.plot(i)
        ax.hold('on')

    ax.set(xlabel='time(s)', ylabel='voltage (mV)',
       title='Raw EEG Signal')

    ax.legend(labels[:-1])

    ax.grid()

    plt.show()


    return x, class_label


def DWT(x):
    cA = []
    cD = []
    ts_rec = []
    for i in x:
        ca, cd = pywt.dwt(i, 'db4')
        cA.append(ca); cD.append(cd)

    cat = pywt.thresholding.soft(cA, np.std(cA)/2)
    cdt = pywt.thresholding.soft(cD, np.std(cA)/2)

    for i, j in zip(cat, cdt):
        ts = pywt.idwt(i, j, 'db4')
        ts_rec.append(ts)

    return cdt

def feature_extract(cdt, x):
    # feature extract

    print cdt.shape
    # the mean of the channel fequency
    means = np.mean(cdt, axis=1)
    # the varience
    var = np.var(cdt, axis=1, dtype=np.float64)
    # mean of energy
    energy = cdt**2
    mean_e = []

    for channel in energy:
        summation = sum(channel)
        mean_e.append(summation)

    # Max Amplitude:
    max_amp = []

    for channel in x:
        maxi_amp = max(channel)
        max_amp.append(maxi_amp)

    # min Amplitude
    min_amp = []

    for channel in x:
        mini_amp = min(channel)
        min_amp.append(mini_amp)

    # min Energy
    min_e = []

    for channel in cdt:
        mini_e = min(channel)
        min_e.append(mini_e)

    # Max energy
    max_e = []

    for channel in cdt:
        maxi_e = min(channel)
        max_e.append(mini_e)

    # min frequecy
    min_freq = []

    for channel in cdt:
        mini_freq = min(channel)
        min_freq.append(mini_freq)

    # max frqunecy
    max_freq = []

    for channel in cdt:
        maxi_freq = max(channel)
        max_freq.append(maxi_freq)

    return means, var, mean_e, max_amp, min_amp, min_e, max_e, min_freq, max_freq

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

def flattener(x):

    """
    Returns a batch of training examples
    """
    arr = []

    x = np.asarray(x)
    for i in x:
        x_ = i.flatten()
        arr.append(x_)

    return np.asarray(arr)



if __name__ == '__main__':

    features = []
    lables = []

    # loading all the data files
    path = "./ag"
    filenames = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in filenames:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)


    #work on each dataframe

    for frame in list_:
        x, label = load_data(frame)
        DWT_coeff = DWT(x)
        values = feature_extract(DWT_coeff, x)

        # save to numpy array
        features.append([values])
        lables.append(label)


    x = features
    y = lables
    #np.savetxt("features", x, '%s')
    #np.savetxt("labels", y, '%s')

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
    y = np.asmatrix(y_)

    # flatten the input arrray
    x = flattener(x)
    print(x.shape)
    print(y.shape)
    print x
    print y[0:2].shape


    size = len(x)

    # split the data into testing and training data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 0)


    import tensorflow as tf

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, 126])
    W = tf.Variable(tf.zeros([126, 2]))
    b = tf.Variable(tf.zeros([2]))

    init = tf.global_variables_initializer()

    # model
    Y = tf.nn.softmax(tf.matmul(X, W) + b)  # compute the predictions of the model

    # Placeholder for corrct answers
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

    batch_size = 2
    start = 0
    end = batch_size
    num_steps = size/batch_size

    for i in range(num_steps):
        #load batch of signal states
        batch_X = X_train[start:end]
        batch_Y = y_train[start:end]
        train_data = {X:batch_X, Y_:batch_Y}
        # train
        sess.run(train_step, feed_dict=train_data)


        # succes?
        a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

        if i % 2 == 0:
            print "Step", i, "current train cost and accuracy: ", c, a

        avg_train_cost += c
        avg_train_accuracy += a
        # update the slice params
        start += batch_size
        end += batch_size

    print ("Completed Training")

    avg_train_cost /= num_steps
    avg_train_accuracy /= num_steps

    print("Average train cost: ", avg_train_cost)
    print("Average train accuracy: ", avg_train_accuracy)

    # How well did we do on the test data?
    test_data ={X:X_test, Y_:y_test}
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

    print("Test cost: ", c)
    print("test accuracy: ", a)
