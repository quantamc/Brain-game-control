from fet_ex import *


features = []
lables = []

# loading all the data files
path = "./"
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
print y

# encoding the text labels for math operations
class_labels = np.unique(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# flatten the input arrray
x = flattener(x)


size = len(x)

# split the data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 0)

# Scale shift the data to standardize the dataset
scaler = StandardScaler()

# fitiing with the training data
scaler.fit(X_train)
# now for transforming the data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

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
    

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV

polynomial_svm_clf = Pipeline((("poly_features", PolynomialFeatures(degree=3)),
                             ("svm_clf", LinearSVC(C=10, loss='hinge'))))

polynomial_svm_clf.fit(X_train, y_train)

params = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, params)
clf.fit(X_train, y_train)

svm_score = polynomial_svm_clf.score(X_test, y_test)
tuned_score = clf.score(X_test, y_test)
print("svm score:" + str(svm_score))
print("Tuned svm score:" + str(tuned_score))

# Decision trees
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier(criterion='entropy')
tree_clf.fit(X_train, y_train)

tree_score = tree_clf.score(X_test, y_test)

print ("Tree score:" + str(tree_score))

from sklearn.metrics import confusion_matrix
import itertools
# Compute confusion matrix for SVM
y_pred = clf.predict(X_test)
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