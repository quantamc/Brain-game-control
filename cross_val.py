import numpy as np   # matrix operations
import pandas as pd # for moving the data around
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import glob
# Decision trees
from sklearn import tree
from sklearn.model_selection import KFold



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

# Scale shift the data to standardize the dataset
scaler = StandardScaler()

# fitiing with the training data
scaler.fit(x)
# now for transforming the data
x = scaler.transform(x)


polynomial_svm_clf = Pipeline((("poly_features", PolynomialFeatures(degree=3)),
                             ("svm_clf", LinearSVC(C=10, loss='hinge'))))


tree_clf = tree.DecisionTreeClassifier(criterion='entropy')



svm_scores = cross_val_score(polynomial_svm_clf, x, y, cv=3, scoring='f1_macro')
tree_scores = cross_val_score(tree_clf, x, y, cv=10, scoring='f1_macro')

print("Accuracy: %0.2f (+/- %0.2f)" % (svm_scores.mean(),svm_scores.std() *2))
print("Accuracy: %0.2f (+/- %0.2f)" % (tree_scores.mean(),tree_scores.std() *2))


