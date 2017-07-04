#encoding: utf-8
# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph. I like it most for plot%matplotlib inline
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Any results you write to the current directory are saved as output.
# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
# here header 0 means the 0 th row is our coloumn
# header in data
data = pd.read_csv('D:\data.csv',header=0)
#print(data.head(3))
# now lets look at the type of data we have. We can use
data.info()
# now we can drop this column Unnamed: 32
data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself
# if you want to save your old data then you can use below code
# data1=data.drop("Unnamed:32",axis=1)
# here axis 1 means we are droping the column
print(data.columns)
# like this we also don't want the Id column for our analysis
data.drop("id",axis=1,inplace=True)
# As I said above the data can be divided into three parts.lets divied the features according to their category
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:21])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
# lets now start with features_mean
# now as you know our diagnosis column is a object type so we can map it to integer value
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
print(data.describe()) # this will describe the all statistical function of our data
# lets get the frequency of cancer stages
sns.countplot(data['diagnosis'], label="Count")
#plt.show()
#Data Analysis a little feature selection
# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are
# dependenig on each other so we should avoid it because what is the use of using same column twice
# lets check the correlation between features
# now we will do this analysis only for features_mean then we will do for others and will see who is doing best
corr = data[features_se].corr() # .corr is used for find corelation
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features_mean, yticklabels= features_mean,
           cmap= 'coolwarm')
#plt.show()
#选择特征值
# now these are the variables which will use for prediction
prediction_var =['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                 'concave points_se','symmetry_se', 'fractal_dimension_se']
#now split our data into train and test
train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train.shape)
print(test.shape)
train_X = train[prediction_var]# taking the training data input
train_y=train.diagnosis# This is output of our training data
# same we have to do for test
test_X= test[prediction_var] # taking test data inputs
test_y =test.diagnosis   #output value of test dat
#数据分割写进CSV文件中
#train_data = pd.DataFrame(train,columns=['diagnosis','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
 #                'concave points_se','symmetry_se', 'fractal_dimension_se'])
#train_data.to_csv(r'D:\tain_data.csv',index=None)
#test_data = pd.DataFrame(test,columns=['diagnosis','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
#                 'concave points_se','symmetry_se', 'fractal_dimension_se'])
#test_data.to_csv(r'D:\test_data.csv',index=None)
#导入数据
import tflearn
from tflearn.data_utils import load_csv
X_data, labels = load_csv('tain_data.csv', target_column=0,
                        categorical_labels=True, n_classes=2)
train_data = np.array(X_data,dtype=float)
# Build neural network
net = tflearn.input_data(shape=[None, 10])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='sigmoid')
net = tflearn.regression(net,optimizer='adam',loss='categorical_crossentropy')
# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(train_data, labels, n_epoch=200, batch_size=8, show_metric=True)
#测试数据的测试
y_data, y_labels = load_csv('test_data.csv', target_column=0,
                        categorical_labels=True, n_classes=2)
test_data = np.array(y_data,dtype=float)
acc = model.evaluate(test_data,y_labels,batch_size=128)
print("DNN准确率：",acc)