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
from xgboost import XGBClassifier
#使用pandas处理文件
data = pd.read_csv('D:\data.csv',header=0)
#print(data.head(3))
#打印数据信息
data.info()
# 去掉32列
data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself
#打印lie
print(data.columns)
# 去掉ID列
data.drop("id",axis=1,inplace=True)
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:21])
features_worst=list(data.columns[21:31])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)
#标签0：1
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
print(data.describe())
sns.countplot(data['diagnosis'], label="Count")
plt.show()
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
plt.show()
#选择特征值
prediction_var =['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
                 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se']
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
'''
#选择随机森林分类器
'''
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)#使用分类器训练数据
'''RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)'''
prediction = model.predict(test_X)#使用测试数据训练
print('随机森林算法（选取特征值为中间10个标准差的变量）准确率：')
print(metrics.accuracy_score(prediction,test_y))#检查准确率
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)
#print(featimp) # this is the property of Random Forest classifier that it provide us the importance
# of the features used
'''
#使用支持向量机
'''
model = svm.SVC()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('支持向量机的准确度：')
print(metrics.accuracy_score(prediction,test_y))
# Here Red color will be 1 which means M and blue foo 0 means B
color_function = {0:'blue',1:'red'}
# mapping the color fuction with diagnosis column
colors = data["diagnosis"].map(lambda x:color_function.get(x))
#plotting scatter plot matrix
#pd.scatter_matrix(data[features_se],c=colors,alpha=0.5,figsize=(20,20))
#plt.show()
'''
提升方法
'''
model = AdaBoostClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('提升算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
装袋算法
'''
model = BaggingClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('装袋算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
ExtraTreesClassifier分类算法
'''
model = ExtraTreesClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('ExtraTreesClassifier分类算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
GradientBoostingClassifier分类器
'''
model = GradientBoostingClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('GradientBoostingClassifier分类器算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
xgboost分类器
'''
model = XGBClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('xgboost分类器算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))

'''
最近邻算法
'''
model = KNeighborsClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('最近邻算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
Guassian贝叶斯
'''
model = GaussianNB()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('高斯贝叶斯算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
MultinomialNB贝叶斯
'''
model = MultinomialNB()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('MultinomialNB贝叶斯算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
 BernoulliNB贝叶斯
'''
model = BernoulliNB()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('BernoulliNB贝叶斯算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))
'''
决策树算法
'''
model = DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print('决策树算法的准确率：')
print(metrics.accuracy_score(prediction,test_y))



'''
K折检验
'''

def model(model,data,prediction,outcome):
# This function will be used for to check accuracy of different model
# model is the m
      kf = KFold(data.shape[0],n_folds=10)
prediction_var =['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                 'concave points_se','symmetry_se', 'fractal_dimension_se']

# As we are going to use many models lets make a function
# Which we can use with different models
def classfication_model(model,data,prediction_input,output):
    # here the model means the model
    # data is used for the data
    #prediction_input means the inputs used for prediction
    # output mean the value which are to be predicted
    # here we will try to find out the Accuarcy of model by using same data for fiiting and
    #comparison for same data
    #Fit the model:
    model.fit(data[prediction_input],data[output])
    predictions = model.predict(data[prediction_input])
    # Print accuracy
    # now checkin accuracy for same data
    accuracy = metrics.accuracy_score(predictions,data[output])
    print("Accuracy：%s"%"{0:.4%}".format(accuracy))
    kf = KFold(data.shape[0],n_folds=5)
    #https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
    #let me explain a little bit data.shape[0] means number of rows in data
    #n_folds is for number of folds
    error = []
    for train,test in kf:
        # as the data is divided into train and test using KFold
        # now as explained above we have fit many models
        # so here also we are going to fit model
        # in the cross validation the data in train and test will change for evry iteration
        # as the data is divided into train and test using KFold
        # now as explained above we have fit many models
        # so here also we are going to fit model
        # in the cross validation the data in train and test will change for evry iteration
        train_X = (data[prediction_input].iloc[train,:])# in this iloc is used for index of trainig data
        # here iloc[train,:] means all row in train in kf and the all columns
        train_y = data[output].iloc[train]# here is only column so it repersenting only row in train
        model.fit(train_X,train_y)
        # now do this for test data also
        test_X = data[prediction_input].iloc[test,:]
        test_y = data[output].iloc[test]
        error.append(model.score(test_X,test_y))
        # printing the score
        print("Cross-validation Score:%s"%"{0:.3%}".format(np.mean(error)))

# Now from Here start using different model
'''
决策树分类
 '''
model = DecisionTreeClassifier()
prediction_var = ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
                  'concave points_se', 'symmetry_se', 'fractal_dimension_se']
outcome_var= "diagnosis"
print('K-折决策树分类')
classfication_model(model,data,prediction_var,outcome_var)
#SVM
model = svm.SVC()
print('K-折SVM分类')
classfication_model(model,data,prediction_var,outcome_var)
#KNN
model = KNeighborsClassifier()
print('K-折KNN分类')
classfication_model(model,data,prediction_var,outcome_var)
#AdaBoost提升
model = AdaBoostClassifier()
print('K-折AdaBoost分类')
classfication_model(model,data,prediction_var,outcome_var)
#装袋bagging
model = BaggingClassifier()
print('K-折bagging分类')
classfication_model(model,data,prediction_var,outcome_var)
'''
使用grid search CV调参数
对决策树模型调参数
'''
data_X = data[prediction_var]
data_y = data['diagnosis']
#使用Grid Search CV
def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):
    clf = GridSearchCV(model, param_grid,cv=10,scoring="accuracy")
    # this is how we use grid serch CV we are giving our model
    # the we gave parameters those we want to tune
    # Cv is for cross validation
    # scoring means to score the classifier
    clf.fit(train_X,train_y)
    print("The best parameter found on development set is :")
    # this will gie us our best parameter to use
    print(clf.best_params_)
    print("the best estimator is ")
    print(clf.best_estimator_)
    print("the best score is")
    # this is the best score that we can achieve using these parameters#
    print(clf.best_score_)

    # Here we have to take parameters that are used for Decison tree Classifier
    # you will understand these terms once you follow the link above
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    # here our gridasearchCV will take all combinations of these parameter and apply it to model
'''
决策树的调参    
'''
model= DecisionTreeClassifier()
print('决策树调参结果：')
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
# call our function
'''
SVC调参数
'''
model=svm.SVC()
param_grid = [
              {'C': [1, 10, 100, 1000],
               'kernel': ['linear']
              },
              {'C': [1, 10, 100, 1000],
               'gamma': [0.001, 0.0001],
               'kernel': ['rbf']
              },
 ]
print('SVC调参结果：')
Classification_model_gridsearchCV(model,param_grid,data_X,data_y)



