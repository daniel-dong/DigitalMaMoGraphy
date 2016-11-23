import pandas as pd
from sklearn import cross_validation,metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
from matplotlib import pyplot
#from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
url = "/home/elliotnam/project/mammography/patient_info.csv2"
dataframe = pd.read_csv(url)
dataframe.drop(dataframe.index[[0]])

#print(dataframe)
array = dataframe.values

#print(array[1,18:273])
#print(array[1,273])
#print(array)
X = array[:,19+20:274-100]
Y = array[:,274]

Y = list(Y)
#print(type(Y))
print(Y)
#Y = label_binarize(Y, classes=[0,1])
num_folds = 4
num_instances = len(X)
seed = 7

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=7)
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)

def runLogisticRegression():

    model = LogisticRegression()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runLinearDiscriment():
    model = LinearDiscriminantAnalysis()
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runKNNClassification():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds,
                                   random_state=seed)
    model = KNeighborsClassifier()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runNaiveBayes():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = GaussianNB()
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runCompareAlgorithms():
    models = []
    models.append(('LR', LogisticRegression()))
#    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DCN', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))

    num_trees = 250
    models.append(('XGB', XGBClassifier(n_estimators=num_trees)))

    models.append(('GBC', GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)))

    num_trees = 30
    models.append(('ABC', AdaBoostClassifier(n_estimators=num_trees, random_state=seed)))

    num_trees = 100
    max_features = 13
    models.append(('RFC', RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
        cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle("Algorithm Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def runXGBoost():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    # model = XGBClassifier(max_depth=3,n_estimators=250)
    model = XGBClassifier()
    print(model)
    model.fit(X_train, y_train)
    print("end fit")
    y_pred = model.predict(X_test)
    print("end predict")
    predictions = [round(value) for value in y_pred]
    print(predictions)
    model = XGBClassifier()
    n_estimators = [100, 200, 300, 400, 500]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    kfold = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=7)

    print("grdiserch cv")
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold)
    print("star grid search fit")
    result = grid_search.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))

    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    results = cross_validation.cross_val_score(model, X, Y.ravel(), cv=kfold)
    print(results.mean())
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    print(model.feature_importances_)

    n_estimators = range(50, 400, 50)
    param_grid = dict(n_estimators=n_estimators)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means, stdevs = [], []
    for params, mean_score, scores in result.grid_scores_:
        stdev = scores.std()
    means.append(mean_score)
    stdevs.append(stdev)
    print("%f (%f) with: %r" % (mean_score, stdev, params))

    max_depth = range(1, 11, 2)
    print(max_depth)
    param_grid = dict(max_depth=max_depth)
    kfold = StratifiedKFold(Y.ravel(), n_folds=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="log_loss", n_jobs=-1, cv=kfold,
                               verbose=1)
    result = grid_search.fit(X, Y.ravel())
    # summarize results
    print("Best: %f using %s" % (result.best_score_, result.best_params_))
    means, stdevs = [], []
    for params, mean_score, scores in result.grid_scores_:
        stdev = scores.std()
    means.append(mean_score)
    stdevs.append(stdev)
    print("%f (%f) with: %r" % (mean_score, stdev, params))


#    thresholds = sort(model.feature_importances_)
##    for thresh in thresholds:
# select features using threshold
#        selection = SelectFromModel(model, threshold=thresh, prefit=True)
#        select_X_train = selection.transform(X_train)
#        # train model
#        selection_model = XGBClassifier()
#        selection_model.fit(select_X_train, y_train.ravel())
#        # eval model
#        select_X_test = selection.transform(X_test)
#        y_pred = selection_model.predict(select_X_test)
#        predictions = [round(value) for value in y_pred]
#        accuracy = accuracy_score(y_test, predictions)
#        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],accuracy*100.0))

def runBaggedDecisionTree():
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runRandomForest():
    num_trees = 100
    max_features = 3
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runExtraTrees():
    num_trees = 100
    max_features = 12
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())


def runadaBust():
    num_trees = 30
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold)
    print(results.mean())
    print(results.mean())
    i = 0
    for train_index, test_index in kfold:
        if i == 2:
            model.fit(X[train_index], Y[train_index].ravel())
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:, 1]
            feat_imp = pd.Series(model.feature_importances_, colNames).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Ada Boost Feature Importance Score')
            plt.show()
        i += 1


def runGradientBust():
    num_trees = 100
    print('run gradeint')
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    model = GradientBoostingClassifier(learning_rate=0.005, n_estimators=num_trees, random_state=seed, max_depth=5,
                                       min_samples_split=1600, min_samples_leaf=50, subsample=0.8)
    results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
    #print(results)

    i = 0
    for train_index, test_index in kfold:
        print(train_index)
        if i == 2:
            model.fit(X[train_index], Y[train_index])
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:, 1]
            feat_imp = pd.Series(model.feature_importances).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Gradient Boost Feature Importance Score')
            plt.show()
        i += 1


#runLogisticRegression()
#runCompareAlgorithms()
#runRandomForest()
#runExtraTrees()
#runBaggedDecisionTree()
runGradientBust()
#runXGBoost()