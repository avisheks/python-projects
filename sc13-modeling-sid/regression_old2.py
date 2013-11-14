import os
import sys
import numpy as np
import math  as m
import random as rnd
import pylab as pl

from sklearn import linear_model as lm
from sklearn import gaussian_process as gpm
from sklearn import svm 
from sklearn import tree as dtr
from sklearn import preprocessing as pp
from sklearn import utils as ut


## params
dataFile = sys.argv[1]
finp = open(dataFile, 'r')
## sanity checks
if not os.path.isfile(dataFile):
    print 'ERROR:',dataFile,'does not exist.'
    sys.exit(0)
## read data file directly from .csv to numpy array
tmp_data = np.genfromtxt(finp, delimiter='', skip_header=1)
#data = np.genfromtxt(finp, dtype="S17,f8,S17,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8", delimiter='', skip_header=1)
idx = 0    
finp.seek(0, 0)
finp.next()
dataRows, dataCols = np.shape(tmp_data)
data = np.zeros(shape=(dataRows,dataCols+2))
for line in finp:
    words = line.split()
    gv_vals = words[0].split('_')
    data[idx,0] = float(gv_vals[0])
    data[idx,1] = float(gv_vals[1])
    data[idx,2] = float(gv_vals[1])
    lv_vals = words[2].split('_')
    data[idx,3] = float(lv_vals[0])
    data[idx,4] = float(lv_vals[1])
    data[idx,5] = float(lv_vals[1])
    for i in range(4,len(words)):
        data[idx,2+i] = float(words[i])        
    idx = idx+1

#print data



## split datasets
X_raw = data[:,[0,1,2,3,4,5,6,8,9,10,11]]
y = data[:, [14]]
"""
X_raw = data[:,[0,1,2,3,4,6,8,9,10,11,14]]
y = data[:, [5]]
"""        
#print X_raw
min_max_scaler = pp.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)
#X = pp.scale(X_raw)
#print X    
#X = X_raw    

"""
def unique_rows(a):
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
#----------    
unique_X = np.unique(X.view([('', X.dtype)]*X.shape[1]))
X = unique_X.view(X.dtype).reshape((unique_X.shape[0], X.shape[1]))
#----------    
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]
#----------    
"""    
order = np.lexsort(X.T)
X = X[order]
y = y[order]
diff = np.diff(X, axis=0)
ui = np.ones(len(X), 'bool')
ui[1:] = (diff != 0).any(axis=1) 
X = X[ui]
y = y[ui]

    
## randomly split datasets and average the models obtained
MAXITER = 1
dataRows, dataCols = np.shape(X)
linregcoef = np.zeros(shape=(MAXITER,11))
ridregcoef = np.zeros(shape=(MAXITER,11))
lasregcoef = np.zeros(shape=(MAXITER,11))
larregcoef = np.zeros(shape=(MAXITER,11))
netregcoef = np.zeros(shape=(MAXITER,11))
dtrregcoef = np.zeros(shape=(MAXITER,11))
svmregcoef = np.zeros(shape=(MAXITER,11))
gpmregcoef = np.zeros(shape=(MAXITER,11))
for iters in range(0,MAXITER):        
    idx_all = range(0,dataRows)
    rnd.shuffle(idx_all)    
    idx_train = idx_all[0:int(m.ceil(0.95*dataRows))]
    idx_test = idx_all[int(m.ceil(0.95*dataRows)):-1]
    data_X_train = X[idx_train]
    data_X_test  = X[idx_test]
    data_y_train = y[idx_train]
    data_y_test  = y[idx_test]
    data_y_train_list = data_y_train.flatten().tolist()
    dataRowsFinal = int(0.95*dataRows)
    

    X = ut.array2d(data_X_train)
    n_samples, n_features = X.shape
    print n_samples
    print n_features
    n_nonzero_cross_dist = n_samples * (n_samples - 1) / 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])
    np.set_printoptions(threshold='nan')
    """
    for row in D:
        print '%s' % (' '.join('%10.6s' % i for i in row))
    print np.sum(D, axis=1)
    print np.shape(np.sum(D, axis=1))
    """
    if (np.min(np.sum(D, axis=1)) == 0.):
        print 'yES'
                                      
    
    ## Create different regression models and train the models using the training sets
    #---- linear regression 
    linreg = lm.LinearRegression()
    linreg.fit(data_X_train, data_y_train)
    linregcoef[iters] = linreg.coef_
    print 'linreg done....'
    #---- ridge regression    
    #ridreg = lm.RidgeCV(alphas=[0.01,0.1,1.0,10.0])
    ridreg = lm.Ridge(alpha=0.9)
    ridreg.fit(data_X_train, data_y_train)
    ridregcoef[iters] = ridreg.coef_
    print 'ridreg done....'
    #---- lasso
    lasreg = lm.Lasso(alpha=0.1)    
    #lasreg = lm.LassoCV()    
    lasreg.fit(data_X_train, data_y_train)
    lasregcoef[iters] = lasreg.coef_
    print 'lasreg done....'
    #---- lasso LARS
    larreg = lm.Lars()    
    #larreg = lm.LassoLars(alpha=0.1)    
    #larreg = lm.LarsCV()    
    larreg.fit(data_X_train, data_y_train)
    larregcoef[iters] = larreg.coef_
    print 'larreg done....'
    #---- elastic net
    netreg = lm.ElasticNet(alpha=0.1,rho=0.5)    
    #netreg = lm.ElasticNet()    
    #netreg = lm.ElasticNetCV()    
    netreg.fit(data_X_train, data_y_train)
    netregcoef[iters] = netreg.coef_
    print 'netreg done....'
    #---- decision tree regressor
    dtrreg = dtr.DecisionTreeRegressor(max_depth=10)
    dtrreg.fit(data_X_train, data_y_train)
    dtrregcoef[iters]
    print 'dtrreg done....'
    #---- gaussian process regression
    #svmreg = svm.SVR(kernel='rbf', C=1000000, gamma=0.1)
    svmreg = svm.SVR(kernel='poly', C=1000000, degree=2)
    #svmreg = svm.SVR(kernel='linear', C=1000000)
    #svmreg = svm.NuSVR(C=1,nu=0.1)
    svmreg.fit(data_X_train, data_y_train_list)
    #svmregcoef[iters] = svmreg.coef_
    print 'svmreg done....'
    """
    print np.shape(data_X_train)        
    print np.shape(data_y_train)
    """
    gpmreg = gpm.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)
    gpmreg.fit(data_X_train, data_y_train_list)
    print 'gpmreg done....'

"""    
print linreg.coef_
print ridreg.coef_
print lasreg.coef_
print larreg.coef_
print netreg.coef_
print svmreg.coef_
"""

avglinreg = np.mean(linregcoef, axis=0)
avgridreg = np.mean(ridregcoef, axis=0)
avglasreg = np.mean(lasregcoef, axis=0)
avglarreg = np.mean(larregcoef, axis=0)
avgnetreg = np.mean(netregcoef, axis=0)
avgdtrreg = np.mean(dtrregcoef, axis=0)
avgsvmreg = np.mean(svmregcoef, axis=0)

### print some results     
## Print the mean square error and the explained variance score: 1 is perfect prediction
#print('Linear: \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((linreg.predict(data_X_test) - data_y_test) ** 2), linreg.score(data_X_test, data_y_test)))
#print('Ridge : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((ridreg.predict(data_X_test) - data_y_test) ** 2), ridreg.score(data_X_test, data_y_test)))
#print('Lasso : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((lasreg.predict(data_X_test) - data_y_test) ** 2), lasreg.score(data_X_test, data_y_test)))
#print('Lars  : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((larreg.predict(data_X_test) - data_y_test) ** 2), larreg.score(data_X_test, data_y_test)))
#print('ENet  : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((netreg.predict(data_X_test) - data_y_test) ** 2), netreg.score(data_X_test, data_y_test)))
#print('DTR   : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((dtrreg.predict(data_X_test) - data_y_test) ** 2), dtrreg.score(data_X_test, data_y_test)))
#print('SVM   : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((svmreg.predict(data_X_test) - data_y_test) ** 2), svmreg.score(data_X_test, data_y_test)))
##print('GPM   : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((gpmreg.predict(data_X_test) - data_y_test) ** 2), gpmreg.score(data_X_test, data_y_test)))
"""
print linreg.predict(data_X_test)
print ridreg.predict(data_X_test)
print lasreg.predict(data_X_test)
print larreg.predict(data_X_test)
print netreg.predict(data_X_test)
"""

print dataRows
print np.shape(data_y_train.flatten().tolist())

### plot the results
##pl.errorbar(range(1,dataRows+1),testdata_y.flatten().tolist(), testdata_y_std.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
#pl.plot(range(1,dataRowsFinal+1),data_y_train.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
#pl.plot(range(1,dataRowsFinal+1),linreg.predict(data_X_train), color='r', linestyle='--', marker='o',linewidth=2, label='linear')
#pl.plot(range(1,dataRowsFinal+1),ridreg.predict(data_X_train), color='b', linestyle='--', marker='o',linewidth=2, label='ridge')
#pl.plot(range(1,dataRowsFinal+1),lasreg.predict(data_X_train), color='g', linestyle='--', marker='o',linewidth=2, label='lasso')
#pl.plot(range(1,dataRowsFinal+1),larreg.predict(data_X_train), color='c', linestyle='--', marker='o',linewidth=2, label='lars')
#pl.plot(range(1,dataRowsFinal+1),netreg.predict(data_X_train), color='m', linestyle='--', marker='o',linewidth=2, label='enet')
#pl.plot(range(1,dataRowsFinal+1),dtrreg.predict(data_X_train), color='m', linestyle='-', marker='o',linewidth=2, label='dtr')
#pl.plot(range(1,dataRowsFinal+1),svmreg.predict(data_X_train), color='c', linestyle='-', marker='o',linewidth=2, label='svm')
##pl.plot(range(1,dataRows+1),gpmreg.predict(data_X_train), color='g', linestyle='-', marker='o',linewidth=2, label='svm')
#pl.legend(loc='upper left')
##pl.ylim([0,300000])
##pl.ylim([0,75000])
#resFile = dataFile.replace('../results/network/','')
##resFile = dataFile.replace('../results/','')
#pl.savefig(resFile+'_plots_train.png', bbox_inches=0, rotation=90)



###########################################################
## TESTING PHASE
## params
if len(sys.argv) > 2:
    print 'NOW TESTING................'

    testdataFile = sys.argv[2]
    finp = open(testdataFile, 'r')
    ## sanity checks
    if not os.path.isfile(testdataFile):
        print 'ERROR:',testdataFile,'does not exist.'
        sys.exit(0)
    ## read data file directly from .csv to numpy array
    tmp_testdata = np.genfromtxt(finp, delimiter='', skip_header=1)
    #data = np.genfromtxt(finp, dtype="S17,f8,S17,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8", delimiter='', skip_header=1)
    idx = 0    
    finp.seek(0, 0)
    finp.next()
    dataRows, dataCols = np.shape(tmp_testdata)
    testdata = np.zeros(shape=(dataRows,dataCols+2))
    for line in finp:
        words = line.split()
        gv_vals = words[0].split('_')
        testdata[idx,0] = float(gv_vals[0])
        testdata[idx,1] = float(gv_vals[1])
        testdata[idx,2] = float(gv_vals[1])
        lv_vals = words[2].split('_')
        testdata[idx,3] = float(lv_vals[0])
        testdata[idx,4] = float(lv_vals[1])
        testdata[idx,5] = float(lv_vals[1])
        for i in range(4,len(words)):
            testdata[idx,2+i] = float(words[i])        
        idx = idx+1

    testdata_X_raw = testdata[:,[0,1,2,3,4,5,6,8,9,10,11]]
    testdata_y = testdata[:, [14]]
    """
    testdata_X_raw = testdata[:,[0,1,2,3,4,6,8,9,10,11,14]]
    testdata_y = testdata[:, [5]]
    """
    testdata_y_std = testdata[:, [15]]
    #print testdata_X_raw
    min_max_scaler = pp.MinMaxScaler()
    testdata_X = min_max_scaler.fit_transform(testdata_X_raw)
    #testdata_X = testdata_X_raw

    #print testdata_X    
    #print np.shape(data_X_test)        
    #print np.shape(testdata_X)

    print(linreg.predict(testdata_X))
    print(ridreg.predict(testdata_X))
    print(lasreg.predict(testdata_X))
    print(larreg.predict(testdata_X))
    print(netreg.predict(testdata_X))
    print(svmreg.predict(testdata_X))
    """
    print(gpmreg.predict(testdata_X))
    """

    #pl.errorbar(range(1,dataRows+1),testdata_y.flatten().tolist(), testdata_y_std.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
    pl.plot(range(1,dataRows+1),testdata_y.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
    pl.plot(range(1,dataRows+1),linreg.predict(testdata_X), color='r', linestyle='--', marker='o',linewidth=2, label='linear')
    pl.plot(range(1,dataRows+1),ridreg.predict(testdata_X), color='b', linestyle='--', marker='o',linewidth=2, label='ridge')
    pl.plot(range(1,dataRows+1),lasreg.predict(testdata_X), color='g', linestyle='--', marker='o',linewidth=2, label='lasso')
    #pl.plot(range(1,dataRows+1),larreg.predict(testdata_X), color='c', linestyle='--', marker='o',linewidth=2, label='lars')
    pl.plot(range(1,dataRows+1),netreg.predict(testdata_X), color='m', linestyle='--', marker='o',linewidth=2, label='enet')
    pl.plot(range(1,dataRows+1),dtrreg.predict(testdata_X), color='m', linestyle='-', marker='o',linewidth=2, label='dtr')
    pl.plot(range(1,dataRows+1),svmreg.predict(testdata_X), color='c', linestyle='-', marker='o',linewidth=2, label='svm')
    #pl.plot(range(1,dataRows+1),gpmreg.predict(testdata_X), color='g', linestyle='-', marker='o',linewidth=2, label='gpm')
    pl.legend(loc='upper left')
    #pl.ylim([0,300000])
    #pl.ylim([0,75000])
    resFile = dataFile.replace('../results/network/','')
    #resFile = dataFile.replace('../results/','')
    pl.savefig(resFile+'_plots_svm.png', bbox_inches=0, rotation=90)

    print 'Hi Hi Hi'
    print np.shape(range(1,dataRows+1))
    print np.shape(linreg.coef_)
    print np.shape(testdata_X)

    #pl.errorbar(range(1,dataRows+1),testdata_y.flatten().tolist(), testdata_y_std.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
    pl.errorbar(range(1,dataRows+1),testdata_y.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avglinreg)), color='r', linestyle='dotted', marker='o',linewidth=4, label='linear')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avgridreg)), color='b', linestyle='dotted', marker='o',linewidth=4, label='ridge')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avglasreg)), color='g', linestyle='dotted', marker='o',linewidth=4, label='lasso')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avglarreg)), color='c', linestyle='dotted', marker='o',linewidth=4, label='lars')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avgnetreg)), color='m', linestyle='dotted', marker='o',linewidth=4, label='enet')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avgdtrreg)), color='m', linestyle='-', marker='o',linewidth=4, label='dtr')
    pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(avgsvmreg)), color='c', linestyle='-', marker='o',linewidth=4, label='svm')
    pl.legend(loc='upper left')
    #pl.ylim([0,75000])
    #pl.ylim([0,300000])
    resFile = dataFile.replace('../results/network/','')
    #resFile = dataFile.replace('../results/','')
    pl.savefig(resFile+'_plots2_svm.png', bbox_inches=0, rotation=90)



## THINGS TO DO:
## -- DONE: try out other regression models (gaussian processes)
##    -- let's ignore bayesian and gpm regression. too difficult to explain. 
##    -- let's go with simpler models
## -- DONE: try to break GV and LV features and see
## -- add cross validation to fine tune the parameters
## -- repeat the learning over multiple subsets and then provide a final answer    
##    (maybe will be subsumed within the CV phase)    


