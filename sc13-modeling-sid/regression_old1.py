import os
import sys
import numpy as np
import math  as m
import random as rnd
import pylab as pl

from sklearn import linear_model as lm
from sklearn import preprocessing as pp


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
#print X    


    
## randomly split datasets and average the models obtained
MAXITER = 10    
linregcoef = np.zeros(shape=(MAXITER,11))
ridregcoef = np.zeros(shape=(MAXITER,11))
lasregcoef = np.zeros(shape=(MAXITER,11))
larregcoef = np.zeros(shape=(MAXITER,11))
netregcoef = np.zeros(shape=(MAXITER,11))
for iters in range(0,MAXITER):        
    idx_all = range(0,dataRows)
    rnd.shuffle(idx_all)    
    idx_train = idx_all[0:int(m.ceil(0.95*dataRows))]
    idx_test = idx_all[int(m.ceil(0.95*dataRows)):-1]
    data_X_train = X[idx_train]
    data_X_test  = X[idx_test]
    data_y_train = y[idx_train]
    data_y_test  = y[idx_test]
    
    
    ## Create different regression models and train the models using the training sets
    #---- linear regression 
    linreg = lm.LinearRegression()
    linreg.fit(data_X_train, data_y_train)
    linregcoef[iters] = linreg.coef_
    #---- ridge regression    
    #ridreg = lm.RidgeCV(alphas=[0.01,0.1,1.0,10.0])
    ridreg = lm.Ridge(alpha=0.9)
    ridreg.fit(data_X_train, data_y_train)
    ridregcoef[iters] = ridreg.coef_
    #---- lasso
    lasreg = lm.Lasso(alpha=0.1)    
    #lasreg = lm.LassoCV()    
    lasreg.fit(data_X_train, data_y_train)
    lasregcoef[iters] = lasreg.coef_
    #---- lasso LARS
    larreg = lm.Lars()    
    #larreg = lm.LassoLars(alpha=0.1)    
    #larreg = lm.LarsCV()    
    larreg.fit(data_X_train, data_y_train)
    larregcoef[iters] = larreg.coef_
    #---- elastic net
    netreg = lm.ElasticNet(alpha=0.1,rho=0.5)    
    #netreg = lm.ElasticNet()    
    #netreg = lm.ElasticNetCV()    
    netreg.fit(data_X_train, data_y_train)
    netregcoef[iters] = netreg.coef_

avglinreg = np.mean(linregcoef, axis=0)
avgridreg = np.mean(ridregcoef, axis=0)
avglasreg = np.mean(lasregcoef, axis=0)
avglarreg = np.mean(larregcoef, axis=0)
avgnetreg = np.mean(netregcoef, axis=0)

## print some results     
# Print the mean square error and the explained variance score: 1 is perfect prediction
print('Linear: \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((linreg.predict(data_X_test) - data_y_test) ** 2), linreg.score(data_X_test, data_y_test)))
print('Ridge : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((ridreg.predict(data_X_test) - data_y_test) ** 2), ridreg.score(data_X_test, data_y_test)))
print('Lasso : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((lasreg.predict(data_X_test) - data_y_test) ** 2), lasreg.score(data_X_test, data_y_test)))
print('Lars  : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((larreg.predict(data_X_test) - data_y_test) ** 2), larreg.score(data_X_test, data_y_test)))
print('ENet  : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((netreg.predict(data_X_test) - data_y_test) ** 2), netreg.score(data_X_test, data_y_test)))
"""
print linreg.predict(data_X_test)
print ridreg.predict(data_X_test)
print lasreg.predict(data_X_test)
print larreg.predict(data_X_test)
print netreg.predict(data_X_test)
"""


## plot the results
"""
pl.plot(data_y_test, color='k', linestyle='-',marker='o',linewidth=3, label='org')
pl.plot(linreg.predict(data_X_test), color='r', linestyle='--', marker='o',linewidth=2, label='linear')
pl.plot(ridreg.predict(data_X_test), color='b', linestyle='--', marker='o',linewidth=2, label='ridge')
pl.plot(lasreg.predict(data_X_test), color='g', linestyle='--', marker='o',linewidth=2, label='lasso')
pl.plot(larreg.predict(data_X_test), color='c', linestyle='--', marker='o',linewidth=2, label='lars')
pl.plot(netreg.predict(data_X_test), color='m', linestyle='--', marker='o',linewidth=2, label='enet')
pl.legend(loc='upper left')
resFile = dataFile.replace('../results/network/','')
pl.savefig(resFile+'_plots.png', bbox_inches=0, rotation=90)
"""



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
    #print testdata_X    

    #print np.shape(data_X_test)        
    #print np.shape(testdata_X)

    print(linreg.predict(testdata_X))
    print(ridreg.predict(testdata_X))
    print(lasreg.predict(testdata_X))
    print(larreg.predict(testdata_X))
    print(netreg.predict(testdata_X))

    #pl.errorbar(range(1,dataRows+1),testdata_y.flatten().tolist(), testdata_y_std.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
    pl.plot(range(1,dataRows+1),testdata_y.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
    pl.plot(range(1,dataRows+1),linreg.predict(testdata_X), color='r', linestyle='--', marker='o',linewidth=2, label='linear')
    pl.plot(range(1,dataRows+1),ridreg.predict(testdata_X), color='b', linestyle='--', marker='o',linewidth=2, label='ridge')
    pl.plot(range(1,dataRows+1),lasreg.predict(testdata_X), color='g', linestyle='--', marker='o',linewidth=2, label='lasso')
    pl.plot(range(1,dataRows+1),larreg.predict(testdata_X), color='c', linestyle='--', marker='o',linewidth=2, label='lars')
    pl.plot(range(1,dataRows+1),netreg.predict(testdata_X), color='m', linestyle='--', marker='o',linewidth=2, label='enet')
    pl.legend(loc='upper left')
    #pl.ylim([0,300000])
    #pl.ylim([0,75000])
    resFile = dataFile.replace('../results/network/','')
    #resFile = dataFile.replace('../results/','')
    pl.savefig(resFile+'_plots.png', bbox_inches=0, rotation=90)

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
    pl.legend(loc='upper left')
    #pl.ylim([0,75000])
    #pl.ylim([0,300000])
    resFile = dataFile.replace('../results/network/','')
    #resFile = dataFile.replace('../results/','')
    pl.savefig(resFile+'_plots2.png', bbox_inches=0, rotation=90)



## THINGS TO DO:
## -- DONE: try out other regression models (gaussian processes)
##    -- let's ignore bayesian and gpm regression. too difficult to explain. 
##    -- let's go with simpler models
## -- DONE: try to break GV and LV features and see
## -- add cross validation to fine tune the parameters
## -- repeat the learning over multiple subsets and then provide a final answer    
##    (maybe will be subsumed within the CV phase)    


