import os
import sys
import numpy as np
import math  as m
import random as rnd
import pylab as pl

from sklearn import ensemble as ens
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



## randomly split datasets and average the models obtained
#for iters in range(0,10):        
## split datasets
X_raw = data[:,[0,1,2,3,4,5,6,8,9,10,11]]
y = data[:, [14]]
#print X_raw
min_max_scaler = pp.MinMaxScaler()
X = min_max_scaler.fit_transform(X_raw)
#print X    

idx_all = range(0,dataRows)
rnd.shuffle(idx_all)    
idx_train = idx_all[0:int(m.ceil(0.8*dataRows))]
idx_test = idx_all[int(m.ceil(0.8*dataRows)):-1]
data_X_train = X[idx_train]
data_X_test  = X[idx_test]
data_y_train = y[idx_train]
data_y_test  = y[idx_test]
"""
print dataRows    
print dataCols    
print idx_all
print idx_train
print idx_test               
print np.shape(data)
print data[0]    
print data[1]
print data
"""


## Create different regression models and train the models using the training sets
# random forest
forreg = ens.RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=4, min_samples_split=1)
forreg.fit(data_X_train, data_y_train)
# extra trees
treereg = ens.ExtraTreesRegressor(n_estimators=500, criterion='mse', max_depth=4, min_samples_split=1)
treereg.fit(data_X_train, data_y_train)
# adaboost
#adareg = ens.AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=500)  
#adareg.fit(data_X_train, data_y_train)
# gradient boosted  
gbdtreg = ens.GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=500, min_samples_split=1, max_depth=4)
gbdtreg.fit(data_X_train, data_y_train)


## print some results     
# The coefficients
#print('Coefficients: \n', regr.coef_)
# Print the mean square error and the explained variance score: 1 is perfect prediction
print("Random Forest Regression:")    
print("Residual sum of squares: %.2f" % np.mean((forreg.predict(data_X_test) - data_y_test) ** 2))
print('Variance score: %.2f' % forreg.score(data_X_test, data_y_test))
print("Extra Trees Regression:")    
print("Residual sum of squares: %.2f" % np.mean((treereg.predict(data_X_test) - data_y_test) ** 2))
print('Variance score: %.2f' % treereg.score(data_X_test, data_y_test))
#print("Adaboost Regression:")    
#print("Residual sum of squares: %.2f" % np.mean((adareg.predict(data_X_test) - data_y_test) ** 2))
#print('Variance score: %.2f' % adareg.score(data_X_test, data_y_test))
print("GBDT Regression:")    
print("Residual sum of squares: %.2f" % np.mean((gbdtreg.predict(data_X_test) - data_y_test) ** 2))
print('Variance score: %.2f' % gbdtreg.score(data_X_test, data_y_test))
"""
print linreg.predict(data_X_test)
print ridreg.predict(data_X_test)
print lasreg.predict(data_X_test)
print larreg.predict(data_X_test)
print netreg.predict(data_X_test)
"""


## plot the results
pl.plot(data_y_test, color='k', linestyle='-',marker='o',linewidth=3, label='org')
pl.plot(forreg.predict(data_X_test), color='r', linestyle='--', marker='o',linewidth=2, label='linear')
pl.plot(treereg.predict(data_X_test), color='b', linestyle='--', marker='o',linewidth=2, label='ridge')
#pl.plot(adareg.predict(data_X_test), color='g', linestyle='--', marker='o',linewidth=2, label='lasso')
pl.plot(gbdtreg.predict(data_X_test), color='c', linestyle='--', marker='o',linewidth=2, label='lars')
pl.legend(loc='upper left')
resFile = dataFile.replace('../results/network/','')
pl.savefig(resFile+'_plots.png', bbox_inches=0, rotation=90)


## THINGS TO DO:
## -- try out other regression models (gaussian processes)
##    -- let's ignore bayesian and gpm regression. too difficult to explain. 
##    -- let's go with simpler models
## -- try to break GV and LV features and see
## -- add cross validation to fine tune the parameters







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
    #print testdata_X_raw
    min_max_scaler = pp.MinMaxScaler()
    testdata_X = min_max_scaler.fit_transform(testdata_X_raw)
    #print testdata_X    

    #print np.shape(data_X_test)        
    #print np.shape(testdata_X)

    print(forreg.predict(testdata_X))
    print(treereg.predict(testdata_X))
    #print(adareg.predict(testdata_X))
    print(gbdtreg.predict(testdata_X))
