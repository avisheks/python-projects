import os
import sys

import numpy as np
import math  as m
import random as rnd
import pylab as pl

import logging as LOG

from sklearn import linear_model as lm
from sklearn import gaussian_process as gpm
from sklearn import svm 
from sklearn import tree as dtr
from sklearn import preprocessing as pp
from sklearn import utils as ut

   
    
## THINGS TO DO:
## -- DONE: try out other regression models (gaussian processes)
##    -- let's ignore bayesian and gpm regression. too difficult to explain. 
##    -- let's go with simpler models
## -- DONE: try to break GV and LV features and see
## -- add cross validation to fine tune the parameters (for all models)
## -- DONE: repeat the learning over multiple subsets and then provide a final answer    
##    (maybe will be subsumed within the CV phase)    
    


######################################################################
def unique_rows(X,y):    
    order = np.lexsort(X.T)
    X = X[order]
    y = y[order]
    diff = np.diff(X, axis=0)
    ui = np.ones(len(X), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    X = X[ui]
    y = y[ui]



######################################################################
def compute_kernel_matrix(data_X_train):
    X = ut.array2d(data_X_train)
    n_samples, n_features = X.shape
    LOG.debug(n_samples)
    LOG.debug(n_features)
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
        LOG.debug('YES')
                                      


######################################################################
if __name__ == '__main__':

    ## params
    dataFile = sys.argv[1]
    finp = open(dataFile, 'r')

    ## create log file
    if 'network' in dataFile:
        logFName = dataFile.replace('../results/network/','')
    elif 'io' in dataFile:
        logFName = dataFile.replace('../results/io/','')
    elif 'combined' in dataFile:
        logFName = dataFile.replace('../results/combined/','')
    print('Dumping log messages in LOG.%s' % logFName)
    LOG.basicConfig(filename='LOG.'+logFName, level=LOG.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    LOG.debug('This is a log message.')

    ## sanity checks
    if not os.path.isfile(dataFile):
        LOG.debug('ERROR:',dataFile,'does not exist.')
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
        data[idx,2] = float(gv_vals[2])
        lv_vals = words[2].split('_')
        data[idx,3] = float(lv_vals[0])
        data[idx,4] = float(lv_vals[1])
        data[idx,5] = float(lv_vals[2])
        for i in range(4,len(words)):
            data[idx,2+i] = float(words[i])        
        idx = idx+1
    
    #LOG.debug(data)
      
    
    ## split datasets
    """
    X_raw = data[:,[0,1,2,3,4,5,6,8,9,10,11]]
    y = data[:, [14]]
    """
    X_raw = data[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
    y = data[:, [18]]
    """
    X_raw = data[:,[0,1,2,3,4,6,8,9,10,11,14]]
    y = data[:, [5]]
    """        
    #LOG.debug(X_raw)
    min_max_scaler = pp.MinMaxScaler()
    X = min_max_scaler.fit_transform(X_raw)
    #X = pp.scale(X_raw)
    #LOG.debug(X)
    #X = X_raw    
    
    unique_rows(X,y)
    compute_kernel_matrix(X)                                          
        
    ## randomly split datasets and average the models obtained
    MAXITER = 2
    dataRows, dataCols = np.shape(X)

    linreg = lm.LinearRegression()
    ridreg = lm.Ridge(alpha=0.9)
    lasreg = lm.Lasso(alpha=0.1)    
    larreg = lm.Lars()    
    netreg = lm.ElasticNet(alpha=0.1,rho=0.5)    
    dtrreg = dtr.DecisionTreeRegressor(max_depth=10)
    svmlin = svm.SVR(kernel='poly', C=1000000, degree=2)
    svmpol = svm.SVR(kernel='linear', C=1000000)
    svmrbf = svm.SVR(kernel='rbf', C=1000000, gamma=0.1)
    gpmreg = gpm.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)

    #for reg,reg_avg in [(linreg,linreg_avg), (ridreg,ridreg_avg), (lasreg,lasreg_avg), (larreg,larreg_avg), (netreg,netreg_avg), (dtrreg,dtrreg_avg), (svmreg,svmreg_avg), (gpmreg,gpmreg_avg)]:
    list_reg     = [linreg, ridreg, lasreg, larreg, netreg, dtrreg, svmlin, svmpol, svmrbf]
    #list_reg_avg = [linreg_avg, ridreg_avg, lasreg_avg, larreg_avg, netreg_avg, dtrreg_avg, svmreg_avg]
    list_reg_avg = np.zeros(shape=(len(list_reg),16))
    idx = 0
    for reg,reg_avg in zip(list_reg,list_reg_avg):
        reg_coef = np.zeros(shape=(MAXITER,16))
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
            
            compute_kernel_matrix(data_X_train)                                          
            reg.fit(data_X_train, data_y_train_list)
            if (reg != dtrreg) and (reg != svmlin) and(reg != svmpol) and (reg != svmrbf):
                reg_coef[iters] = reg.coef_
        
        print reg_coef
        list_reg_avg[idx] = np.copy(np.mean(reg_coef, axis=0))
        idx = idx + 1
        print('reg_avg:')
        print reg_avg
    
    print(list_reg_avg[0])
    print(list_reg_avg[1])
    print(list_reg_avg[2])
    print(list_reg_avg[3])
    print(list_reg_avg[4])
    print(list_reg_avg[5])
    print(list_reg_avg[6])
    
    #LOG.debug(linreg.coef_)
    #LOG.debug(ridreg.coef_)
    #LOG.debug(lasreg.coef_)
    #LOG.debug(larreg.coef_)
    #LOG.debug(netreg.coef_)
    #LOG.debug(svmreg.coef_)


    ### LOG.debug some results     
    ## LOG.debug the mean square error and the explained variance score: 1 is perfect prediction
    #LOG.debug('Linear: \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((linreg.predict(data_X_test) - data_y_test) ** 2), linreg.score(data_X_test, data_y_test)))
    #LOG.debug('Ridge : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((ridreg.predict(data_X_test) - data_y_test) ** 2), ridreg.score(data_X_test, data_y_test)))
    #LOG.debug('Lasso : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((lasreg.predict(data_X_test) - data_y_test) ** 2), lasreg.score(data_X_test, data_y_test)))
    #LOG.debug('Lars  : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((larreg.predict(data_X_test) - data_y_test) ** 2), larreg.score(data_X_test, data_y_test)))
    #LOG.debug('ENet  : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((netreg.predict(data_X_test) - data_y_test) ** 2), netreg.score(data_X_test, data_y_test)))
    #LOG.debug('DTR   : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((dtrreg.predict(data_X_test) - data_y_test) ** 2), dtrreg.score(data_X_test, data_y_test)))
    #LOG.debug('SVM   : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((svmreg.predict(data_X_test) - data_y_test) ** 2), svmreg.score(data_X_test, data_y_test)))
    ##LOG.debug('GPM   : \t Residual sum of squares: %.2f \t Var score: %.2f' % (np.mean((gpmreg.predict(data_X_test) - data_y_test) ** 2), gpmreg.score(data_X_test, data_y_test)))


    LOG.debug(linreg.predict(data_X_test))
    LOG.debug(ridreg.predict(data_X_test))
    LOG.debug(lasreg.predict(data_X_test))
    LOG.debug(larreg.predict(data_X_test))
    LOG.debug(netreg.predict(data_X_test))
    

    LOG.debug(dataRows)
    LOG.debug(np.shape(data_y_train.flatten().tolist()))
    
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
    
    
    
    ## ---------------------------------------------------------
    ## TESTING PHASE

    if len(sys.argv) > 2:

        LOG.debug('NOW TESTING................')
        testdataFile = sys.argv[2]
        finp = open(testdataFile, 'r')

        ## sanity checks
        if not os.path.isfile(testdataFile):
            LOG.debug('ERROR:',testdataFile,'does not exist.')
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
            print words
            gv_vals = words[0].split('_')
            testdata[idx,0] = float(gv_vals[0])
            testdata[idx,1] = float(gv_vals[1])
            testdata[idx,2] = float(gv_vals[2])
            lv_vals = words[2].split('_')
            testdata[idx,3] = float(lv_vals[0])
            testdata[idx,4] = float(lv_vals[1])
            testdata[idx,5] = float(lv_vals[2])
            for i in range(4,len(words)):
                testdata[idx,2+i] = float(words[i])        
            idx = idx+1
    
        """
        testdata_X_raw = testdata[:,[0,1,2,3,4,5,6,8,9,10,11]]
        testdata_y = testdata[:, [14]]
        testdata_y_std = testdata[:, [15]]
        """
        """
        testdata_X_raw = testdata[:,[0,1,2,3,4,6,8,9,10,11,14]]
        testdata_y = testdata[:, [5]]
        """
        testdata_X_raw = testdata[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
        testdata_y = testdata[:, [18]]
        testdata_y_std = testdata[:, [19]]
        #LOG.debug testdata_X_raw
        min_max_scaler = pp.MinMaxScaler()
        testdata_X = min_max_scaler.fit_transform(testdata_X_raw)
        #testdata_X = testdata_X_raw
    
    
        ## log results
        for reg in list_reg:
            LOG.debug(reg.predict(testdata_X))



        ###########################################################################################
        ## plot all the results

        ## set name for output plot
        resFile = testdataFile.replace('testdata/','figs/')
        if 'network' in dataFile:
            resFile = resFile + '__network'
        elif 'io' in dataFile:
            resFile = resFile + '__io'
        elif 'combined' in dataFile:
            resFile = resFile + '__combined'
        if 'vsD' in dataFile:
            resFile = resFile + '__vsD'
        elif 'vsN' in dataFile:
            resFile = resFile + '__vsN'
        elif 'ALL' in dataFile:
            resFile = resFile + '__ALL'


        ## choose colors
        colors = ('b','g','r','c','m','y','Orange','DarkBlue','DarkGreen','DarkRed','DarkCyan','DarkMagenta','LighCoral')
        labels = ('linear','ridge','lasso','lars','enet','dtr','svmlin','svmpol','svmrbf','gpm')


        ## plot all the results -- NON-AVERAGED results
        pl.figure()
        pl.plot(range(1,dataRows+1),testdata_y.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
        for reg,idx in zip(list_reg,range(0,len(colors))):
            pl.plot(range(1,dataRows+1),reg.predict(testdata_X), color=colors[idx], linestyle='-', marker='o',linewidth=2, label=labels[idx])
        pl.legend(loc='upper left')
        if "hopper" in testdataFile:
            pl.ylim([0,30000])
        elif "titan" in testdataFile:
            pl.ylim([0,15000])
        pl.savefig(resFile+'_plots.png', bbox_inches=0, rotation=90)
        pl.close()
    


        ## plot all the results -- AVERAGED results
        pl.figure()
        pl.plot(range(1,dataRows+1),testdata_y.flatten().tolist(), color='k', linestyle='-',marker='o',linewidth=3, label='org')
        for reg_avg,idx in zip(list_reg_avg,range(0,len(colors))):
            pl.plot(range(1,dataRows+1),np.dot(testdata_X,np.transpose(reg_avg)), color=colors[idx], linestyle='--', marker='o',linewidth=2, label=labels[idx])
        pl.legend(loc='upper left')
        if "hopper" in testdataFile:
            pl.ylim([0,30000])
        elif "titan" in testdataFile:
            pl.ylim([0,15000])
        pl.savefig(resFile+'_plots_avg.png', bbox_inches=0, rotation=90)
        pl.close()
        print resFile
