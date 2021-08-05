#functions for high computational capacity##
def Split(ndev, ntest, DataX, DataY, trainX_final, devX_final, testX_final, trainY_final, devY_final, testY_final, muTser, sigmaTser, printDetails):
    from featureNormalize import featNorm
    import pandas as pd
    import numpy as np
    import csv
    DataMX = pd.read_csv(DataX, index_col = None, na_values = ['NA']);
    DataMY = pd.read_csv(DataY, index_col = 0, na_values = ['NA']);  #tells that the first column is the serial number
    Data_numpy = pd.merge(DataMX, DataMY, on = 'id').to_numpy();  
    Data_numpy = featNorm(Data_numpy, 1, Data_numpy.shape[1] - 1, muTser, sigmaTser)[:,1:];
    ids = (np.array([np.arange(1, Data_numpy.shape[0] + 1)]).astype(int)).T;
    Data_numpy = np.random.permutation(Data_numpy);     #permutes the data
    FullRandomData = np.append(ids, Data_numpy, axis = 1);
    
    trange = FullRandomData.shape[0] - ndev - ntest;
    train = FullRandomData[0: trange, :];  
    dev = FullRandomData[trange : trange + ndev, :];
    test = FullRandomData[trange + ndev  : trange + ndev + ntest, :];
    lx = DataMX.columns.values;
    ly = DataMY.columns.values;
    SeparateX_Y_Write(train, lx, ly, trainX_final, trainY_final);
    SeparateX_Y_Write(dev, lx, ly, devX_final, devY_final);
    SeparateX_Y_Write(test, lx, ly, testX_final, testY_final);
    return True;

def SeparateX_Y_Write(Data, ylabelX, ylabelY, nameX, nameY):
    import pandas as pd
    import numpy as np
    import csv
    Y = np.append(np.array([Data[:, 0]]).T, np.array([Data[:, Data.shape[1]-1]]).T, axis = 1);
    X = Data[:, :-1];
    rowlab = np.arange(0, X.shape[0]).astype(int);
    X = pd.DataFrame(data = X, columns = ylabelX, index = rowlab); 
    Y = pd.DataFrame(data = Y, columns = ylabelY, index = rowlab);
    X.to_csv(nameX);
    Y.to_csv(nameY);
###############################################################################################
#low computational capacity
def Split_labelwise(ntest, DataX, DataY, trainX, testX, trainY, testY):
	import pandas as pd
	import numpy as np
	import os
    #inspect for the number of examples in each language (extra 1 as rows start from 1)
	label_ranges = [1, 30019, 30019, 30008, 29876, 30030, 27698]
    #for all the classes, igonoring label_ranges[0]
    
	for i in range(1,len(label_ranges)):
		label_ranges[i] = label_ranges[i] + label_ranges[i-1]
	col_listX = pd.read_csv(DataX, nrows=0)
	col_listY = pd.read_csv(DataY, nrows=0, index_col=0)
	
	Xtest = pd.read_csv(DataX, nrows=0)
	
	Ytest = pd.read_csv(DataY, nrows=0, index_col=0)
	Ytrain = pd.read_csv(DataY, nrows=0, index_col=0)
	for i in range(1, len(label_ranges)):
		X = pd.read_csv(DataX, skiprows=label_ranges[i-1], nrows=label_ranges[i]-label_ranges[i-1], header=None, names=col_listX.columns)
		Y = pd.read_csv(DataY, index_col=0, skiprows=label_ranges[i-1], nrows=label_ranges[i]-label_ranges[i-1], header=None, names=col_listY.columns)
		XY = pd.merge(X, Y, on='id')
		del X
		del Y
		XY = XY.sample(frac=1).reset_index(drop=True)
		Ytest = pd.concat([Ytest, XY.iloc[0:ntest][['id', 'label']]])
		Ytrain = pd.concat([Ytrain, XY.iloc[ntest:][['id', 'label']]])
		X = XY.drop(['label'], axis=1)
		del XY
		Xtest = pd.concat([Xtest, X.iloc[0:ntest]])
		Xtrain = X.drop(np.arange(ntest), axis=0)
		del X
		
		if not os.path.isfile(trainX):
			Xtrain.to_csv(trainX, index=False)
		else:
			Xtrain.to_csv(trainX, mode='a', header=False, index=False)
		del Xtrain
		print("Completed for label ", i)

	Xtest.to_csv(testX, index=False)

	Ytest.to_csv(testY, index=False)
	Ytrain.to_csv(trainY, index=False)