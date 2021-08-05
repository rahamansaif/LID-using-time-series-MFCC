
def extractClipWiseFeatures(loc, listOfNoises, clipsPerLang, initXsave, initYsave, muMFCC, sigmaMFCC, action, printDetails):    
    from CreateTrainMatrix import createTrainData
    from tsfresh import extract_features
    import pandas as pd
    import numpy as np
    import csv
    Data, Y = createTrainData(loc, clipsPerLang, muMFCC, sigmaMFCC, listOfNoises, printDetails);
    colLabX = np.arange(0, Data.shape[0]).astype(int);
    colLabY = np.arange(0, Y.shape[0]).astype(int);
    xlab = [];
    for i in range(13):
        xlab.append('M_'+str(i+1));
    xlab.append('time');
    xlab.append('id');
    ylab = ['id','label'];
    #extracted_features = extract_features();
    X = pd.DataFrame(data = Data, columns = xlab, index = colLabX);
    Y = pd.DataFrame(data = Y, columns = ylab, index = colLabY);
    X.to_csv("MFCC_"+action+".csv");
    #writing to csv file  
    Y.to_csv(initYsave);
    #writing to csv file  
    
    #uncomment if highly memory intensive computation is possible/data is small
    #XtimeSer = extract_features(X, column_id='id', column_sort = 'time');
    #XtimeSer.to_csv(initXsave);
    del X
    #del XtimeSer
    return True;

def extract_fromMFCC(mfcc_loc, printDetails):   
    from tsfresh import extract_features
    import pandas as pd
    import csv
    
    
    label_ranges = []
    label_ranges.append(1)
    ctr = 
    incr = 2848840          #replace with maximum capacity of mfcc data that can allow the feature extraction 
    maxlab = 99709424       #the the number of rows in the mfcc file
    while True:
        ctr = ctr + incr
        if ctr >= (int((maxlab)/(incr)))*incr:
            label_ranges.append(maxlab)
            break;
        label_ranges.append(ctr)
    
    col_list = pd.read_csv(mfcc_loc, skiprows=0, nrows=0, index_col=0)
    for i in range(1, len(label_ranges)):
        X = pd.read_csv(mfcc_loc, skiprows=label_ranges[i-1], nrows=label_ranges[i]-label_ranges[i-1], index_col=0, header=None, names=col_list.columns)
        print("Completed reading MFCCs for label "+str(i))
        XtimeSer = extract_features(X, column_id='id', column_sort='time')
        print("Completed feature extraction for label "+str(i))
        XtimeSer.to_csv("CSVFiles/Data/LanguageWise/DataX_"+str(i)+".csv")

    print("Completed feature extraction\n", file=printDetails, flush=True)
    print("Completed feature extraction\n")
    return True;

def SelectSubSet_FeatSelection(subsetX, subsetY, trainX_beforefinal, trainY_final, testX_beforefinal, trainX_final_INT, testX_final_INT, printDetails):
    import pandas as pd
    import os
    nreq = 7500     #total number of examples per class for feature selection
    
    select_from = pd.read_csv(trainY_final)
    info = select_from.groupby('label').size()
    skips = [1]
    for i in range(1, len(info)):
       skips.append(skips[i-1]+info[i])
    
    col_listX = pd.read_csv(trainX_beforefinal, skiprows=0, nrows=0)
    col_listY = pd.read_csv(trainY_final, skiprows=0, nrows=0)
    for i in range(len(skips)):
        X = pd.read_csv(trainX_beforefinal, skiprows=skips[i], nrows=nreq, header=None, names=col_listX.columns)
        Y = pd.read_csv(trainY_final, skiprows=skips[i], nrows=nreq, header=None, names=col_listY.columns)
        print("Completed selecting for label "+str(i))
        if not os.path.isfile(subsetX):
            X.to_csv(subsetX, index=False)
            Y.to_csv(subsetY, index=False)
        else:
            X.to_csv(subsetX, mode='a', header=False, index=False)
            Y.to_csv(subsetY, mode='a', header=False, index=False)
        del X
        del Y
    print("Completed feature extraction\n", file=printDetails, flush=True)
    print("Completed selecting random subset for feature selection\n")
    
    basicFeatureSelection(subsetX, subsetY, trainX_beforefinal, trainY_final, testX_beforefinal, trainX_final_INT, testX_final_INT, printDetails)

def basicFeatureSelection(subsetX, subsetY, trainX_beforefinal, trainY_final, testX_beforefinal, trainX_final_INT, testX_final_INT, printDetails):
    from tsfresh import select_features
    from tsfresh.utilities.dataframe_functions import impute
    import pandas as pd
    import numpy as np
    import csv
    filteredFeatures_union = set();
    filteredFeatures_INT = set();
    print(len(list(filteredFeatures_INT)))
    trainMatrix = pd.read_csv(subsetX, index_col = 0, na_values = ['NA']);
    Y = pd.read_csv(subsetY, index_col = 0, na_values = ['NA']);
    
    impute(trainMatrix);        #removes as NAN values obtained in feature extraction
    
    for label in Y['label'].unique(): 
        bin_label = (Y['label']==label);
        #print("label = ",label,"array created is: \n",bin_label,"\n");
        #select features for all ones-vs-all method
        filteredMatrix = select_features(trainMatrix, bin_label);
        print("Number of relevant features for class {}: {}/{}".format(label, filteredMatrix.shape[1], trainMatrix.shape[1]), flush=True);
        print("Number of relevant features for class {}: {}/{}".format(label, filteredMatrix.shape[1], trainMatrix.shape[1]), file=printDetails, flush=True);  
        #union of all relavent features of all classes
        if len(list(filteredFeatures_INT)) == 0:
            filteredFeatures_INT = filteredFeatures_INT.union(set(filteredMatrix.columns));
        else:    
            filteredFeatures_INT = filteredFeatures_INT.intersection(set(filteredMatrix.columns));  #getting intersection of the labels
        del filteredMatrix
    
    print("\nTotal number of selected features by intersection: ",len(filteredFeatures_INT), file=printDetails, flush=True);
    
    del trainMatrix
    del Y
    #saving selected features
    INTfeat = open("CSVFiles/SelectedFeatures/INTfeat.txt", 'w');
    print(list(filteredFeatures_INT), file=INTfeat, flush=True);
    INTfeat.close();

    filter_Train(trainX_final_INT, trainX_beforefinal, filteredFeatures_INT, printDetails)
    print("Completed filtering the trainset", flush=True)
   
    testMatrix = pd.read_csv(testX_beforefinal, index_col = 0, na_values = ['NA']);
    testMatrixFiltered_INT = testMatrix[list(filteredFeatures_INT)];
    impute(testMatrixFiltered_INT) 
    print("\n After Extraction (INT): number of features=",testMatrixFiltered_INT.shape[1], file=printDetails, flush=True)
    testMatrixFiltered_INT.to_csv(testX_final_INT)
    del testMatrixFiltered_INT
    del testMatrix   
    #writing test matrices
   
    return True;

def filter_Train(trainX_final_INT, trainX_beforefinal, filteredFeatures_INT, printDetails):
    import pandas as pd
    from tsfresh.utilities.dataframe_functions import impute
    import os
    print("For your Train Set", flush=True, file=printDetails)
    label_ranges = []
    label_ranges.append(1)
    ctr = 
    incr = 10000          #replace with maximum capacity of mfcc data that can allow the feature extraction 
    maxlab = 171651       #the the number of rows in the mfcc file
    while True:
        ctr = ctr + incr
        if ctr >= (int((maxlab)/(incr)))*incr:
            label_ranges.append(maxlab)
            break;
        label_ranges.append(ctr)
    

    col_list = pd.read_csv(trainX_beforefinal, index_col=0, skiprows=0, nrows=0)
    for i in range(1, len(label_ranges)):
        X = pd.read_csv(trainX_beforefinal, skiprows=label_ranges[i-1], nrows=label_ranges[i]-label_ranges[i-1], header=None, names=col_list.columns, index_col=0, na_values=['NA'])
        X_int = X[list(filteredFeatures_INT)]
        impute(X_int)
        print("Shape of X = ", X.shape[1], "of intersection: ", X_int.shape[1], "\n", flush=True, file=printDetails)
        print("Shape of X = ", X.shape[1], "of intersection: ", X_int.shape[1], flush=True)
        if not os.path.isfile(trainX_final_INT):
            X_int.to_csv(trainX_final_INT)
        else:
            X_int.to_csv(trainX_final_INT, mode='a', header=False)
        del X_int
        del X
        print("Completed for batch, ",i, flush=True)