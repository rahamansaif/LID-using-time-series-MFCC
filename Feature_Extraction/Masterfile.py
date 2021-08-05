#ntest is per language
#nclips is per language
def Caller(dataloc, noiseloc, nClips, ntest):
    import ExtTserFeat as extract
    import pandas as pd
    import csv
    import featureNormalize as fn
    import SplitExampleSet as se
    from GetListOfNoises import getListOfNoises
    #nclips denote the number of train data for each language
    
    printDetails = open('SelectionDetails.txt', 'w');
    
    DataX = "CSVFiles/Data/DataX.csv";  
    DataY = "CSVFiles/Data/DataY.csv";
    
    
    trainX_beforefinal = "CSVFiles/Data/TrainSetInitX.csv";
    testX_beforefinal = "CSVFiles/Data/TestSetInitX.csv";
    
    trainX_final_INT = "CSVFiles/Data/TrainSetX_INT.csv";
    testX_final_INT = "CSVFiles/Data/TestSetX_INT.csv";
    
    trainY_final = "CSVFiles/Data/TrainSetY.csv";
    testY_final = "CSVFiles/Data/TestSetY.csv";
    
    listOfNoises = getListOfNoises(noiseloc)
    #only a subset of training examples are to be used for feature selection
    subsetX = "CSVFiles/Data/subsetX.csv"
    subsetY = "CSVFiles/Data/subsetY.csv"
    
    if(extract.extractClipWiseFeatures(dataloc, listOfNoises, nClips, DataX, DataY, "CSVFiles/Stats/muMFCC.csv", "CSVFiles/Stats/sigmaMFCC.csv", "createData", printDetails)):
        #if your computer can handle highly memory intensive computation the above function will suffice, just uncomment the last few lines an comment the if statement below
        if(extract.extract_fromMFCC("CSVFiles/AllClipsMFCC/MFCC_createData.csv",  printDetails)):
            #returns the chunk of data to be normalized and split
            fn.createDataX("CSVFiles/Data/LanguageWise", "CSVFiles/Data/DataX_not_Norm.csv", "CSVFiles/Stats/Mu_Tser.npy", "CSVFiles/Stats/Sigma_Tser.npy", DataX)
            newX = pd.read_csv(DataX);
            newX = newX.groupby('id', as_index=False).mean()
            newX.to_csv(DataX);
            del newX
            print("Extraction Done", file=printDetails, flush=True);
            #uncomment if dataset is small/high computation
            #if(se.Split(ndev, ntest, DataX, DataY, trainX_beforefinal, devX_beforefinal, testX_beforefinal, trainY_final, devY_final, testY_final, "CSVFiles/Stats/muTser.csv", "CSVFiles/Stats/sigmaTser.csv", printDetails)):
            #    se.basicFeatureSelection(trainX_beforefinal, trainY_final, devX_beforefinal, testX_beforefinal, trainX_final_union, devX_final_union, testX_final_union, trainX_final_INT, devX_final_INT, testX_final_INT, printDetails);
    
    se.Split_labelwise(ntest, DataX, DataY, trainX_beforefinal, testX_beforefinal, trainY_final, testY_final)
    extract.SelectSubSet_FeatSelection(subsetX, subsetY, trainX_beforefinal, trainY_final, testX_beforefinal, trainX_final_INT, testX_final_INT, printDetails)
    printDetails.close()