def createTrainData(loc, n, mu, sigma, listOfNoises, printDetails):
    from audioRead import readDirectory
    from featureNormalize import featNorm
    import numpy as np
    import os
    import csv
    folders = os.listdir(loc);
    TrainMatrix, nRead = readDirectory(loc+'/'+folders[0], n, 0, listOfNoises, printDetails)
    ids = np.linspace(1, nRead, nRead).astype(int);
    label = np.array(np.ones(nRead)).astype(int);
    Y = np.column_stack((ids, label));
    READ = nRead;
    for i in range(1,len(folders)):
        langi, nRead = readDirectory(loc+'/'+folders[i], n, READ, listOfNoises, printDetails);
        TrainMatrix = np.append(TrainMatrix,langi,axis=0);
        ids = np.linspace(READ+1, READ+nRead, nRead).astype(int);
        label = np.array(np.ones(nRead)*(i+1)).astype(int);
        toappend = np.column_stack((ids, label));
        Y = np.append(Y, toappend, axis = 0);
        READ = READ + nRead;
   
    TrainMatrix = featNorm(TrainMatrix,0,13,mu,sigma);

    return TrainMatrix, Y;
    