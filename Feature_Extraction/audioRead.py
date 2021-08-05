def readDirectory(Directory, nfiles, startIndex, listOfNoises, printDetails): 
    from SplitAudio import divAud
    import numpy as np
    from glob import glob
    #librosa is a library to access and manipulate audio data   
    import librosa as lib
    import DataAugmentation as DAug
    #setting data directory
    
    print("reading ",nfiles," files from ",Directory, file=printDetails);
    audio_files = glob(Directory+'/*.wav');
    y, fs = lib.load(audio_files[0], sr = None);
    Y = divAud(y,fs);
    Y = DAug.addNoise(Y, listOfNoises)
    windz = int(fs*0.025);   #window size
    shift = int(fs*0.010);
    mfcc = lib.feature.mfcc(y=Y[0,:], sr=fs, n_mfcc = 14, hop_length = shift, n_fft = windz); 
    mfcc = mfcc[1:14, :];
    t = np.arange(0, mfcc.shape[1]).astype(int);
    Xlang = np.append(mfcc, np.array([t]), axis = 0);  #add an mfcc column to denote the timestamps
    Xlang = np.append(Xlang, np.array([np.ones(mfcc.shape[1])+startIndex]).astype(int), axis = 0); #these denotes the ids
    Xlang = Xlang.T;
    for i in range(1, len(Y[:,])):
        mfcc = lib.feature.mfcc(y=Y[i,:], sr=fs, n_mfcc = 14, hop_length = shift, n_fft = windz)
        mfcc = mfcc[1:14, :];
        t = np.arange(0, mfcc.shape[1]).astype(int);
        mfcc = np.append(mfcc, np.array([t]) ,axis = 0);  #add an mfcc column 
        mfcc = np.append(mfcc, np.array([(1+i)*np.ones(mfcc.shape[1]) + startIndex]).astype(int),axis = 0).T;
        Xlang = np.append(Xlang, mfcc, axis=0);
    
    nRead = len(Y[:,]);
    #reading the files
    for i in range(1, min(len(audio_files), nfiles)):
        if(nRead>=nfiles):
            print('Number of separate files read: ',i,'\n','Number of training exmps formed: ',nRead,'\n', file=printDetails);
            break;
        y, fs = lib.load(audio_files[i], sr = None);
        windz = int(fs*0.025);   #window size
        shift = int(fs*0.010);
        Y = divAud(y,fs);
        Y = DAug.addNoise(Y, listOfNoises)
        for j in range(len(Y[:,])):#{
            mfcc = lib.feature.mfcc(y=Y[j,:], sr=fs, n_mfcc = 14, hop_length = shift, n_fft = windz)
            mfcc = mfcc[1:14, :];
            t = np.arange(0, mfcc.shape[1]).astype(int);
            mfcc = np.append(mfcc, np.array([t]) ,axis = 0);  #add an mfcc column 
            mfcc = np.append(mfcc, np.array([(1+nRead+j)*np.ones(mfcc.shape[1]) + startIndex]).astype(int),axis = 0).T;
            Xlang = np.append(Xlang, mfcc, axis=0);
        
        nRead = nRead + len(Y[:,]);
    return Xlang, nRead;
