For training and testing:
Use Train_Tester in the DDN_library.py file


##########use appropriate directory in the code################
relative to the position of the code, maintain the following directory structure
./CSVFiles/
		   AllClipsMFCC/        ---> contains the .csv file of MFCC features of the entire dataset 
		   Stats/              ---> store the mu, sigma for the features (both for raw MFCC and timeseries)
		   Data/			---> train/test sets
				LanguageWise/	---> all the subfiles after extraction 

#######################################################
In feature extraction:

Masterfile.py: pass the number of clips per language, number of test clips per and location of data and noise. Drives the process of feature extraction
ExtTserFeat.py: extracts the timeseries features from raw data and feature selection
CreateTrainMatrix.py: called by functions ExtTserFeat, to drive the reading of clips folder wise (per class)
audioRead.py: reads one directory and calls augmentation, on uniformly sized clips
SplitAudio.py: splits larger clips into uniform length clips of 5 s
DataAugmentaion.py: uses GetListOfNoises.py and GetNoise.py to select a random combination of noise and add it to the original clip (to increase the size of the data n time: n=11 here)
featureNormalize.py: normalizes the features assuming them to independent
SplitExampleSet.py: splits the dataset into train and test set after filtering unnecessary features 





