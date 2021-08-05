import numpy as np
import pandas as pd
import os
def featNorm(X, start, end, mupath, sigmapath):
    MU = [];
    SIGMA = [];
    epsilon = 1e-7;
    for i in range(start, end):#{
        mew = np.mean(X[:,i]);
        MU.append(mew);
        sigma = np.std(X[:,i]);
        SIGMA.append(sigma);
        Z = (X[:,i] - mew)/(sigma+epsilon) ; 
        #print("mean: ",np.mean(Z),"std: ",np.std(Z),"\n");
        X[:,i] = Z;
    #}
    pd.DataFrame(MU).to_csv(mupath);
    pd.DataFrame(SIGMA).to_csv(sigmapath);
    return X;


def createDataX(directory, initXsave,mu_loc, sigma_loc, finalloc):

	#Ex2, mu
	mu = np.zeros([1, 9919], dtype=np.float64)	#number of features
	Ex2 = np.zeros([1, 9919], dtype=np.float64)

	X = pd.read_csv(directory+"/DataX_0.csv")
	row1 = X.iloc[len(X.index)-1]
	X = X.head(-1)
	mu, Ex2 = calc_Mu_Ex2(mu, Ex2, X, 177650)
	X.to_csv(initXsave, index=False)
	#mu
	del X
	for i in range(1,35):
		X = pd.read_csv(directory+"/DataX_"+str(i)+".csv")
		row2 = X.iloc[0]
		if row1['id'] == row2['id']:
			row2 = (row1 + row2)/2
			X.replace(to_replace=X.iloc[0], value=row2, inplace=True)
		else:
			row1 = row1.to_frame().transpose()
			X = pd.concat([row1, X])
		row1 = X.iloc[len(X.index)-1]
		if i<34:
			X = X.head(-1)
		mu, Ex2 = calc_Mu_Ex2(mu, Ex2, X, 177650)
		X.to_csv(initXsave, mode='a', header=False, index = False)
		#mu
		print("DataX_"+str(i)+".csv appended!")
		del X
	sigma = np.sqrt(Ex2 - mu*mu)
	np.save(mu_loc, mu)
	np.save(sigma_loc, sigma)
	print("mu, sigma written!", flush = True)
	#label_ranges = [1, 1000, 2000, 3000, 4000, 5000]
	#mu = np.load(mu_loc)
	#sigma = np.load(sigma_loc)
	ctr = 1
	label_ranges = []
	label_ranges.append(ctr);
	while True:
		ctr = ctr + 5000
		if ctr>=177651:
			label_ranges.append(177651)
			break;
		label_ranges.append(ctr)
	col_list = pd.read_csv(initXsave, skiprows=0, nrows=0)
	for i in range(1, len(label_ranges)):
		X = pd.read_csv(initXsave, skiprows=label_ranges[i-1], nrows=label_ranges[i]-label_ranges[i-1], header=None, names=col_list.columns)
		print("Completed reading for label "+str(0))
		X_norm = Normalize_df(X, mu, sigma, col_list)
		print("Completed normalizing for label, ",i)
		if not os.path.isfile(finalloc):
			X_norm.to_csv(finalloc, index=False)
		else:
			X_norm.to_csv(finalloc, mode='a', header=False, index=False)
		del X_norm
		del X
	#print("Completed feature extraction\n", file=printDetails, flush=True)
	print("Completed Appeding\n")
	#XtimeSer = extract_features(X, column_id='id', col	

def Normalize_df(X, mu, sigma, col_list):
	epsilon = 1E-9;
	X = X.to_numpy()
	X_temp = X[:, 1:]
	X_temp = (X_temp - mu)/(sigma+epsilon);
	ids = np.array([X[:, 0]]).T
	X = np.append(ids, X_temp, axis=1)
	X = pd.DataFrame(data=X, columns=col_list.columns)
	del X_temp
	return X

def calc_Mu_Ex2(mu, Ex2, X_batch, N):
	X_iter = X_batch.to_numpy()[:, 1:]
	mu = mu + np.sum(X_iter, axis=0, keepdims=True)/N
	Ex2 = Ex2 + np.sum(X_iter*X_iter, axis=0, keepdims=True)/N
	return mu, Ex2
