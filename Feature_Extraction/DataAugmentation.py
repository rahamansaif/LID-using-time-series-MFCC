import numpy as np
import librosa as lib
import math
import random
import GetNoise as GN

#must install ffmpeg

def addNoise(audioClips, listOfNoises):
	nClips = len(audioClips)
	for itr in range(10):
		for i in range(nClips):
			data = audioClips[i,:]
			data = np.array([data])
			Edata = getAvgEnergy(data)
			n = itr % 5
			if n==0:
				noiseClips = GN.getHomeNoise(listOfNoises)
			elif n==1:
				noiseClips = GN.getNatureNoise(listOfNoises)
			elif n==2:
				noiseClips = GN.getStreetNoise(listOfNoises)
			elif n==3:
				noiseClips = GN.getHomeAndNatureNoise(listOfNoises)
			else:
				noiseClips = GN.getNatureAndStreetNoise(listOfNoises)
			nNoise = len(noiseClips)
			newData = data
			for j in range(nNoise):
				noise, fs = lib.load(noiseClips[j], sr=None)
				noise = np.array([noise])
				if (len(noise[0]) > len(data[0])):
					startIndex = random.randint(0,len(noise[0])-len(data[0]))
					noise = noise[:,startIndex:startIndex+len(data[0])]
				Enoise = getAvgEnergy(noise)
				scaleFactor = (random.random()*0.1+0.02) * math.sqrt(Edata/Enoise)
				if (len(noise[0]) < len(data[0])):
					paddingSize = len(data[0]) - len(noise[0])
					frontPaddingSize = random.randint(0,paddingSize)
					noise = np.append(np.zeros((1,frontPaddingSize)), noise, axis=1)
					noise = np.append(noise, np.zeros((1, paddingSize-frontPaddingSize)), axis=1)
				newData = newData + scaleFactor*noise
			audioClips = np.append(audioClips, newData, axis=0)
	return audioClips
    
    
def getAvgEnergy(vector):
	return np.dot(vector,vector.T)[0,0]/vector.shape[1]
    