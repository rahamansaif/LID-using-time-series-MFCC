import numpy as np
#from numpy import random

def getHomeNoise(listOfNoises):
	while True:
		choices = np.random.randint(0,2,3)
		if (np.array_equal(choices, np.zeros(3)) == False):
			break
	noiseClips = list()
	for i in range(3):
		if (choices[i]==1):
			clipNo = np.random.randint(0, len(listOfNoises[i]))
			noiseClips.append(listOfNoises[i][clipNo])
	return noiseClips


def getNatureNoise(listOfNoises):
	while True:
		choices = np.random.randint(0,2,2)
		if (np.array_equal(choices, np.zeros(2)) == False):
			break
	noiseClips = list()
	for i in range(2):
		if (choices[i]==1):
			clipNo = np.random.randint(0, len(listOfNoises[i+3]))
			noiseClips.append(listOfNoises[i+3][clipNo])
	return noiseClips


def getStreetNoise(listOfNoises):
	noiseClips = list()
	clipNo = np.random.randint(0, len(listOfNoises[5]))
	noiseClips.append(listOfNoises[5][clipNo])
	return noiseClips


def getHomeAndNatureNoise(listOfNoises):
	noiseClips = list()
	homeChoice = np.random.randint(0,3)
	clipNo = np.random.randint(0, len(listOfNoises[homeChoice]))
	noiseClips.append(listOfNoises[homeChoice][clipNo])
	natureChoice = np.random.randint(3,5)
	clipNo = np.random.randint(0, len(listOfNoises[natureChoice]))
	noiseClips.append(listOfNoises[natureChoice][clipNo])
	return noiseClips


def getNatureAndStreetNoise(listOfNoises):
	noiseClips = list()
	natureChoice = np.random.randint(3,5)
	clipNo = np.random.randint(0, len(listOfNoises[natureChoice]))
	noiseClips.append(listOfNoises[natureChoice][clipNo])
	clipNo = np.random.randint(0, len(listOfNoises[5]))
	noiseClips.append(listOfNoises[5][clipNo])
	return noiseClips