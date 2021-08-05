def getListOfNoises(directory):
	from glob import glob
	ac_fan = glob(directory+"/Home/AC_Fan/*")
	door_doorbell = glob(directory + "/Home/Door_Doorbell/*")
	home_others = glob(directory + "/Home/Others/*")
	birds_animals = glob(directory + "/Nature/Birds_Animals/*")
	rain_wind = glob(directory + "/Nature/Rain_Wind/*")
	streets = glob(directory + "/Streets/*")
	return [ac_fan, door_doorbell, home_others, birds_animals, rain_wind, streets]