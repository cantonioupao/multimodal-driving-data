# Deep Learning for Autonomous Driving
# Material for Problems 1-3 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import pickle

def load_data(data_path):
	''' 
    Load data dictionary from data_path.
    '''
	with open(data_path, 'rb') as fp:
	    data = pickle.load(fp)
	return data