from load_data import load_data
import numpy as np
import os

print("So let's start the task")
data_path = os.path.join("./data", "data.p")
data = load_data(data_path)
velodyne =  data['velodyne']
objects = data['objects']
print("The first 3 dimensions represent x,y,z and the last one represents the reflectance intensity  between 0 - 1")
print("Shape is "+ str(velodyne.shape))
print("We know that the maximum range is 120 m" )
num_of_points = velodyne.shape[0]
max_velodyne = np.amax(velodyne, 0)
max_x = max_velodyne [0]
max_y = max_velodyne [1]
max_z = max_velodyne [2]
min_velodyne = np.amin(velodyne, 0)
min_x = min_velodyne [0]
min_y = min_velodyne [1]
min_z = min_velodyne [2]
print("The maximum and minimum values for x, y  and z are ", max_x , max_y, max_z, "and respectively ", min_x, min_y,min_z)
z_per_bin = []
value_per_bin = []
total_x = max_x - min_x
total_y = max_y -min_y
resolution_x = 0.2 
resolution_y = 0.2
pixels_x= int(total_x/resolution_x)
pixels_y = int(total_y / resolution_y)
tot_bins = pixels_x * pixels_y 
pixels = [[0 for x in range(pixels_x)] for y in range(0, pixels_y)] 

print("Pixels in x direction is ", pixels_x)
print("Pixels in y direction is ", pixels_y)
print("Total number of bins (pixels) is ", tot_bins)

for x_bin in range(0,pixels_x):

    for y_bin in range(0,pixels_y):

        for i in range (0,num_of_points ): #the ith example
            exists = 0
            x = velodyne[i][0]
            y = velodyne[i][1]
            z = velodyne[i][2]
            x1 = max_x - x_bin*0.2
            x2 = max_x - (x_bin+1)*0.2
            y1 = max_y - y_bin*0.2
            y2 = max_y - (y_bin+1)*0.2
            value = velodyne[i][3] #this refers to the intensity value
            if ( x <= x1 and x > x2):
                if( y <= y1 and y > y2):
                    z_per_bin.append(z)
                    value_per_bin.append(value)
                    print("cool")
                    exists = 1
        if(exists == 1):
            highest_z = np.amax(np.array(z_per_bin), 0)
            print(highest_z)
            pixels[x_bin][y_bin] = np.array(value_per_bin)[highest_z]
                

print (np.array(doubles).shape)
print(doubles)





#print("The objects list is " , objects)
#num_of_objects = np.array(objects).shape[0]
#print("Number of object in image ", num_of_objects)