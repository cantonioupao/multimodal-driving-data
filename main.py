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
max_velodyne = np.amax(velodyne, 0)
max_x = max_velodyne [0]
max_y = max_velodyne [1]
max_z = max_velodyne [2]
min_velodyne = np.amin(velodyne, 0)
min_x = min_velodyne [0]
min_y = min_velodyne [1]
min_z = min_velodyne [2]
print("The maximum and minimum values for x, y  and z are ", max_x , max_y, max_z, "and respectively ", min_x, min_y,min_z)
doubles = []

for i in range (0,50): #the ith example
    print (" x = %a" % velodyne[i][0] , " y = %a" % velodyne[i][1] , " z = %a" % velodyne[i][2], " intensity = %a" % velodyne[i][3])
    #search for same x and y
    for j in range (0,50):
        if (velodyne[i][0] == velodyne [j][0] and velodyne[i][1] == velodyne[j][1]):
            highest_z = max(velodyne[i][2], velodyne[j][2])
            doubles.append(highest_z)

print (np.array(doubles).shape)
print(doubles)


#print("The objects list is " , objects)
#num_of_objects = np.array(objects).shape[0]
#print("Number of object in image ", num_of_objects)