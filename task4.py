
#Velodyne HDL64 runs at 10Hz -> 10 rotations per second (10 times ->360 deg)
# 36000 deg per sec 
#linear motion at 50 km/h causes a gap of 1.38 m between start & end of scan
#rotation motion at a rotation motion at 25/s creates a gap of 2.19 m 
#at a distance of 50 m.
# Use Camera, Lidar & IMU/GPS location data and make use of frame timestamps
# Remember that the velodyne goes clockwise -> towards the negative direction
from load_data import load_data
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
from vispy.color import ColorArray
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from data_utils import load_from_bin
from data_utils import calib_velo2cam
from data_utils import depth_color
from data_utils import load_oxts_velocity
from data_utils import load_oxts_angular_rate
from data_utils import calib_cam2cam
from data_utils import compute_timestamps
from data_utils import print_projection_plt
#data = load_data(os.getcwd()+"/data/demo.p")
data = load_data(os.getcwd()+"/data/demo.p")
velodyne =  data['velodyne']
objects = data['objects']
print("The first 3 dimensions represent x,y,z and the last one represents the reflectance intensity  between 0 - 1")
print("Shape is "+ str(velodyne.shape))
print("We know that the maximum range is 120 m" )
num_of_points = velodyne.shape[0]

from dateutil import parser
def import_date_from_file(file):
    lines = []
    with open( file , 'r') as file:
        data = file.read()
    date_strings= data.split('\n')
    date_strings = date_strings[:-1]
    for date_string in date_strings:
        date_string.split(' ')[1]
        datetime_obj = parser.parse(date_string)
        lines.append(datetime_obj)
    
    return np.array(lines)

file_end = "data/problem_4/velodyne_points/timestamps_end.txt"
file_start = "data/problem_4/velodyne_points/timestamps_start.txt"
file_trigger = "data/problem_4/velodyne_points/timestamps.txt"
file_image_taken = "data/problem_4/image_02/timestamps.txt"
velodyne_points = "data/problem_4/velodyne_points"
oxts_file = "data/problem_4/oxts/timestamps.txt"
calib_velo2cam_filepath = "data/problem_4/calib_velo_to_cam.txt"
calib_cam2cam_filepath = "data/problem_4/calib_cam_to_cam.txt"
calib_imu2velo_filepath = "data/problem_4/calib_imu_to_velo.txt"
example = 37    

# Timestamp of Velodyne
t_velo_start = compute_timestamps(file_start,example)
t_velo_end = compute_timestamps(file_end,example)

# Get velodyne point clouds for all recordings/rotations
def load_vel_pts(mypath, example):
    onlyvelodynefiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    point_cloud = 0 ; #initialize
    velo_file = 0
    for file in onlyvelodynefiles:
        if (file == str(example).zfill(10)+'.bin'):
            file_path = join(mypath, file)
            velo_file = file_path
            points = load_from_bin(file_path)
    return points
mypath = "data/problem_4/velodyne_points/data/"
points = load_vel_pts(mypath, example)

#get all corresponding frames
image_path = 'data/problem_4/image_02/data'
onlyimagefiles = [f for f in listdir(image_path) if isfile(join(image_path,f))]
img = 0 #initialize
for file in onlyimagefiles:
    if (file == str(example).zfill(10)+'.png'):
            file_path = join(image_path, file)
            print(file_path)
            img = cv2.imread(file_path)
            im = Image.fromarray(np.uint8(img), "RGB")
        

# Timestamp of IMU sensor activation
oxts_filepath = "data/problem_4/oxts/timestamps.txt"
t_imu = compute_timestamps(oxts_filepath,example)
imu_filepath = "data/problem_4/oxts/data/{0:010d}.txt".format(example)
lin_velocity = load_oxts_velocity(imu_filepath)
ang_velocity = load_oxts_angular_rate(imu_filepath)
dt = t_velo_end - t_velo_start

def find_distance_of_3D_points(points):
    '''This takes as input argument the 3D point cloud np.array
        and returns an nparray corresponding to the distance for each point'''
    print(points)
    distances  = np.sum(points**2,axis = 1)
    print(distances[100])
    return distances

def project_colored_points(calib_velo2cam_filepath, calib_cam2cam_filepath, points, img):
    #get real world distance for each projected point on the image
    distances = find_distance_of_3D_points(points)
    distances /= 75
    
    #perform Rigid transformation
    ind = points[:,0]>=0 # keep only the points that have an x>=0 
    points = points[ind] #keep the points only with the corresponding indeces
    distances = distances[ind]
    # Homonegenous
    points = np.concatenate([points,np.ones([points.shape[0],1])],axis=1)
    R , T = calib_velo2cam(calib_velo2cam_filepath) #get the R and T matrices to project on camera coordinates
    print("The shape of the rotation matrix is ", R.shape)
    print("The shape of the translation matrix is ", T.shape)
    R1 = np.block([[R , T],[0,0,0,1]])
    R2 = calib_cam2cam(calib_cam2cam_filepath,'02')
    
    points_uv = np.empty([points.shape[0],2])
    for ind,point in enumerate(points):
        # Projection
        projection = np.dot(np.dot(R2,R1),point)
        # Normalization
        points_uv[ind,:] = projection[:-1]/projection[-1] 
    points = points_uv.astype(int)
    

    # Filter out ones not inside the perspective
    ind_x = np.logical_and(points[:,1] < img.shape[0], points[:,1] >= 0)
    ind_y = np.logical_and(points[:,0] < img.shape[1], points[:,0] >= 0)
    ind_z = points[:,-1]>=0
    ind = np.logical_and(ind_x,ind_y)
    ind = np.logical_and(ind,ind_z)
    

    #Convert the index to integer
    points = points[ind].astype(int)
    distances = distances[ind]


    # Color for each pixel position
    color = depth_color(distances)
    
    img = print_projection_plt(points.transpose(),color,img)
    return img
    
# Determine which actual point_cloud and test_frame corresponds to the 37th frame   
#TASK 4 PART A
points = load_vel_pts(mypath, example)
test_image= project_colored_points(calib_velo2cam_filepath, calib_cam2cam_filepath, points, img)
cv2.imshow("TASK4a" , test_image)
cv2.waitKey(5000)
cv2.imwrite("task4a.jpg" , test_image)
img2 = Image.fromarray(np.uint8(test_image), "RGB")
img2

def transform(lin_velocity,ang_velocity,calib_imu2velo_filepath):
    ry = ang_velocity[1] * -dt
    R = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]]) 
    T = np.array([0,0,0]).reshape(-1,1)
    T = lin_velocity.reshape(-1,1)* -dt/10
    R1 = np.block([[R,T],[0,0,0,1]])
    R1 = R1 
    R1 = R1 / R1[-1,-1]

    # imu2velo
    S = []
    with open(calib_imu2velo_filepath) as f:
        i = 0
        for line in f:
            if i == 0:
                i += 1
                continue
            (key, val) = line.split(": ")
            S.append(val.split("\n")[0])
    R = np.array([float(i) for i in S[0].split(" ")]).reshape(3,3)
    T = np.array([float(i) for i in S[1].split(" ")]).reshape(-1,1)
    R_end = np.block([[R,T],[0,0,0,1]])
    # vel2imu -> imu rotate -> imut2vel
    return np.dot(R_end,np.dot(R1,np.linalg.inv(R_end)))

def project_colored_points_adjusted(calib_velo2cam_filepath, calib_cam2cam_filepath, points, img):
    #get real world distance for each projected point on the image
    distances = find_distance_of_3D_points(points)
    distances /= 75
    
    #perform Rigid transformation

    ind = points[:,0]>=0 # keep only the points that have an x>=0 
    points = points[ind] #keep the points only with the corresponding indeces
    distances = distances[ind]
    # Homonegenous
    points = np.concatenate([points,np.ones([points.shape[0],1])],axis=1)
    R , T = calib_velo2cam(calib_velo2cam_filepath) #get the R and T matrices to project on camera coordinates
    print("The shape of the rotation matrix is ", R.shape)
    print("The shape of the translation matrix is ", T.shape)
    R1 = np.block([[R , T],[0,0,0,1]])
    R2 = calib_cam2cam(calib_cam2cam_filepath,'02')
    
    points_uv = np.empty([points.shape[0],2])
    # Transform due to the movement
    R = transform(lin_velocity,ang_velocity,calib_imu2velo_filepath)
    for ind,point in enumerate(points):
        # Projection
        projection = np.dot(np.dot(R2,R1),np.dot(R,point))
        # Normalization
        points_uv[ind,:] = projection[:-1]/projection[-1] 
 
    points = points_uv.astype(int)
    
    

    # Filter out ones not inside the perspective
    ind_x = np.logical_and(points[:,1] < img.shape[0], points[:,1] >= 0)
    ind_y = np.logical_and(points[:,0] < img.shape[1], points[:,0] >= 0)
    ind_z = points[:,-1]>=0
    ind = np.logical_and(ind_x,ind_y)
    ind = np.logical_and(ind,ind_z)
    

    #Convert the index to integer
    points = points[ind].astype(int)
    distances = distances[ind]


    # Color for each pixel position
    color = depth_color(distances)
    
    img = print_projection_plt(points.transpose(),color,img)
    return img
    
# Determine which actual point_cloud and test_frame corresponds to the 37th frame   
points = load_vel_pts(mypath, example)
test_image= project_colored_points_adjusted(calib_velo2cam_filepath, calib_cam2cam_filepath, points, img)
cv2.imshow("TASK4b" , test_image)
cv2.waitKey(5000)
#save image
cv2.imwrite("task4b.jpg" , test_image)
img3 = Image.fromarray(np.uint8(test_image), "RGB")
img3
