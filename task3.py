from load_data import load_data
from data_utils import line_color
import os
from task2 import project_lidar_data_on_image
from matplotlib import pyplot as plt
import numpy as np
import cv2
from vispy.color import ColorArray
from PIL import Image

def convert_rad_to_deg(value):
    return value * 57.2958

def range_of(array):
    print(array.shape)
    max_ = np.amax(array)
    min_ = np.amin(array)
    return tuple([min_,max_])

def project_laser_id(points , projected_points , img , save):
    ''' this method receives the lidar point cloud, 
        as well as the corresponding 2D projected points on the image '''

    lines_of_search = 64 
    # According to the VELODYNE HDL datashet HDL-64E
    min_FOV = -24.9
    max_FOV = 2.0
    FOV = max_FOV - min_FOV
    # so this gives us a range for our theta angle and we have to divide each 3D point to which part of the FOV it belongs to
    FOV_interval = FOV/lines_of_search
    angle = 2.0
    theta = []
    # Divide the angles in different bins. Seperate bins with interval of approximately 2 degrees (according to calculation)
    for i in range(0 ,lines_of_search):
        theta.append(angle)
        angle -= FOV_interval
    theta_vector =np.array(theta)

    #Find which points belong to which FOV lines, depending on their atan
    # we only need x (the forward direction) and z (the upward direction)
    x = points[: , 0]
    z = points[: ,2]
    tan_values = np.divide (z , x)
    theta_of_points = np.arctan(tan_values) # get the angles in radians for all points
    theta_deg_of_points = convert_rad_to_deg(theta_of_points) # get the agnels in degrees for all points
    print(range_of(theta_deg_of_points))

    #Filter out indexes that are outside this range 
    index = np.logical_and(max_FOV>=theta_deg_of_points , theta_deg_of_points >=min_FOV)
    theta_deg_of_points = theta_deg_of_points[index] # keep only the respective angles, for the respective ponits in the FOV
    projected_points  = projected_points [index] #get also all the projected points that are within are FOV 



    # get the angles as range with each array element holding the starting angle and end_angle for each bin
    list_theta_range= []
    for i in range(1, theta_vector.shape[0]):
        list_theta_range.append([theta_vector[i], theta_vector[i-1]])
    theta_vector_ranges = np.array(list_theta_range)


    #Find the indeces corresponding to the various bins
    line_ids = []
    for i , theta_range in enumerate(theta_vector_ranges):
        max_theta = theta_range[0]
        min_theta = theta_range[1]
        index_bin = np.logical_and(theta_deg_of_points<=min_theta , theta_deg_of_points>=max_theta) #find index of points that match this bin
        line_ids.append(index_bin)
        
    #replace points with this index with appropriate line ID
    id = 1
    for index in line_ids:
        projected_cloud_pixels = projected_points[index]
        for point in enumerate(projected_cloud_pixels):
            pixels = point[1]
            u = pixels[1] # corresponds to y
            v = pixels[0] # corresponds to x 
            color_HSV  = np.uint8([[[line_color(id) , 255 , 255]]])
            color_RGB =  cv2.cvtColor(color_HSV, cv2.COLOR_HSV2RGB)
            color_of_line = (int(color_RGB[0,0,0]) , int(color_RGB[0,0,1]), int(color_RGB[0,0,2]))
            img = cv2.circle(img, (v,u), radius=1, color= color_of_line, thickness=1) # color is determined according to laser id
            #img[point[1],point[0],:] = color[i,:]
        id += 1
    


    #Check if you need to save the image
    if(save == 1):
        cv2.imwrite("task3.jpg" ,cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) #the image is expressed as BGR , so convert when saving
    #PIL method - Show pic with lines of search
    image= Image.fromarray(img, "RGB")
    image.show("TASK 3 - LASER ID ON IMAGE")
    return image



################## MAIN ####################
def main():
    print("So let's start the task")
    data_path = os.path.join("./data", "demo.p") # change to data.p for submission
    data = load_data(data_path)
    velodyne =  data['velodyne']
    objects = data['objects']
    #Get image and display it
    image= data['image_2']
    print( "our image is ", image.dtype)
    image = np.uint8(image) #convert uint32 to uint8 with RGB channels
    print( "our image is ", image.dtype , " with shape " , image.shape)
    image_original = np.copy(image)
    lidar_points , projected_points , output_image  = project_lidar_data_on_image(data , objects , image , bbox = 'no')
    output2_image = project_laser_id(lidar_points ,projected_points , image_original , save = 1)


if __name__ == "__main__":
    main()