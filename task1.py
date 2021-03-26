from load_data import load_data
import numpy as np
import os
import cv2

print("So let's start the task")
data_path = os.path.join("./data", "data.p")
data = load_data(data_path)
velodyne =  data['velodyne']
objects = data['objects']
print("The first 3 dimensions represent x,y,z and the last one represents the reflectance intensity  between 0 - 1")
print("Shape is "+ str(velodyne.shape))
print("We know that the maximum range is 120 m" )
num_of_points = velodyne.shape[0]


# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def BEV_from_cloud(points, saveto , side_range = 0,fwd_range= 0,res = 0 ,min_height= 0, max_height=0):
    """ Creates an 2D birds eye view representation of the point Lidar 3D cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z and the intensity value/reflectance
                    shape is n x 4
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    max_velodyne = np.amax(points, 0)
    max_x = max_velodyne [0]
    max_y = max_velodyne [1]
    max_z = max_velodyne [2]
    min_velodyne = np.amin(points, 0)
    min_x = min_velodyne [0]
    min_y = min_velodyne [1]
    min_z = min_velodyne [2]


    #Assign to the corresponding variables
    side_range =(min_y ,max_y )
    fwd_range = (min_x , max_x)
    max_height = max_z
    min_height = min_z
    
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3]  # Reflectance
    
    #side_range = range_of_vec(points[:,1])
    #fwd_range = range_of_vec(points[:,0])
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))



    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    #pixel_values  = scale_to_255(z_lidar, min=min_height, max=max_height)
    pixel_values = scale_to_255(r_lidar,min=0.0,max=1.0)
    
    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    img = np.zeros([y_max, x_max], dtype=np.uint8)
    img[-y_img, x_img] = pixel_values # -y because images start from top left


    
    
    # SAVE THE IMAGE
    if saveto :
        cv2.imwrite(saveto, img) 
        print("The file is saved under ", saveto)
    else:
        cv2.imshow('image', img)
        cv2.waitKey(10000)
        print("You can view the BEV from the 3D Lidar points")
    


def main():
    #Convert the 3D Lidar points to BEV view
    BEV_from_cloud(velodyne , res = 0.2 , saveto = "")

if __name__ == "__main__":
    main()