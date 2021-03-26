# Project 1 - Multimodal Driving Data

This is Project 1 of DLAD (Deep learning for Autonomous Driving) that focuses on understanding multimodal driving data.

## Task 1 - BEV
Task 1, focuses on getting the BEV (Bird Eye View) from the velodyne point cloud. The BEv resolution is 0.2 in each direction. The BEV is shown in the figure below.

![BEV of Velodyne point cloud](/pics/task1bev.jpg)

## Task 2 - Semantic Segmentation & Bbox Projection
Task 2 is about projecting the velodyne pint cloud on the image. To achieve this the extrinsic and instrinsic projection matrices are utilized, after the velodyne points are augmeneted with an extra dimension, to implement homegenous coordinates. The extrinsic projection matrix is used, to transform the velodyne point cloud from the world coordinate system, to the camera cooridnate system (cam0). Then the intrinsic projection matrix is further utilized, to transform the points to the camera reference frame of the other camera(cam2). Finally each point of the cloud, now corresponds to a homegenous matrix(3x1) that holds (u,v,1),where u and v correspond to the image2 pixels. Image 2 corresponds naturally to the image taken by camera 2. Also, semantic segmentation of the image scene was performed and different objects have been asssigned different colors
![Velodyne point cloud projection to image 2 with semantic object segmentation](/pics/task2_2.jpg)

For this task, bbox regression of the scene objects was the core focus. The bboxes were projected on the image2. However the bbox coordinates had to be rotated with respect to their origin. In order to do that, and not just rotate them around the camera origin, each object was translated accordingly back to the camera origin frame. Only then the objects bbox points, rotation was performed. This can be viewed as yaw of each of the different cars. After the rotation of each bbox of each object was performed, only then the bbox points were translated back, to where they were on the image. The bbox drawing on the image can be shown below. ![Bounding Box drawing on image 2](/pics/task2_2.jpg) Finally the velodyne point cloud with the respective bounding boxes could be visualized in 3D. Both velodyne points and bbox points are expressed in the camera 0 coordinate system, when displayed in the 3D visualization. It would be nice to see the visualization in the 3D world coordinate system, however expressing everything in the camera 0 3D coordinate system, was much simpler. This can be viewed below ![3D visualization of point cloud and bbox](/pics/3dvis.jpg)

The final image with semantic segmentation and the translated bounding boxes can be shown in the Figure below. ![Semantic Segmentation with bbox on image](/pics/task2.jpg)

## Task 3 - Laser ID
Task 3 emphasized on projecting on image 2, each line of scan from the 64 line of scans of the velodyne. Each different line id, is projected with a different color depending on each id number. The line colors start with the first 8 HSV colors and for the rest of the lines, the colors iterate. ![Laser Line of Scans projected](/pics/task3.jpg)

## Task 4 - Motion Distrotion
Task 4 consisted of 2 subparts. Initially it was the projection of the velodyne points on the image frame, where color would signify the distance of the car from each object/pixel on the scene. The farther away the less blue an object/pixel appears in the image.
The difficult part for this exercise is that we had to construct the extrinsic projection matrix ourselves by stacking the R (rotation matrix) and T (translation matrix) together as one => (R|T). 
Then after using it to transform to the cam0 coordinate system, we had to use the rectified projection matrix to change camera reference frame. Remember that again we were working with homogenous coordinates to enable simpler and easier transformations among coordinate systems. For the second part of the task, the motion blur caused by the motion of the car, as well as the rotation of the velodyne has to be removed. 
During velodyne projection, because of the motion, the points seemed off, relative to the actual image frame. 
![Color distance](/pics/task4a.jpg) ![Removed motion distortion](/pics/task4b.jpg)
The solution was to use the angular velocity and the corresponding timestamps, to calculate the distance that has been travelled/not taken into account. Then this distance is converted to the camera 0 reference frame and the final projection result can be improved, by incorporating the improved projected pixel positions.


## License
[MIT](https://choosealicense.com/licenses/mit/)

***The project assignments are part of [DLAD](https://www.trace.ethz.ch/teaching/DLAD/index.html) from [ETH Zurich](https://ethz.ch/en.html)***
