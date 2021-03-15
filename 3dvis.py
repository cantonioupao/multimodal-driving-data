# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data
from vispy.color import ColorArray

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
            data = load_data('data/demo.p') # Change to data.p for your final submission 
            P = np.concatenate([points,np.ones([points.shape[0],1])],axis=1)
            # Multiply different parameter matrices to project 3D points on the 2D camera 2 image
            P_2 = np.empty(points.shape)
            R2 = data['P_rect_20'] #intrinsic
            R1 = data['T_cam0_velo'] #extrinsic
            for ind,point in enumerate(P):
                # Projection
                P_2[ind,:] = np.dot(np.dot(R2,R1),point)
                # Normalization
                P_2[ind,:] = P_2[ind,:]/P_2[ind,-1]

            # Filter out ones not inside the perspective
            img = data['image_2']
            ind_x = np.logical_and(P_2[:,1] <= img.shape[0], P_2[:,1] >= 0)
            ind_y = np.logical_and(P_2[:,0] <= img.shape[1], P_2[:,0] >= 0)
            ind = np.logical_and(ind_x,ind_y)

            # Convert the index to integer
            P_clip = P_2[ind].astype(int)
            label_clip = data['sem_label'][ind]

            # Color for each pixel position
            color = np.empty([P_clip.shape[0],3])
            for ind,label in enumerate(label_clip):
                color[ind,:] = data['color_map'][label[0]]
                # BGR -> RGB
                color[ind,:] = [color[ind,-1],color[ind,1],color[ind,0]]

            color = ColorArray(color,clip=True)
        self.sem_vis.set_data(P_clip,edge_color=color, size=3)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordinly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('data/data.p') # Change to data.p for your final submission 
    visualizer = Visualizer()
    visualizer.update(data['velodyne'][:,:3])
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    # Finding the bbox
    # TODO: Clip the point to the edge
    # (N,8,3) dimension array of corners
    corners = np.empty([objects.shape[0],8,3])
    for i,individual in enumerate(objects):
        # Height, Width, Length 
        dimension = np.array(individual[8:11])
        # Right, Down, Front (X,Y,Z)
        location = np.array(individual[11:14])
        rotation_y = individual[14]
        corner = np.zeros([8,3])
        corner[0,:] = location + np.array([dimension[2]/2,-dimension[0],dimension[1]/2])
        corner[1,:] = corner[0,:] + np.array([-dimension[-1],0,0])
        corner[2,:] = corner[1,:] + np.array([0,0,-dimension[1]])
        corner[3,:] = corner[2,:] + np.array([dimension[2],0,0])
        corner[4,:] = corner[0,:] + np.array([0,dimension[0],0])
        corner[5,:] = corner[4,:] + np.array([0,0,-dimension[1]])
        corner[6,:] = corner[5,:] + np.array([0,0,-dimension[1]])
        corner[7,:] = corner[6,:] + np.array([dimension[2],0,0])
        # Rotate along the y-axis
        R = np.array([np.cos(rotation_y),0,np.sin(rotation_y),0;0,1,0,0;-np.sin(rotation_y),0,np.cos(rotation_y),0;0,0,0,1])
        for ind,point in enunmerate(corner):
            corner[ind,:] = np.dot(R,point)
        corners[i,:,:] = corner
    
    vispy.app.run()




