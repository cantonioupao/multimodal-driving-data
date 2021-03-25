<<<<<<< HEAD
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
        self.data = load_data('data/demo.p') # Change to data.p for your final submission 
    def update(self, points):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        
        color = np.empty([points.shape[0],3])
        for ind,label in enumerate(self.data['sem_label']):
            color[ind,:] = self.data['color_map'][label[0]]
            # BGR -> RGB
            color[ind,:] = [color[ind,-1],color[ind,1],color[ind,0]]

        color = ColorArray(color,clip=True)
        self.sem_vis.set_data(points,edge_color=color, size=3)
    
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
    corners = np.empty([len(data['objects']),8,3])
    i = 0
    for individual in data['objects']:
        # Height, Width, Length 
        dimension = np.array(individual[8:11])
        bbox_height = dimension[0]
        bbox_width = dimension[1]
        bbox_length =  dimension[2]
        # # Right, Down, Front (X,Y,Z)
        location = np.array(individual[11:14])
        bbox_x = location[0]
        bbox_y = location[1]
        bbox_z = location[2]
        rotation_y = individual[14]
        corner = np.zeros([8,3])

        corner[0,:] = location + np.array([bbox_length/2,-bbox_height,bbox_width/2])
        corner[1,:] = corner[0,:] + np.array([-bbox_length,0,0])
        corner[2,:] = corner[1,:] + np.array([0,0,-bbox_width])
        corner[3,:] = corner[2,:] + np.array([bbox_length,0,0])
        for ind in range(4):
            corner[ind+4,:] = corner[ind,:] + np.array([0,bbox_height,0])
        # Rotate along the y-axis
        T = np.array([[location[0]],[location[1]-bbox_height/2],[location[2]]])
        R = np.array([[np.cos(rotation_y),0,np.sin(rotation_y)],[0,1,0],[-np.sin(rotation_y),0,np.cos(rotation_y)]])
        R1 = np.block([[R,np.dot(-R,T)+T],[0,0,0,1]])
        ind = 0
        for point in corner:
            pos= np.dot(R1,point)
            pos /= pos[-1]
            corner[ind,:] = pos[:-1]
            ind += 1
        corners[i,:,:] = corner
        i += 1
    visualizer.update_boxes(corners)
    vispy.app.run()




=======
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
		self.data = load_data('data/data.p') # Change to data.p for your final submission 
	def update(self, points):
		'''
		:param points: point cloud data
		                shape (N, 3)          
		Task 2: Change this function such that each point
		is colored depending on its semantic label
		'''

		R1 = data["T_cam2_velo"]
		velodyne_points_cam0 = np.dot(np.dot(R1),points)
		color = np.empty([points.shape[0],3])
		for ind,label in enumerate(self.data['sem_label']):
		    color[ind,:] = self.data['color_map'][label[0]]
		    # BGR -> RGB
		    color[ind,:] = [color[ind,-1],color[ind,1],color[ind,0]]

		color = ColorArray(color,clip=True)
		self.sem_vis.set_data(points,edge_color=color, size=3)
    
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
	corners = np.empty([len(data['objects']),8,3])
	i = 0
	for individual in data['objects']:
		dimension = np.array(individual[8:11])
	
		bbox_height = dimension[0]
		bbox_width = dimension[1]
		bbox_length =  dimension[2]
		# Right, Down, Front (X,Y,Z)
		location = np.array(individual[11:14])
		bbox_x = location[0]
		bbox_y = location[1]
		bbox_z = location[2]
		rotation_y = individual[14]
		corner = np.zeros([8,3])

		# corner[0,:] = location + np.array([bbox_width/2,-bbox_height, bbox_length/2])
		# corner[1,:] = corner[0,:] + np.array([-bbox_width ,0 , 0])
		# corner[2,:] = corner[1,:] + np.array([0 ,0,-bbox_length])
		# corner[3,:] = corner[2,:] + np.array([bbox_width, 0, 0])
		# corner[4,:] = corner[0,:] + np.array([0, bbox_height ,0])
		# corner[5,:] = corner[4,:] + np.array([-bbox_width,0, 0])
		# corner[6,:] = corner[5,:] + np.array([0,0,-bbox_length])
		# corner[7,:] = corner[6,:] + np.array([bbox_width,0,0])



	
		# # Height, Width, Length 
		# dimension = np.array(individual[8:11])
		# # Right, Down, Front (X,Y,Z)
		# location = np.array(individual[11:14])
		# rotation_y = individual[14]
		# corner = np.zeros([8,3])
		corner[0,:] = location + np.array([bbox_length/2,-bbox_height ,bbox_width/2])
		corner[1,:] = corner[0,:] + np.array([-bbox_length,0,0])
		corner[2,:] = corner[1,:] + np.array([0  , 0,-bbox_width])
		corner[3,:] = corner[2,:] + np.array([bbox_length ,0,0])
		corner[4,:] = corner[0,:] + np.array([0, bbox_height,0])
		corner[5,:] = corner[4,:] + np.array([-bbox_length,0,0])
		corner[6,:] = corner[5,:] + np.array([0,0,-bbox_width])
		corner[7,:] = corner[6,:] + np.array([bbox_length , 0 , 0])
		# Rotate along the y-axis
		Rx = np.array([[1,0,0] , [0,np.cos(rotation_y), -np.sin(rotation_y)] , [0 ,np.sin(rotation_y) ,np.cos(rotation_y)] ])
		Rz = np.array([[np.cos(rotation_y), -np.sin(rotation_y) , 0] , [np.sin(rotation_y) , np.cos(rotation_y) , 0 ] , [0,0,1] ])
		Ry = np.array([[np.cos(rotation_y),0,np.sin(rotation_y)],[0,1,0], [-np.sin(rotation_y),0, np.cos(rotation_y)]])
		ind = 0
		for point in corner:
			corner[ind,:] = np.dot(Rx,point)
			corner[ind,:] = np.dot(Rz,point)
			ind += 1
		corners[i,:,:] = corner
		i += 1
	visualizer.update_boxes(corners)
	vispy.app.run()


>>>>>>> 91b4c8ee24d8bff22c4492fa657a264e2c1e6f63
