import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve
from scipy.linalg import null_space


# This as to add transparent patches in figures
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class _vis:
    def __init__(self, K, radius):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')        
        #self.plot_coordinate_system()
        
        self.cam_centers = {}
        self.cam_rots = {}
        self.img_corners = {}
        self.__cam_counter = 0
        
        self.cam_centers_noisy = {}
        self.cam_rots_noisy = {}
        self.img_corners_noisy = {}
        
        self.__landmark_counter = 0
        
        # Intrinsic matrix
        self.K = K
        self.K_inv = np.linalg.inv(K)
        
        # Visualization
        self.img_widht = 2
        self.img_heigth = 2
        
        self.scale_x = self.img_widht / self.K[0,0]
        self.scale_y = self.img_heigth / self.K[1,1]
        
        self.radius = radius
        
        # Store the intersection points
        self.intersections = {}
        
        
    def plot_coordinate_system(self):
        length = 1
        self.ax.plot([0, length], [0, 0], [0, 0], "--", color='black')  # X-axis
        self.ax.plot([0, 0], [0, length], [0, 0], "--", color='black')  # Y-axis
        self.ax.plot([0, 0], [0, 0], [0, length], "--", color='black')  # Z-axis

        
    def add_cam(self,cam_center,R_c2g):
        w = self.img_widht
        h = self.img_heigth
        corners = np.array([
            -0.5*w, 0.5*h, 1.0,
            0.5*w, 0.5*h, 1.0,
            0.5*w, -0.5*h, 1.0,
            -0.5*w, -0.5*h, 1.0,
            -0.5*w, 0.5*h, 1.0
        ]).reshape(5,3)
        cam_center = cam_center.reshape(3,1)
        
        # First rotate the corners
        corners = corners @ R_c2g.transpose()
        
        # Now translate the corners so that they are expressed in global coordinate system
        for col_idx in range(3):
            corners[:,col_idx] = corners[:,col_idx] + cam_center[col_idx,0]
            
        self.ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color='b')
        # Create a Poly3DCollection with the specified vertices and faces
        poly3d = Poly3DCollection([corners], alpha=0.2, facecolors='cyan', edgecolors='k')

        # Add the collection to the 3D plot
        self.ax.add_collection3d(poly3d)

        self.ax.scatter3D(cam_center[0,0], cam_center[1,0], cam_center[2,0], color='green')
        self.ax.text(cam_center[0,0], cam_center[1,0]-0.2, cam_center[2,0]-0.2, r'$\mathcal{C}$'+f'$_{self.__cam_counter}$', color='black', fontsize = 20)
        
        # Append the cam into our vector
        self.cam_centers.update({self.__cam_counter : cam_center})
        self.cam_rots.update({self.__cam_counter : R_c2g})
        self.img_corners.update({self.__cam_counter : corners})
        
        # Increase the counter
        self.__cam_counter += 1
        
    def add_cam_noisy(self,cam_center,R_c2g):
        w = self.img_widht
        h = self.img_heigth
        corners = np.array([
            -0.5*w, 0.5*h, 1.0,
            0.5*w, 0.5*h, 1.0,
            0.5*w, -0.5*h, 1.0,
            -0.5*w, -0.5*h, 1.0,
            -0.5*w, 0.5*h, 1.0
        ]).reshape(5,3)
        cam_center = cam_center.reshape(3,1)
        
        # First rotate the corners
        corners = corners @ R_c2g.transpose()
        
        # Now translate the corners so that they are expressed in global coordinate system
        for col_idx in range(3):
            corners[:,col_idx] = corners[:,col_idx] + cam_center[col_idx,0]
            
        self.ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color='b')
        # Create a Poly3DCollection with the specified vertices and faces
        poly3d = Poly3DCollection([corners], alpha=0.2, facecolors='pink', edgecolors='k')

        # Add the collection to the 3D plot
        self.ax.add_collection3d(poly3d)

        self.ax.scatter3D(cam_center[0,0], cam_center[1,0], cam_center[2,0], color='red')
        self.ax.text(cam_center[0,0], cam_center[1,0]-0.2, cam_center[2,0]-0.2, r'$\mathcal{C}$'+f'$_{self.__cam_counter-1}$', color='black', fontsize = 20)
        
        # Append the cam into our vector
        self.cam_centers_noisy.update({self.__cam_counter -1 : cam_center})
        self.cam_rots_noisy.update({self.__cam_counter -1 : R_c2g})
        self.img_corners_noisy.update({self.__cam_counter -1 : corners})
             
    def add_landmark(self,t_in_g):
        self.__landmark_counter += 1
        t_in_g = t_in_g.reshape(3,1)
    
        # Show the landmark
        self.ax.scatter3D(t_in_g[0,0], t_in_g[1,0], t_in_g[2,0], color='green')
        
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx].reshape(3,1)
            R_g2c = self.cam_rots[cam_idx].transpose()
            t_in_cam = R_g2c @ (t_in_g - cam_center)
            
            pixels = self.K @ t_in_cam
            pixels_normalized = pixels / pixels[2,0]

            feature_vector_in_cam = self.K_inv @ pixels_normalized
            feature_vector_in_g = R_g2c.transpose() @ feature_vector_in_cam
            feature_coordinates_in_g = feature_vector_in_g + cam_center
            
            self.ax.scatter3D(feature_coordinates_in_g[0,0], feature_coordinates_in_g[1,0], feature_coordinates_in_g[2,0], marker=(5, 2), color='green', s = 50)
            
            x_points = [t_in_g[0,0], cam_center[0,0]]
            y_points = [t_in_g[1,0], cam_center[1,0]]
            z_points = [t_in_g[2,0], cam_center[2,0]]
            self.ax.plot(x_points,y_points,z_points, color='green')
            
            
    def plot_noisy_observations(self,observations):
                
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx]
            R_g2c = self.cam_rots[cam_idx].transpose()

            # Visualize Observation
            observation_in_cam = np.array([observations[cam_idx,0],observations[cam_idx,1],1]).reshape(3,1)
            observation_direction_in_cam = self.K_inv @ observation_in_cam
            observation_direction_in_g = R_g2c.transpose() @ observation_direction_in_cam
            observation_in_g = observation_direction_in_g + cam_center
            
            # print(f"observation_in_cam : {observation_in_cam}")
            # print(f"observation_direction_in_global : {observation_direction_in_g}")
            # print(f"observation_in_global : {observation_in_g}")
            
            self.ax.scatter3D(observation_in_g[0,0], observation_in_g[1,0], observation_in_g[2,0], marker=(5, 2), color='black', s = 50)
            
            
            x_points = [cam_center[0,0], cam_center[0,0] + (observation_in_g[0,0] - cam_center[0,0]) * self.radius]
            y_points = [cam_center[1,0], cam_center[1,0] + (observation_in_g[1,0] - cam_center[1,0]) * self.radius]
            z_points = [cam_center[2,0], cam_center[2,0] + (observation_in_g[2,0] - cam_center[2,0]) * self.radius]
            self.ax.plot(x_points,y_points,z_points, color='black')
            
    def plot_noisy_observations_on_noisy_cams(self,observations):
                
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers_noisy[cam_idx]
            R_g2c = self.cam_rots_noisy[cam_idx].transpose()

            # Visualize Observation
            observation_in_cam = np.array([observations[cam_idx,0],observations[cam_idx,1],1]).reshape(3,1)
            observation_direction_in_cam = self.K_inv @ observation_in_cam
            observation_direction_in_g = R_g2c.transpose() @ observation_direction_in_cam
            observation_in_g = observation_direction_in_g + cam_center
            
            # print(f"observation_in_cam : {observation_in_cam}")
            # print(f"observation_direction_in_global : {observation_direction_in_g}")
            # print(f"observation_in_global : {observation_in_g}")
            
            self.ax.scatter3D(observation_in_g[0,0], observation_in_g[1,0], observation_in_g[2,0], marker=(5, 2), color='orange', s = 50)
            
            
            x_points = [cam_center[0,0], cam_center[0,0] + (observation_in_g[0,0] - cam_center[0,0]) * self.radius]
            y_points = [cam_center[1,0], cam_center[1,0] + (observation_in_g[1,0] - cam_center[1,0]) * self.radius]
            z_points = [cam_center[2,0], cam_center[2,0] + (observation_in_g[2,0] - cam_center[2,0]) * self.radius]
            self.ax.plot(x_points,y_points,z_points, color='orange')
            
    def get_plot(self):
        #self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        # Set the limits for each axis
        self.ax.set_xlim([-2, 4])
        self.ax.set_ylim([-2, 4])
        self.ax.set_zlim([-2, 4])
        
        # Disable the grids
        self.ax.grid(False)
        
        # Hide the axes
        self.ax.set_axis_off()

        plt.show()
        


if __name__ == "__main__":
    
    K = np.array([
        [1012.0027, 0, 1054],
        [0., 1012.0027, 581],
        [0., 0., 1.],
    ])
    
    vis = _vis(K)
    
    cam_center = np.array([0,0,2])
    R_c2g = R.from_euler("xyz",[0,0,90], degrees=True).as_matrix()
    vis.add_cam(cam_center,R_c2g)
    

    landmark = np.array([-1,3,1])
    vis.add_landmark(landmark)
    vis.ax.text(landmark[0], landmark[1], landmark[2], r'$\mathcal{P}_1$', color='black', fontsize = 20)
        
    landmark = np.array([1,3,0])
    vis.add_landmark(landmark)
    vis.ax.text(landmark[0], landmark[1], landmark[2], r'$\mathcal{P}_2$', color='black', fontsize = 20)
    
    landmark = np.array([0,3,3])
    vis.add_landmark(landmark)
    vis.ax.text(landmark[0], landmark[1], landmark[2], r'$\mathcal{P}_3$', color='black', fontsize = 20)
    
    
    vis.get_plot()
    
    