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
        
        
        self.__landmark_counter = 0
        
        self.K = K
        self.K_inv = np.linalg.inv(K)
        
        self.radius = radius
        
        
        # Store the intersection points
        self.intersections = {}
        
        
    def plot_coordinate_system(self):
        length = 1
        self.ax.plot([0, length], [0, 0], [0, 0], "--", color='black')  # X-axis
        self.ax.plot([0, 0], [0, length], [0, 0], "--", color='black')  # Y-axis
        self.ax.plot([0, 0], [0, 0], [0, length], "--", color='black')  # Z-axis

        
    def add_cam(self,cam_center,R_c2g):
        w = 2
        h = 2
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

        self.ax.scatter3D(cam_center[0,0], cam_center[1,0], cam_center[2,0], color='r')
        self.ax.text(cam_center[0,0], cam_center[1,0]-0.2, cam_center[2,0]-0.2, r'$\mathcal{C}$'+f'$_{self.__cam_counter}$', color='black', fontsize = 20)
        
        # Append the cam into our vector
        self.cam_centers.update({self.__cam_counter : cam_center})
        self.cam_rots.update({self.__cam_counter : R_c2g})
        self.img_corners.update({self.__cam_counter : corners})
        
        # Increase the counter
        self.__cam_counter += 1
        
    def add_landmark(self,t_in_g):
        self.__landmark_counter += 1
        t_in_g = t_in_g.reshape(3,1)
        
        # Show the landmark
        self.ax.scatter3D(t_in_g[0,0], t_in_g[1,0], t_in_g[2,0], color='green')
        
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx]
            x_points = [t_in_g[0,0], cam_center[0,0]]
            y_points = [t_in_g[1,0], cam_center[1,0]]
            z_points = [t_in_g[2,0], cam_center[2,0]]
            self.ax.plot(x_points,y_points,z_points, color='green')
            
            # Find the plane Equation
            corners =  self.img_corners[cam_idx]
            corners_homogenous = np.ones((5,4))
            corners_homogenous[:,0:3] = corners
            plane_homogenous_coordinates = null_space(corners_homogenous)
            plane_homogenous_coordinates = plane_homogenous_coordinates / plane_homogenous_coordinates[3,0]
            
            # Homogeneous coordinates of center
            center_homogeneous = np.ones((4,1))
            center_homogeneous[0:3,:] = cam_center.reshape(3,1)
            
            # This is the vector from cam center to point
            m = t_in_g - cam_center
            m_homogeneous = np.ones((4,1))
            m_homogeneous[0:3,:] = m.reshape(3,1)
            
            
            # Now we are ready to compute the intersection of the img plane and the projection line
            t = - np.dot(plane_homogenous_coordinates.transpose(), center_homogeneous) / ( np.dot(plane_homogenous_coordinates.transpose(), m_homogeneous) -  plane_homogenous_coordinates[3,0])
            intersection = cam_center + t * m
            intersection = intersection.reshape(3,1)
            self.ax.scatter3D(intersection[0,0], intersection[1,0], intersection[2,0], marker=(5, 2), color='green', s = 50)
            #self.ax.text(intersection[0,0]+0.05, intersection[1,0], intersection[2,0], r'$\mathcal{p}$' + f"$_{self.__landmark_counter}$", color='black', fontsize = 20)
            
            self.intersections.update({cam_idx : intersection})
            
    def add_landmark2(self,t_in_g):
        self.__landmark_counter += 1
        t_in_g = t_in_g.reshape(3,1)
        
        
        # Show the landmark
        self.ax.scatter3D(t_in_g[0,0], t_in_g[1,0], t_in_g[2,0], color='green')
        
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx].reshape(3,1)
            R_g2c = self.cam_rots[cam_idx].transpose()
            t_in_cam = R_g2c @ t_in_g - R_g2c @ cam_center
            
            pixel_coordinates = self.K @ t_in_cam
            
            pixel_coordinates_normalized = pixel_coordinates / pixel_coordinates[2,0]
            
            print(f"pixel_coordinate : {pixel_coordinates}")
            print(f"pixel_coordinates_normalized : {pixel_coordinates_normalized}")
 
            # Find the plane Equation
            corners =  self.img_corners[cam_idx]
            corners_homogenous = np.ones((5,4))
            corners_homogenous[:,0:3] = corners
            plane_homogenous_coordinates = null_space(corners_homogenous)
            plane_homogenous_coordinates = plane_homogenous_coordinates / plane_homogenous_coordinates[3,0]
            
            # Homogeneous coordinates of center
            center_homogeneous = np.ones((4,1))
            center_homogeneous[0:3,:] = cam_center.reshape(3,1)
            
            # Visualize Observation
            observation_in_global = self.K_inv @ pixel_coordinates
            observation_in_global_normalized = self.K_inv @ pixel_coordinates_normalized
            
            # This is the vector from cam center to the pixels in 3D
            m = observation_in_global - cam_center
            m_homogeneous = np.ones((4,1))
            m_homogeneous[0:3,:] = m.reshape(3,1)
            
            
            # Now we are ready to compute the intersection of the img plane and the projection line
            t = - np.dot(plane_homogenous_coordinates.transpose(), center_homogeneous) / ( np.dot(plane_homogenous_coordinates.transpose(), m_homogeneous) -  plane_homogenous_coordinates[3,0])
            intersection = cam_center + t * m
            intersection = intersection.reshape(3,1)
            self.ax.scatter3D(observation_in_global[0,0], observation_in_global[1,0], observation_in_global[2,0], marker=(5, 2), color='blue', s = 50)
            self.ax.scatter3D(observation_in_global_normalized[0,0], observation_in_global_normalized[1,0], observation_in_global_normalized[2,0], marker=(5, 2), color='red', s = 50)
            
            
            # x_points = [cam_center[0,0], cam_center[0,0] + (intersection[0,0] - cam_center[0,0]) * self.radius]
            # y_points = [cam_center[1,0], cam_center[1,0] + (intersection[1,0] - cam_center[1,0]) * self.radius]
            # z_points = [cam_center[2,0], cam_center[2,0] + (intersection[2,0] - cam_center[2,0]) * self.radius]
            # self.ax.plot(x_points,y_points,z_points, color='green')
            
            
    
    def plot_noisy_observations(self,observations):
                
        # Plot a line between the cam centers and the landmark
        for cam_idx in range(self.__cam_counter):
            cam_center = self.cam_centers[cam_idx]

            # Find the plane Equation
            corners =  self.img_corners[cam_idx]
            corners_homogenous = np.ones((5,4))
            corners_homogenous[:,0:3] = corners
            plane_homogenous_coordinates = null_space(corners_homogenous)
            plane_homogenous_coordinates = plane_homogenous_coordinates / plane_homogenous_coordinates[3,0]
            
            # Homogeneous coordinates of center
            center_homogeneous = np.ones((4,1))
            center_homogeneous[0:3,:] = cam_center.reshape(3,1)
            
            # Visualize Observation
            observation_in_cam = np.array([observations[cam_idx,0],observations[cam_idx,1],1]).reshape(3,1)
            observation_in_global = self.K_inv @ observation_in_cam
            
            # This is the vector from cam center to the pixels in 3D
            m = observation_in_global - cam_center
            m_homogeneous = np.ones((4,1))
            m_homogeneous[0:3,:] = m.reshape(3,1)
            
            
            # Now we are ready to compute the intersection of the img plane and the projection line
            t = - np.dot(plane_homogenous_coordinates.transpose(), center_homogeneous) / ( np.dot(plane_homogenous_coordinates.transpose(), m_homogeneous) -  plane_homogenous_coordinates[3,0])
            intersection = cam_center + t * m
            intersection = intersection.reshape(3,1)
            self.ax.scatter3D(intersection[0,0], intersection[1,0], intersection[2,0], marker=(5, 2), color='black', s = 50)
            
            
            x_points = [cam_center[0,0], cam_center[0,0] + (intersection[0,0] - cam_center[0,0]) * self.radius]
            y_points = [cam_center[1,0], cam_center[1,0] + (intersection[1,0] - cam_center[1,0]) * self.radius]
            z_points = [cam_center[2,0], cam_center[2,0] + (intersection[2,0] - cam_center[2,0]) * self.radius]
            self.ax.plot(x_points,y_points,z_points, color='black')
            


            
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
    vis = _vis()
    
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
    
    