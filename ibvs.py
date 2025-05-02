import cv2
import argparse
import numpy as np
import pyrealsense2 as rs

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 10
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.maxMarkerPerimeterRate = 4.0
parameters.polygonalApproxAccuracyRate = 0.03
parameters.minCornerDistanceRate = 0.05


class Ibvs():
    def __init__(self, mu=0.008):
        '''
        mu: lambda in control law eqn; (λ is the proportional gain)
        '''
        self.args = self.parse_args()

        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    def parse_args(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--mu', type=str, default='0.008', help='lambda in control law eqn; (λ is the proportional gain)')
        return parser.parse_args()

    def get_camera_matrix(self, profile):
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])
        return camera_matrix

    def get_interaction_point(self, s, KK, Z):
        '''
        s: feature points (current) flattened (x1, y1, x2, y2, ..., x4, y4)
        KK: camera intrinsic matrix
        Z:  array for depths of all points 
            OR
            single depth value i.e. taking all points at same depth
        '''
        fx = KK[0,0]
        fy = KK[1,1]
        px = KK[0,2]
        py = KK[1,2]
        
        if isinstance(Z, (int, float)) or (hasattr(Z, 'shape') and len(Z.shape) == 1 and Z.shape[0] == 1):
            # If Z is a single value or array with single value, expand it for all points
            Zarr = np.ones(s.shape[0]) * (Z[0] if hasattr(Z, 'shape') else Z)
        else:
            # If Z already has values for each point
            Zarr = Z

        # Create interaction matrix for all points
        Lsd = np.zeros((s.shape[0], 6), dtype=np.float32)

        for m in range(0, Lsd.shape[0], 2):
            x = (s[m] - px)/fx 
            y = (s[m+1] - py)/fy 
        
            Zinv = 1/Zarr[m//2]
            Lsd[m,:] = np.array([-Zinv, 0, x*Zinv, x*y, -(1+x**2), y])
            Lsd[m+1,:] = np.array([0, -Zinv, y*Zinv, 1+y**2, -x*y, -x])
        
        return Lsd

    def get_cam_velocity(self, Lsd, error):
        '''
        Lsd: Interaction matrix
        error: current_features - target_features
        '''
        vc = -(float(self.args.mu)) * np.matmul(np.linalg.pinv(Lsd), error)
        return vc
    
    def aruco_processing(self, image):
        current_pt = None
        image_copy = image.copy()
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        corner_points = None
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            for i, corner in enumerate(corners):
                corner_points = corner.reshape((4, 2)).astype(int)
            center_x = int(np.mean(corner_points[:, 0]))
            center_y = int(np.mean(corner_points[:, 1]))

            # Draw center point
            cv2.circle(image_copy, (center_x, center_y), 4, (255, 0, 0), -1)

            current_pt = [center_x, center_y]
        return current_pt, image_copy
    
    def calc_points(self, camera_matrix, depth_frame, target_pt, current_pt):
        depth = error = Lsd = vc = None

        current_pt = np.array(current_pt)
        depth = depth_frame.get_distance(int(current_pt[0]), int(current_pt[1]))
        print(f"Current point: {current_pt}, Depth: {depth}")
                    
        if depth is not None and depth!=0:
            target_pt = target_pt.flatten()
            current_pt = current_pt.flatten()

            # Calculate error between current and target features
            error = current_pt - target_pt

            # Calculate the interaction matrix i.e. Image Jacobian
            Lsd = self.get_interaction_point(current_pt, camera_matrix, depth)

            # Calculate camera velocity using pseudo-inverse
            vc = self.get_cam_velocity(Lsd, error)
                        
        return error, Lsd, vc
            

    def run(self):
        profile = self.pipeline.start(self.config)
        camera_matrix = self.get_camera_matrix(profile)
        print(f"Camera  matrix: {camera_matrix}")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                if frames is not None:
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    color_image = np.asanyarray(color_frame.get_data())
                    cv2.imshow("Capture", color_image)

                    height, width, _ = color_image.shape
                    target_pt = [width // 2, height // 2]
                    target_pt = np.array(target_pt)
                    print(f"Target point: {target_pt}")

                    current_pt, processed_image = self.aruco_processing(color_image)
                    cv2.imshow("Aruco processed", processed_image)

                    if all(x is not None for x in (target_pt, current_pt, camera_matrix)):
                        error, Lsd, vc = self.calc_points(camera_matrix, depth_frame, target_pt, current_pt)
                        if error is not None:
                            print(f"Error vector: {error}\n")
                            print(f"Interaction matrix: {Lsd.shape}\n{Lsd}\n")
                            print(f"Camera spatial velocity vector: {vc}\n")
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break                
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()



if __name__ == "__main__":
    ibvs_inst = Ibvs(mu=0.008)
    ibvs_inst.run()
