import cv2
import os
import time
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

corner_colors = {
    "Top-Left": (255, 0, 0),      # Blue
    "Top-Right": (0, 255, 0),     # Green
    "Bottom-Right": (0, 0, 255),  # Red
    "Bottom-Left": (0, 255, 255)  # Yellow
}
corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]


class Aruco():
    def __init__(self, cam_index=0, image_path="", foldername="images_captured"):
        self.cam_index = cam_index
        self.image_path = image_path
        self.foldername = foldername
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def get_camera_matrix(self):
        try:
            profile = self.pipeline.start(self.config)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            camera_matrix = np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1]
            ])
            return camera_matrix
        finally:
            self.pipeline.stop()
    
    def aruco_processing(self, image):
        image_copy = image.copy()
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        corner_points = None
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            for i, corner in enumerate(corners):
                corner_points = corner.reshape((4, 2)).astype(int)
                for j, point in enumerate(corner_points):
                    color = corner_colors[corner_labels[j]]
                    x, y = point
                    cv2.circle(image_copy, (x, y), 6, color, -1)
        return corner_points, image_copy
    
    def calc_points(self, static):
        corner_points = None
        depths = None
        if static:
            try:
                self.pipeline.start(self.config)
                image = cv2.imread(self.image_path)
                corner_points, processed_image = self.aruco_processing(image)
                cv2.imshow("Processed Image", processed_image)
                time_sec = time.time()
                filename = f"static_{time.ctime(time_sec)}.png"
                cv2.imwrite(f"{self.foldername}/{filename}", processed_image)
                cv2.waitKey(0)
            finally:
                cv2.destroyAllWindows()
                self.pipeline.stop()

        else:
            try:
                self.pipeline.start(self.config)
                while True:
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()

                    if frames is not None:
                        color_image = np.asanyarray(color_frame.get_data())
                        cv2.imshow("Capture", color_image)
                        corner_points, processed_image = self.aruco_processing(color_image)
                        if corner_points is not None:
                            cv2.imshow("Processed Image", processed_image)
                            time_sec = time.time()
                            filename = f"Nstatic_{time.ctime(time_sec)}.png"
                            
                            # Get depth for all 4 corner points
                            depths = []
                            for point in corner_points:
                                x, y = int(point[0]), int(point[1])
                                depth = depth_frame.get_distance(x, y)
                                depths.append(depth)
                                print(f"Corner ({x}, {y}) - Depth: {depth}")
                            
                            # Take average depth if needed (or keep individual depths)
                            avg_depth = sum(depths)/len(depths) if depths else None
                            
                            if cv2.waitKey(1) & 0xFF == ord('s'):
                                cv2.imwrite(f"{self.foldername}/{filename}", processed_image)
                                break

            finally:
                cv2.destroyAllWindows()
                self.pipeline.stop()

        # Return average depth for backward compatibility
        avg_depth = sum(depths)/len(depths) if depths else None
        return corner_points, avg_depth