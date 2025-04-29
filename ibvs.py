import cv2
import numpy as np
from aruco import Aruco

mu = 0.008  #lambda in control law eqn; (λ is the proportional gain)

# Calculate Interaction matrix
def get_interaction_point(s, KK, Z):
    '''
    s: feature points flattened (x1, y1, x2, y2, ..., x4, y4)
    KK: camera intrinsic matrix
    Z:  array for depths of all points 
        OR
        single depth value i.e. taking all points at same depth
    '''
    px = KK[0,0]
    py = KK[1,1]
    u0 = KK[0,2]
    v0 = KK[1,2]
     
    if isinstance(Z, (int, float)) or (hasattr(Z, 'shape') and len(Z.shape) == 1 and Z.shape[0] == 1):
        # If Z is a single value or array with single value, expand it for all points
        Zarr = np.ones(s.shape[0]) * (Z[0] if hasattr(Z, 'shape') else Z)
    else:
        # If Z already has values for each point
        Zarr = Z

    # Create interaction matrix for all points (8×6 matrix for 4 points)
    Lsd = np.zeros((s.shape[0], 6), dtype=np.float32)

    for m in range(0, Lsd.shape[0], 2):
        x = (s[m] - u0)/px 
        y = (s[m+1] - v0)/py 
    
        Zinv = 1/Zarr[m//2]
        Lsd[m,:] = np.array([-Zinv, 0, x*Zinv, x*y, -(1+x**2), y])
        Lsd[m+1,:] = np.array([0, -Zinv, y*Zinv, 1+y**2, -x*y, -x])
    
    return Lsd


aruco_inst = Aruco(cam_index=0, image_path="")
camera_matrix = aruco_inst.get_camera_matrix()
print(f"Camera Matrix:\n{camera_matrix}\n")

# Get desired features (all 4 corners)
corners_des, desired_depth = aruco_inst.calc_points(static=False)
if corners_des is not None:
    # Use all four corner points
    kp_des = corners_des.flatten()
    print(f"\nDesired feature points (all corners):\n{kp_des}\n")

# Using the same depth for all points
if desired_depth is not None:
    desired_depths = np.ones(4) * desired_depth
    print(f"\nDesired depths: {desired_depths}\n")

# Get current features
corners_curr, _ = aruco_inst.calc_points(static=False)
if corners_curr is not None:
    # Use all four corner points
    kp_curr = corners_curr.flatten()
    print(f"\nCurrent feature points (all corners):\n{kp_curr}\n")

    # Calculate error between current and desired features
    error = kp_curr - kp_des
    print(f"\nError vector:\n{error}\n")

    # Calculate interaction matrix for all points
    Lsd = get_interaction_point(kp_des, camera_matrix, desired_depths)
    print(f"\nInteraction matrix: {Lsd.shape}\n{Lsd}\n")

    # Calculate camera velocity using pseudo-inverse
    vc = -mu * np.matmul(np.linalg.pinv(Lsd), error)
    print(f"\nCamera spatial velocity vector:\n{vc}")
else:
    print("Failed to detect marker corners")