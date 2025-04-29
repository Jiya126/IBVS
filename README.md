# Image based Visual Servoing - Interaction Matrix calculation

Interaction matrix calculation using aruco markers four corner points, using realsense camera

## Setup
1. Python >= 3.8
2. `pip install -r requirements.txt`
3. Intel Realsense Camera connected
4. Aruco marker [(DICT_5X5_100)](aruco_marker.png)
### Directions of use
`python3 ibvs.py`

- Modify ` mu ` (1/sec) i.e. lambda in control law equation
- Save the desired Image features, press ` s ` key when on the desired position above aruco marker
- Similarly save the current Image feature using ` s ` key with aruco marker in frame  (Saves these images inside the `images_captured` folder)
- The matrix calculation is done as per these saved poses and images

### Output
- Interaction matrix (8,6) - (2*4,6) for 4 aruco corner points
- Camera Spatial velocity (1,6)
- Units: m/s, rad/s

