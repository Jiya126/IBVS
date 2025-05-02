# Image based Visual Servoing - Interaction Matrix and Camera Velocity

Interaction matrix and camera velocity calculation using aruco markers centre point and Intel Realsense camera

## Setup
1. Python >= 3.9
2. `pip install -r requirements.txt`
3. Intel Realsense Camera connected
4. Aruco marker [(DICT_5X5_100)](aruco_marker.png)
### Directions of use
`python3 ibvs.py --mu "0.008"`

- Modify ` mu ` (1/sec) i.e. lambda in control law equation

### Output
- Error (1,2)
- Interaction matrix (2,6)
- Camera Spatial velocity (1,6)
- Units: m/s, rad/s
