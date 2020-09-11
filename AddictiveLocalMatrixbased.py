# Import numpy and OpenCV
import numpy as np
import cv2
import math as math


video_path = "video_stairs.mp4"
boundary_constraint = "DIRICHLET_BC"
sigma = 5

# Read input video
cap = cv2.VideoCapture(video_path)

# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = (cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Set up output video
out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

# Read first frame
_, prev = cap.read() 

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

# Pre-define transformation-store array
transforms = [] 

feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.2,
                        minDistance = 10)

lk_params = dict( winSize  = (50,50),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


for i in range(n_frames-2):
  # Detect feature points in previous frame
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,mask= None, **feature_params)
   
  # Read next frame
  success, curr = cap.read() 
  if not success: 
    break 

  # Convert to grayscale
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

  # Calculate optical flow (i.e. track feature points)
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params) 

  assert prev_pts.shape == curr_pts.shape 

  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]

  #Find transformation matrix
  m, _ = cv2.estimateAffine2D(prev_pts,curr_pts)
  if m is None:
    m = np.array([[0,0,0],[0,0,0]])
  
  m = np.concatenate((m, np.array([[0,0,1]])), axis = 0)
    #shjshdjshd AAaaaaAAAAAAMMMMM
  # Store transformation
  transforms.append(m)
   
  # Move to next frame
  prev_gray = curr_gray


#Return smoothed transform matrix from affinity
def AddictiveLocalSmoothing(listH, o, bc):
    N = len(listH) + 1

    Virtual_Trajectory = [np.identity(3, dtype=float)]
    
    #Calculate virtual trajectory of transform matrix at each temporal
    for i in range(1, N):
        Virtual_Trajectory.append(Virtual_Trajectory[-1]+transforms[i-1]-np.identity(3,dtype=float))

    #Calculate smoothed trajectory at each temporal
    smoothed_trajectory = DCTbased_Guassian_Convolution(Virtual_Trajectory, transforms, o, bc)

    #Calculate H'
    smoothed_transforms = []
    for i in range(N):
        correction_matrix = smoothed_trajectory[i] - Virtual_Trajectory[i] + np.identity(3, dtype=float)
        smoothed_transforms.append(np.linalg.inv(correction_matrix))

    return smoothed_transforms
    
#Return smoothed trajectory based on boundary condition and sigma
def DCTbased_Guassian_Convolution(vir_trajectory, listM, o, bc):
    src = []

    N = len(vir_trajectory)

    for i in range(N-1):
        if bc == "CONSTANT_BC":
            src.append(np.identity(3,dtype=float))
        elif bc == "NEUMANN_BC":
            src.append(vir_trajectory[N-2-i])
        elif bc == "DIRICHLET_BC":
            src.append(2*np.identity(3,dtype=float) - vir_trajectory[N-2-i])

    for i in range(N-1, 2*N-1):
        src.append(vir_trajectory[i-N+1])

    for i in range(2*N-1, 3*N-2):
        if bc == "CONSTANT_BC":
            src.append(vir_trajectory[N-1])
        elif bc == "NEUMANN_BC":
            src.append(vir_trajectory[3*N-2-i])
        elif bc == "DIRICHLET_BC":
            src.append(2*vir_trajectory[N-1] - vir_trajectory[3*N-2-i])

    dest = []

    DCTbased_Guassian_Conv(dest, src, 3*N-3, o)

    M_return = []
    for i in range(N):
        M_return.append(dest[N+i-1])

    return M_return

#Algorithm for dct-based guassian filtering  
def DCTbased_Guassian_Conv(dest, src, k, o):
    N = k + 1
    
    Fk = []
    Uk = []
    
    for i in range(N):
        cos_sum = np.zeros((3,3), dtype=float)
        for j in range(N):
            cos_sum += 2*src[j]*math.cos(math.pi*(j+1/2)*i/N)
        Fk.append(cos_sum)

    for i in range(N):
        uk = Fk[i]*math.exp(-2*(math.pi**2)*(o**2)*((i/(2*N))**2))
        Uk.append(uk)
        print(uk)
    
    for i in range(N):
        un = Uk[0]/(2*N)
        for j in range(1,N):
            un += Uk[j]*math.cos(math.pi*(i+1/2)*j/N)/N    
        dest.append(un)

    
#Get transform matrix H' from stabilized frame to original frame
transforms_smoothed = AddictiveLocalSmoothing(transforms, sigma, boundary_constraint)

#return T transform that crop and zoom based on matrix H'
def CropZoom(nx, ny, list_M):
    x1 = 0;
    y1= 0;
    x2 = nx - 1;
    y2 = ny - 1;

    for M in list_M:
        top_left = np.linalg.inv(M)@np.array([[0,0,1]]).T
        if top_left[0][0]/top_left[2][0] > x1:
            x1 = top_left[0][0]/top_left[2][0]
        if top_left[1][0]/top_left[2][0] > y1:
            y1 = top_left[1][0]/top_left[2][0]

        top_right = np.linalg.inv(M)@np.array([[nx-1,0,1]]).T
        if top_right[0][0]/top_right[2][0] > x2:
            x2 = top_right[0][0]/top_right[2][0]
        if top_right[1][0]/top_right[2][0] > y1:
            y1 = top_right[1][0]/top_right[2][0]

        bottom_left = np.linalg.inv(M)@np.array([[0,ny-1,1]]).T
        if bottom_left[0][0]/bottom_left[2][0] > x1:
            x1 = bottom_left[0][0]/bottom_left[2][0]
        if bottom_left[1][0]/bottom_left[2][0] < y2:
            y2 = bottom_left[1][0]/bottom_left[2][0]

        bottom_right = np.linalg.inv(M)@np.array([[nx-1,ny-1,1]]).T
        if bottom_right[0][0]/bottom_right[2][0] < x2:
            x2 = bottom_right[0][0]/bottom_right[2][0]
        if bottom_right[1][0]/bottom_right[2][0] < y2:
            y2 = bottom_right[1][0]/bottom_right[2][0]

    s = min((x2-x1)/nx, (y2-y1)/ny)

    xm = (x1 + x2)/2
    ym = (y1 + y2)/2

    T = np.zeros((3,3), dtype=float)
    
    T[0][0] = s
    T[1][1] = s
    T[0][2] = xm - s*nx/2
    T[1][2] = ym - s*ny/2
    T[2][2] = 1

    return T

#Calculate T
T = CropZoom(w, h, transforms_smoothed)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(n_frames-1):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break


  H = transforms_smoothed[i]
  # Apply persective wrapping to the given frame
  frame_stabilized = cv2.warpPerspective(frame,np.linalg.inv(H), (w,h))

  # Write the frame to the file
  frame_out = cv2.hconcat([frame, frame_stabilized])

  # If the image is too big, resize it.
  if(frame_out.shape[1] > 1920): 
    frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
  
  cv2.imshow("Before and After", frame_out)
  cv2.waitKey(30)
  out.write(frame_stabilized)

cap.release()
out.release()
cv2.destroyAllWindows()