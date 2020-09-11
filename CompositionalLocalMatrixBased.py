# Import numpy and OpenCV
import numpy as np
import cv2
import math as math

# Read input video
cap = cv2.VideoCapture("video_dancing.mp4") 

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

feature_params = dict( maxCorners = 50,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )

lk_params = dict( winSize  = (15,15),
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

  # Sanity check
  assert prev_pts.shape == curr_pts.shape 

  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]

  #Find transformation matrix
  m, _ = cv2.estimateAffine2D(prev_pts,curr_pts)

  m = np.concatenate((m, np.array([[0,0,1]])), axis = 0 )
   
   
  # Store transformation
  transforms.append(m)
   
  # Move to next frame
  prev_gray = curr_gray


#Local Matrix smoothing
def localMatrixbasedSmoothing(ListH, o, bc):
    radius = 3*o
    N = len(ListH)+1
    if radius > N:
        radius = N

    Smoothed_H = []
    
    for i in range(N):
        t1 = i - radius
        t2 = i + radius

        if t1 < 0:
            t1 = 0
        
        if t2 > N-1:
            t2 = N-1

        
        indexToH = {}

        indexToH[i] = np.identity(3)

        #Backward transformation 
        if  i > 0:
            for j in range(i-1,t1-1,-1):
                indexToH[j] = np.linalg.inv(ListH[j])@indexToH[j+1]

        #Forward transformation
        if i < N-1:
            for j in range(i + 1,t2+1):
                indexToH[j] = ListH[j-1]@indexToH[j-1]

        h_Smoothed = Guassian_convolution(indexToH,t1,t2, i, o, bc)
        Smoothed_H.append(h_Smoothed)
    
    return Smoothed_H

def Guassian_convolution(H_dict,Nmin, Nmax, index, o, bc):
    radius = 3*o
    if radius > len(H_dict):
        radius = len(H_dict)

    average = np.zeros((3,3), dtype=float)
    sum_h = np.zeros((3,3),dtype=float)
    
    for i in range(index-radius, index + radius + 1):
        value =  np.zeros((3,3), dtype=float)
        if i < Nmin:
            if bc == "CONSTANT_BC":
                value = np.identity(3,dtype=float)
            elif bc == "NEUMANN_BC":
                value = H_dict[Nmin+(Nmin-i)]
            elif bc == "DIRICHLET_BC":
                value == 2*np.identity(3,dtype=float) - H_dict[Nmin+(Nmin-i)]
        elif i > Nmax:
            if bc == "CONSTANT_BC":
                value = H_dict[Nmax]
            elif bc == "NEUMANN_BC":
                value = H_dict[2*Nmax-i]
            elif bc == "DIRICHLET_BC":
                value == 2*H_dict[Nmax] - H_dict[2*Nmax - i]
        else:
            value = H_dict[i]
        
        gauss = math.exp(-(((i - index)**2)/(2*(o**2))))*np.ones((3,3), dtype=float)
        average = average + np.multiply(gauss,value)
        sum_h = sum_h + gauss

    return np.linalg.inv(average/sum_h)

transforms_smoothed = localMatrixbasedSmoothing(transforms, 30, "DIRICHLET_BC")
# import matplotlib.pyplot as plt
# x = []
# for H in transforms_smoothed:
#    x.append(H[0,2])
# plt.plot(range(len(x)), x, '-r')
# x_old = []
# for H in transforms:
#    x_old .append(H[0,2])
# plt.plot(range(len(x_old)), x_old, '--b')
# plt.show()
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

T = CropZoom(w, h, transforms_smoothed)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(n_frames-1):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break


  H = transforms_smoothed[i]
  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpPerspective(frame,np.linalg.inv(H@T), (w,h))

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