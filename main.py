import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
 lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


left_image = cv2.imread('/home/exception/PycharmProjects/pythonProject6/Dataset1/im0.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('/home/exception/PycharmProjects/pythonProject6/Dataset1/im1.png', cv2.IMREAD_GRAYSCALE)

K_left = np.array([[5299.313, 0, 1263.818],
                   [0, 5299.313, 977.763],
                   [0, 0, 1]])

K_right = np.array([[5299.313, 0, 1438.004],
                    [0, 5299.313, 977.763],
                    [0, 0, 1]])

# Step 1: Camera Calibration
# feature detection using sift
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(left_image, None)
kp2, des2 = sift.detectAndCompute(right_image, None)

# feature matching using flann
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []

# ratio test
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# find fundamental matrix
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
print(F)

# find essential matrix
E = np.dot(np.dot(K_right.T, F), K_left)
print(E)

# find rotation and translation matrix
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K_left)
print(R)
print(t)

# Step 2: Rectification
# - Calculate the rectification transformation matrices (homography matrices) for both left and right images
_, H_left, H_right = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F,
                                                   (left_image.shape[1], left_image.shape[0]))
print(H_left)
print(H_right)

# - Rectify the images to ensure epipolar lines are horizontal
rectified_left = cv2.warpPerspective(left_image, H_left, (left_image.shape[1], left_image.shape[0]))
rectified_right = cv2.warpPerspective(right_image, H_right, (right_image.shape[1], right_image.shape[0]))


# show epipolar lines on both images along with feature points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]
lines_left = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines_left = lines_left.reshape(-1, 3)
left_with_epipolar, left_with_points = draw_epipolar_lines(left_image, right_image, lines_left, pts1, pts2)

lines_right = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines_right = lines_right.reshape(-1, 3)
right_with_epipolar, right_with_points = draw_epipolar_lines(right_image, left_image, lines_right, pts2, pts1)

plt.subplot(121), plt.imshow(left_with_epipolar)
plt.subplot(122), plt.imshow(right_with_epipolar)
plt.show()


# step 3: Correspondence
# - Calculate the disparity map
win_size = 3
min_disp = -240
max_disp = 240
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=19,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=1,
    disp12MaxDiff=1,
    P1=8 * 2 * win_size ** 2,
    P2=32 * 2 * win_size ** 2,
)
disparity = stereo.compute(rectified_left, rectified_right)
disparity = (disparity/16.0 - min_disp)/num_disp

# Step 4: Normalize and Save Disparity Map
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity_8bit = np.uint8(disparity_normalized)

cv2.imwrite('disparity_grayscale.png', disparity_8bit)

# Apply a color map (heat map) to the disparity map for visualization
disparity_colormap = cv2.applyColorMap(disparity_8bit, cv2.COLORMAP_JET)
cv2.imwrite('disparity_colormap.png', disparity_colormap)

# Display the rectified images, disparity map, and depth map using OpenCV
plt.subplot(221), plt.imshow(left_image, cmap='gray')
plt.title('Left Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(right_image, cmap='gray')
plt.title('Right Image'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(disparity_8bit, cmap='gray')
plt.title('Disparity Map'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(disparity_colormap)
plt.title('Disparity Map with Color'), plt.xticks([]), plt.yticks([])
plt.show()

# compute depth map
baseline = 0.177  # Adjust the baseline value as needed (in meters)
focal_length = 5299.313  # Adjust the focal length value as needed

# Compute the depth map using the disparity map
depth_map = (baseline * focal_length) / (disparity_normalized + 1e-6)  # Add a small value to avoid division by zero
print(depth_map)
# Normalize the depth map to a suitable range for visualization
depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_8bit = np.uint8(depth_normalized)

# Save the depth map as a grayscale image
cv2.imwrite('depth_grayscale.png', depth_8bit)

# Apply a color map (heat map) to the depth map for visualization
depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

# Save the depth map as a color image with a heat map
cv2.imwrite('depth_colormap.png', depth_colormap)

# Display the depth map (grayscale and color)
plt.subplot(121), plt.imshow(depth_8bit, cmap='gray')
plt.title('Depth Map'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(depth_colormap)
plt.title('Depth Map with Color'), plt.xticks([]), plt.yticks([])
plt.show()
