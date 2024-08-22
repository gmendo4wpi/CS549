import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def compute_gradients(img):
    # compute gradients along x and y axes
    dx = np.diff(img, axis=1)
    dy = np.diff(img, axis=0)
    
    # pad missing values to maintain array dimensions
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')
    
    # calculate gradient magnitude and orientation
    mag = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360  # convert to degrees and normalize
    
    return mag, orientation

def generate_gaussian_pyramid(img, num_octaves, s):
    pyramids = []
    k = 2**(1/s)
    initial_sigma = 1.6  # initial sigma based on Lowe's suggestion
    for o in range(num_octaves):
        octave = [img]
        sigma_previous = initial_sigma
        for i in range(1, s+3):
            sigma_desired = initial_sigma * k**i
            sigma_to_apply = np.sqrt(sigma_desired**2 - sigma_previous**2)
            print(f"Octave {o}, Level {i}, Sigma Desired: {sigma_desired}, Sigma to Apply: {sigma_to_apply}")
            img = gaussian_filter(img, sigma=sigma_to_apply)
            octave.append(img)
            sigma_previous = sigma_desired
        pyramids.append(octave)
        img = octave[-3][::2, ::2]  # downsample image for next octave
    return pyramids

def generate_DoG_pyramid(gaussian_pyramid):
    DoG_pyramid = []
    for octave in gaussian_pyramid:
        DoG_octave = [octave[i] - octave[i-1] for i in range(1, len(octave))]
        DoG_pyramid.append(DoG_octave)
    return DoG_pyramid

def display_images(images, title):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, n, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title + " " + str(i))
        plt.axis('off')
    plt.show()

def refine_keypoint_location(octave, s, x, y):
    # compute first derivatives
    dx = (octave[s][x, y+1] - octave[s][x, y-1]) / 2.0
    dy = (octave[s][x+1, y] - octave[s][x-1, y]) / 2.0
    ds = (octave[s+1][x, y] - octave[s-1][x, y]) / 2.0

    # compute second derivatives
    dxx = octave[s][x, y+1] - 2 * octave[s][x, y] + octave[s][x, y-1]
    dyy = octave[s][x+1, y] - 2 * octave[s][x, y] + octave[s][x-1, y]
    dss = octave[s+1][x, y] - 2 * octave[s][x, y] + octave[s-1][x, y]
    dxy = ((octave[s][x+1, y+1] - octave[s][x+1, y-1]) - (octave[s][x-1, y+1] - octave[s][x-1, y-1])) / 4.0
    dxs = ((octave[s+1][x, y+1] - octave[s+1][x, y-1]) - (octave[s-1][x, y+1] - octave[s-1][x, y-1])) / 4.0
    dys = ((octave[s+1][x+1, y] - octave[s+1][x-1, y]) - (octave[s-1][x+1, y] - octave[s-1][x-1, y])) / 4.0

    # create Jacobian and Hessian for subpixel refinement
    J = np.array([dx, dy, ds])
    H = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]
    ])

    # solve for offset using inverse Hessian
    try:
        inv_H = np.linalg.inv(H)
        offset = -np.dot(inv_H, J)
        return offset, True
    except np.linalg.LinAlgError:
        return None, False

def find_scale_space_extrema(DoG_pyramid, r=10.0, contrast_threshold=0.00001):
    keypoints = []
    potential_keypoints = 0
    filtered_by_edge = 0
    for o, octave in enumerate(DoG_pyramid):
        for s in range(1, len(octave)-1):
            for x in range(1, octave[s].shape[0]-1):
                for y in range(1, octave[s].shape[1]-1):
                    if is_extremum(octave, s, x, y, contrast_threshold, r):
                        potential_keypoints += 1
                        offset, success = refine_keypoint_location(octave, s, x, y)
                        if success and np.all(np.abs(offset) < 0.5):  # apply offset if within subpixel range
                            new_x, new_y, new_s = x + offset[0], y + offset[1], s + offset[2]
                            keypoints.append((o, new_s, new_x, new_y))
                        else:
                            filtered_by_edge += 1
    print(f"Total keypoints found: {len(keypoints)}, Potential keypoints: {potential_keypoints}, Filtered by edge: {filtered_by_edge}")
    return keypoints

def assign_orientation(keypoints, gaussian_pyramid):
    num_bins = 36
    window_width = 1.5  # scale factor for Gaussian window used in orientation assignment

    keypoint_orientations = []

    for o, s, x, y in keypoints:
        o = int(o)
        s = int(s)
        sigma = 1.6 * 2 ** (o + s / 3)  # compute sigma based on octave and scale
        radius = int(round(3 * window_width * sigma))
        weight_kernel = gaussian_filter(np.zeros((2 * radius + 1, 2 * radius + 1)), sigma=sigma)

        if s < len(gaussian_pyramid[o]):
            mag, orientation = compute_gradients(gaussian_pyramid[o][s])

        histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nx, ny = int(x + i), int(y + j)
                if 0 <= nx < mag.shape[0] and 0 <= ny < mag.shape[1]:
                    weight = mag[nx, ny] * weight_kernel[radius + i, radius + j]
                    bin = int(num_bins * orientation[nx, ny] // 360)
                    histogram[bin] += weight

        peak_val = np.max(histogram)
        peaks = np.where(histogram >= 0.8 * peak_val)[0]

        for peak in peaks:
            peak_angle = (360. / num_bins) * peak
            keypoint_orientations.append((o, s, x, y, peak_angle))

    return keypoint_orientations

def is_extremum(octave, s, x, y, contrast_threshold, r):
    patch = np.array([
        octave[s-1][x-1:x+2, y-1:y+2].flatten(),
        octave[s][x-1:x+2, y-1:y+2].flatten(),
        octave[s+1][x-1:x+2, y-1:y+2].flatten()
    ]).flatten()
    center_value = octave[s][x, y]
    max_value = np.max(patch)
    min_value = np.min(patch)

    if np.abs(center_value) < contrast_threshold:
       # print(f"Rejected by contrast threshold at ({x},{y}) in scale {s} with value {center_value}")
        return False
    if center_value == max_value or center_value == min_value:
        return True
    return False


def is_edge_like(slice, x, y, r):
    dxx = slice[x+1, y] + slice[x-1, y] - 2 * slice[x, y]
    dyy = slice[x, y+1] + slice[x, y-1] - 2 * slice[x, y]
    dxy = (slice[x+1, y+1] + slice[x-1, y-1] - slice[x+1, y-1] - slice[x-1, y+1]) / 4.0
    tr = dxx + dyy
    det = dxx * dyy - dxy**2

    if det <= 0:
        print("Rejected by edge response: non-positive determinant")
        return True

    curvature_ratio = (tr**2) / det
    threshold = ((r + 1)**2) / r
    if curvature_ratio > threshold:
        print("Rejected by edge response: curvature ratio")
        return True
    return False

def create_descriptor(keypoint, gaussian_pyramid, width=4, num_bins=8):
    o, s, x, y, orientation = keypoint
    img = gaussian_pyramid[o][s]
    descriptor = np.zeros((width * width, num_bins))
    
    # Rotate coordinates according to keypoint orientation to achieve rotation invariance
    angle = np.deg2rad(orientation)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    hist_width = 3 * 1.5 * 1.6
    subregion_width = hist_width / width
    bin_width = 360 / num_bins
    
    for i in range(-width // 2, width // 2):
        for j in range(-width // 2, width // 2):
            nx = int(x + (i * subregion_width * cos_angle - j * subregion_width * sin_angle))
            ny = int(y + (i * subregion_width * sin_angle + j * subregion_width * cos_angle))
            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                mag, ori = compute_gradients(img[nx, ny])
                weight = gaussian_filter(img, sigma=1.5)[nx, ny]
                bin = int((ori - orientation + 360) % 360 / bin_width)
                descriptor[i + width // 2, j + width // 2, bin] += weight * mag
    
    descriptor = descriptor.flatten()
    # Normalize descriptor to unit length for illumination invariance
    descriptor /= np.linalg.norm(descriptor, ord=2) + 1e-10
    return descriptor

# Load and preprocess image
lenna_path = r'C:\Users\Gio\Downloads\lenna.png'  # path to image file
img = plt.imread(lenna_path)
if img.ndim == 3:
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # convert to grayscale using luminosity method

img = img.astype(np.float32) / 255.0  # normalize pixel values

plt.imshow(img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

num_octaves = 2  # number of octaves in Gaussian pyramid
s = 1 # intervals per octave

gaussian_pyramid = generate_gaussian_pyramid(img, num_octaves, s)
DoG_pyramid = generate_DoG_pyramid(gaussian_pyramid)

display_images(gaussian_pyramid[0], "Gaussian Octave 0")
display_images(DoG_pyramid[0], "DoG Octave 0")

keypoints = find_scale_space_extrema(DoG_pyramid, r=10, contrast_threshold=0.00022)
keypoints_with_orientation = assign_orientation(keypoints, gaussian_pyramid)
print(f"Number of keypoints detected: {len(keypoints)}")

fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
for keypoint in keypoints_with_orientation:
    octave, scale, x, y, orientation = keypoint
    ax.plot(y, x, 'r+')
plt.show()
