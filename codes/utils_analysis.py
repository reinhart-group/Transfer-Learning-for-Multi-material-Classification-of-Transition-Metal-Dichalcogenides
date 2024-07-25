import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.feature import canny
from skimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter, convolve
from sklearn.datasets import make_regression
from sklearn.feature_selection import r_regression
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage import feature # import blob_dog

from sklearn.linear_model import LinearRegression

import numpy as np
from skimage import io, filters, measure, morphology, color
import cv2


def obtain_rms_roughness(X, features):
    RMS_roughness = []
    for item in X:
        img = item.transpose(1, 2, 0)
        smoothed_image = gaussian_filter(img, sigma=1.0)

    # Calculate the root-mean-square (RMS) roughness
        rms_roughness = np.sqrt(mse(img, smoothed_image))
        RMS_roughness.append(rms_roughness)


    RMS_roughness = np.array(RMS_roughness)
    r_property = r_regression(features, RMS_roughness)
    
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_roughness_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, RMS_roughness


def obtain_step_edges(img):
    
    # Apply Canny edge detection
    edges = canny(img[:, :, 0], sigma=0.3)

    # Perform dilation and erosion to enhance step edges
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_edges = binary_dilation(edges, kernel)
    step_edges = binary_erosion(dilated_edges, kernel)
    #step_edge_coords = np.where(step_edges)
    labeled_steps, num_steps = label(step_edges)
    return step_edges, num_steps

def obtain_num_steps(X, features):
    
    Num_steps = []
    for item in X:
        img = item.transpose(1, 2, 0)
        step_edges, num_steps =obtain_step_edges(img)

    # Calculate the root-mean-square (RMS) roughness
        Num_steps.append(num_steps)

    Num_steps = np.array(Num_steps)
    image_property = Num_steps
    r_property = r_regression(features, image_property)
    
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_num_steps_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Num_steps

def quantify_defect_density(afm_image, min_distance=1):
    """
    Quantify the defect density and distribution from an AFM image.

    Args:
        afm_image (np.ndarray): The AFM image of the TMD material.
        min_distance (int): Minimum distance between detected defects.

    Returns:
        defect_density (float): The defect density (defects per unit area).
        defect_coordinates (np.ndarray): The (x, y) coordinates of the detected defects.
    """
    # Apply Gaussian smoothing to reduce noise
    smoothed_image = gaussian(afm_image, sigma=1.0, channel_axis=-1)

    # Detect local maxima as potential defect locations
    defect_coordinates = peak_local_max(smoothed_image, min_distance=min_distance)

    # Calculate the defect density
    defect_density = len(defect_coordinates) / np.prod(afm_image.shape)

    return defect_density, defect_coordinates


def obtain_defect_density(X, features):
    
    Defect_density = []
    for item in X:
        img = item.transpose(1, 2, 0)
        defect, defect_coord = quantify_defect_density(img)
        Defect_density.append(defect)


    Defect_density = np.array(Defect_density)
    print(Defect_density.shape)

    image_property =Defect_density

    r_property = r_regression(features, image_property)
    
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_defect_density_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Defect_density

def calculate_anisotropy(afm_image, kernel_size=5):
    """
    Quantify the anisotropic surface features of an AFM image.

    Args:
        afm_image (np.ndarray): The AFM image of the TMD material.
        kernel_size (int): Size of the Sobel kernel.

    Returns:
        anisotropy (float): The degree of anisotropy in the surface features.
    """
    # Calculate the gradients along the x and y directions
    #grad_x = convolve(afm_image, np.array([[-1, 0, 1]] * kernel_size), mode='reflect')
    #grad_y = convolve(afm_image, np.array([[-1], [0], [1]] * kernel_size), mode='reflect')
    
    grad_x, grad_y, grad_z = np.gradient(afm_image)

    # Calculate the gradient magnitudes
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Calculate the angle of the gradients
    grad_angle = np.arctan2(grad_y, grad_x)

    # Calculate the anisotropy index
    anisotropy = np.std(grad_angle) / np.mean(grad_magnitude)

    return anisotropy

def obtain_anisotropy(X, features):
    Anisotropy = []
    for item in X:
        img = item.transpose(1, 2, 0)
        ani = calculate_anisotropy(img)
        Anisotropy.append(ani)


    Anisotropy = np.array(Anisotropy)

    image_property =Anisotropy

    r_property = r_regression(features, image_property)
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_anisotropy_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Anisotropy

def quantify_local_variation(afm_image, sigma=1.0):
    """
    Quantify the local structural variations in an AFM image.

    Args:
        afm_image (np.ndarray): The AFM image of the TMD material.
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        local_variation (float): The degree of local structural variation.
    """
    # Apply Gaussian smoothing to remove noise
    smoothed_image = gaussian_filter(afm_image, sigma=sigma)

    # Calculate the difference between the original and smoothed images
    local_variation = np.abs(afm_image - smoothed_image)

    # Normalize the local variation
    local_variation /= np.max(local_variation)

    return np.mean(local_variation)

def obtain_local_variation(X, features):

    Local_variation = []
    for item in X:
        img = item.transpose(1, 2, 0)
        local_variation = quantify_local_variation(img, sigma=1.0)
        Local_variation.append(local_variation)




    Local_variation = np.array(Local_variation)
    image_property =Local_variation

    r_property = r_regression(features, image_property)
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_local_variation_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Local_variation

def obtain_num_blob_dog(X, features):
    
    Num_blob_dog = []
    for item in X:
        img = item.transpose(1, 2, 0)
        coordi = feature.blob_dog(img,threshold=0.001, min_sigma=5, max_sigma=40)#, *, threshold_rel=None)#, exclude_border=False)

        Num_blob_dog.append(len(coordi))


    image_property =Num_blob_dog

    r_property = r_regression(features, image_property)
    
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_num_blog_dog_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Num_blob_dog


def extract_glcm_features(afm_image, distances=[1], angles=[0], levels=None, props=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'], symmetric=True, normed=True):
    """
    Extract GLCM features from an AFM image.

    Args:
        afm_image (np.ndarray): The AFM image of the TMD material.
        distances (list): List of pixel distances for GLCM computation.
        angles (list): List of angles (in degrees) for GLCM computation.
        levels (int or None): Number of gray levels to quantize the image to.
        props (list): List of GLCM properties to compute.
        symmetric (bool): Whether to calculate symmetric GLCM.
        normed (bool): Whether to normalize the GLCM.

    Returns:
        features (dict): A dictionary of GLCM feature names and their values.
    """
    #print('shape', afm_image.shape)
    #print('max', afm_image.max())
    
    afm_image = afm_image.astype(int)
    #afm_image = np.clip(afm_image, 0, 255)
    
    features = {}

    if levels is None:
        levels = np.max(afm_image) - np.min(afm_image) + 1
        print(levels)

    glcm = graycomatrix(afm_image, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)

    for prop in props:
        feature_val = graycoprops(glcm, prop)
        if isinstance(feature_val, np.ndarray):
            feature_val = feature_val.mean()
        features[prop] = feature_val

    return features


def obtain_contrast(GLCM_results, features):
    contrast = GLCM_results['contrast']

    image_property =contrast

    r_property = r_regression(features, image_property)

    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_contrast_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    
    y = features[:, r_ordered[0]]

    return x, y, contrast

def obtain_dissimilarity(GLCM_results, features):
    dissimilarity = GLCM_results['dissimilarity']

    image_property =dissimilarity

    r_property = r_regression(features, image_property)

    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_dissimilarity_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]

    return x, y, dissimilarity

def obtain_homogeneity(GLCM_results, features):
    homogeneity = GLCM_results['homogeneity']

    image_property =homogeneity

    r_property = r_regression(features, image_property)

    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_homogeneity_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]

    return x, y, homogeneity

def obtain_energy(GLCM_results, features):
    energy = GLCM_results['energy']

    image_property =energy

    r_property = r_regression(features, image_property)

    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_energy_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]

    return x, y, energy

def obtain_correlation(GLCM_results, features):
    correlation = GLCM_results['correlation']

    image_property =correlation

    r_property = r_regression(features, image_property)

    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_correlation_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]

    return x, y, correlation

def obtain_ASM(GLCM_results, features):
    ASM = GLCM_results['ASM']

    image_property =ASM

    r_property = r_regression(features, image_property)

    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_ASM_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]

    return x, y, ASM



def analyze_afm_image(image_array):
    """
    Analyzes an AFM image numpy array to obtain grain and domain properties.
    
    Args:
        image_array (numpy.ndarray): A 2D NumPy array representing the AFM image.
        
    Returns:
        dict: A dictionary containing the following keys:
            'num_grains' (int): Number of grains in the image.
            'num_domains' (int): Number of domains in the image.
            'domain_sizes' (list): List of domain sizes (in pixels).
            'domain_density' (float): Density of domains (number of domains per unit area).
            'grain_density' (float): Density of grains (number of grains per unit area).
            'grain_sizes' (list): List of grain sizes (in pixels).
            'marked_image' (numpy.ndarray): The input image with marked domain boundaries and labels.
    """
    # Convert the image array to grayscale if it's not already
    #if image_array.ndim > 2:
    #    image = image_array.mean(axis=2)
    #else:
     #   image = image_array
    
    # Apply gaussian filter to reduce noise
    img = image_array.transpose(1, 2, 0)*255
    image = img[:, :, 0]
    image_filtered = filters.gaussian(image, sigma=1)
    
    # Binarize the image
    thresh = filters.threshold_otsu(image_filtered)
    binary = image_filtered > thresh
    
    # Label the grains
    label_grains = measure.label(binary)
    num_grains = np.max(label_grains)
    
    # Obtain the grain properties
    grain_props = measure.regionprops(label_grains)
    grain_areas = [prop.area for prop in grain_props]
    grain_sizes = [prop.equivalent_diameter for prop in grain_props]
    grain_density = num_grains / image.size
    
    # Label the domains
    selem = morphology.disk(5)
    binary_opened = morphology.opening(binary, selem)
    label_domains = measure.label(binary_opened)
    num_domains = np.max(label_domains)
    
    # Obtain the domain properties
    domain_props = measure.regionprops(label_domains)
    domain_sizes = [prop.area for prop in domain_props]
    domain_density = num_domains / image.size
    
    # Create a color image with marked domain boundaries and labels
    marked_image = color.label2rgb(label_domains, image=image, bg_label=0)
    for region in measure.regionprops(label_domains):
        minr, minc, maxr, maxc = region.bbox
        marked_image[minr:maxr, minc:minc+1] = [1, 0, 0]  # Left edge
        marked_image[minr:maxr, maxc:maxc+1] = [1, 0, 0]  # Right edge
        marked_image[minr:minr+1, minc:maxc] = [1, 0, 0]  # Top edge
        marked_image[maxr:maxr+1, minc:maxc] = [1, 0, 0]  # Bottom edge
        rx, ry = region.centroid
        label_text = f"{region.label}"
        marked_image = cv2.putText(marked_image, label_text, (int(rx), int(ry)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 1, cv2.LINE_AA)
    
    return {
        'num_grains': num_grains,
        'num_domains': num_domains,
        'avg_domain_size': np.mean(domain_sizes),
        'domain_density': domain_density,
        'grain_density': grain_density,
        'avg_grain_size': np.mean(grain_sizes),
        'marked_image': marked_image
    }


def obtain_grain_size(X, features):

    Grain_size = []
    for item in X:
        results = analyze_afm_image(item)
        grain_size = results['avg_grain_size']
        Grain_size.append(grain_size)




    Grain_size = np.array(Grain_size)
    image_property =Grain_size

    r_property = r_regression(features, image_property)
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_grain_size_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Grain_size

def obtain_grain_density(X, features):

    Grain_density = []
    for item in X:
        results = analyze_afm_image(item)
        grain_density = results['domain_density']
        Grain_density.append(grain_density)




    Grain_density = np.array(Grain_density)
    image_property =Grain_density

    r_property = r_regression(features, image_property)
    print('min r: ', min(r_property), 'max r: ', max(r_property))
    r_ordered = sorted(range(len(r_property)), key=lambda k: r_property[k])
    r_ordered = np.array(list(reversed(r_ordered)))
    print('Model_grain_density_index: ', r_ordered[-1], r_ordered[0])
    x = features[:, r_ordered[-1]]
    y = features[:, r_ordered[0]]
    
    return x, y, Grain_density
