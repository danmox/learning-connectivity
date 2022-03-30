import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.stats import norm
from skimage.filters import gaussian
from skimage.filters.thresholding import threshold_local


def pos_to_subs(res, img_size, pts):
    """assumes center of image corresponds to (0,0) meters"""
    return np.floor(pts / res).astype(int) + int(img_size) / 2


def sub_to_pos(res, img_size, subs):
    """assumes center of image corresponds to (0,0) meters"""
    return (subs - img_size / 2.0) * res


def compute_paramaters(p):
    img_res = p["img_res"]
    p["img_size"] = (img_res, img_res)  # square images used with CNN

    # bounding box within which all agent positions must lie; this prevents
    # agents being placed near the edge of the image where their Gaussian
    # kernel would get cut off

    p["img_side_len"] = img_res * p["meters_per_pixel"]  # for convenience
    p["max_agent_bbx"] = (p["img_side_len"] / 2.0 - p["kernel_std"]) * np.asarray(
        [-1, 1, -1, 1]
    )
    p["bbx"] = p["img_side_len"] / 2.0 * np.asarray([-1, 1, -1, 1])
    p["coverage_range"] = 1.5 * p["kernel_std"]

    # the xy location of the center of each pixel
    # NOTE the metric location (0,0) is chosen to be the center of the image
    ij = np.stack(
        np.meshgrid(np.arange(img_res), np.arange(img_res), indexing="ij"), axis=2
    )
    p["xy"] = p["meters_per_pixel"] * (ij + 0.5) - p["img_side_len"] / 2.0
    return p


def compute_peaks(image, threshold_val=80, blur_sigma=1, region_size=7):
    # remove noise in image
    blurred_img = gaussian(image, sigma=blur_sigma)

    # only keep the max value in a local region
    thresh_fcn = lambda a: max(a.max(), 0.01)
    thresh_mask = threshold_local(
        blurred_img, region_size, method="generic", param=thresh_fcn
    )
    peaks = np.argwhere(blurred_img >= thresh_mask)

    # only pixels above a threshold value should be considered peaks
    # NOTE in rare cases no peaks with intensity greater than 80 are found in
    # the image, so walk back the threshold until one is found
    out_peaks = np.zeros((0, 2))
    while threshold_val > 0 and out_peaks.shape[0] == 0:
        out_peaks = peaks[image[peaks[:, 0], peaks[:, 1]] > threshold_val]
        threshold_val -= 10
    return out_peaks, blurred_img


def compute_voronoi(pts, bbx):

    # reflect points about all sides of bbx
    pts_l = np.copy(pts)
    pts_l[:, 0] = -pts_l[:, 0] + 2 * bbx[0]
    pts_r = np.copy(pts)
    pts_r[:, 0] = -pts_r[:, 0] + 2 * bbx[1]
    pts_d = np.copy(pts)
    pts_d[:, 1] = -pts_d[:, 1] + 2 * bbx[2]
    pts_u = np.copy(pts)
    pts_u[:, 1] = -pts_u[:, 1] + 2 * bbx[3]
    pts_all = np.vstack((pts, pts_l, pts_r, pts_d, pts_u))

    # Voronoi tesselation of all points, including the reflected ones: as a
    # result the Voronoi cells associated with pts are all closed
    vor = Voronoi(pts_all)

    # vor contains Voronoi information for pts_all but we only need the 1st
    # fifth that correspond to pts
    region_idcs = vor.point_region[: vor.point_region.shape[0] // 5].tolist()
    vertex_idcs = [vor.regions[i] for i in region_idcs]
    cell_polygons = [vor.vertices[idcs] for idcs in vertex_idcs]

    # generate triangulations of cells for use in integration
    hulls = []
    for poly in cell_polygons:
        hulls.append(Delaunay(poly))

    return hulls


def lloyd_step(image, xy, config, voronoi_cells, coverage_range):
    centroids = np.zeros((len(voronoi_cells), 2))
    for i, cell in enumerate(voronoi_cells):
        # assemble the position of the center and corresponding intensity
        # of every pixel that falls within the voronoi cell
        cell_pixel_mask = cell.find_simplex(xy) >= 0
        coverage_range_mask = np.linalg.norm(xy - config[i], axis=2) < coverage_range
        pixel_mask = cell_pixel_mask & coverage_range_mask
        cell_pixel_pos = xy[pixel_mask]
        cell_pixel_val = image[pixel_mask]
        cell_volume = np.sum(cell_pixel_val)
        
        # if there are intensity values >0 within the voronoi cell compute
        # the intensity weighted centroid of of the cell otherwise compute
        # the geometric centroid of the cell
        if cell_volume < 1e-8:
            centroids[i] = np.sum(cell_pixel_pos, axis=0) / cell_pixel_pos.shape[0]
        else:
            centroids[i] = (
                np.sum(cell_pixel_val[:, np.newaxis] * cell_pixel_pos, axis=0)
                / cell_volume
            )
    return centroids


def kernelized_config_img(config, params):
    """generate kernelized image from agents configuration

    inputs:
      config - Nx2 numpy array of agent positions
      params - image parameters from cnn_image_parameters()

    ourputs:
      img - image with node positions marked with a gaussian kernel
    """
    img = np.zeros(params["img_size"])
    for agent in config:
        dist = np.linalg.norm(params["xy"] - agent, axis=2)
        mask = dist < 3.0 * params["kernel_std"]
        img[mask] = np.maximum(
            img[mask], norm.pdf(dist[mask], scale=params["kernel_std"])
        )
    img *= 255.0 / norm.pdf(
        0, scale=params["kernel_std"]
    )  # normalize image to [0.0, 255.0]
    return np.clip(img, 0, 255)
