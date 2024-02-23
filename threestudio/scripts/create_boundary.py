import numpy as np
import os
from typing import Tuple


def create_rectangle_boundary(resolution: int,
                              boundary_width: float,
                              boundary_height: float,
                              boundary_thickness: float,
                              center: Tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    assert resolution > 1
    assert boundary_width > 0 and boundary_height > 0 , \
        "Boundary width, height, and thickness must be greater than 0"
    assert boundary_height < 1 and boundary_width < 1, \
        "Boundary width, height, and thickness must be less than 1"

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32) 
    boundary_array[int(resolution//2 - boundary_width * resolution//2):int(resolution// 2 + boundary_width * resolution//2),  
                   int(resolution//2 - boundary_height * resolution//2):int(resolution//2 + boundary_height * resolution//2), 0] = -boundary_thickness
    boundary_array[int(resolution//2 - boundary_width * resolution//2):int(resolution// 2 + boundary_width * resolution//2),  
                   int(resolution//2 - boundary_height * resolution//2):int(resolution//2 + boundary_height * resolution//2), 1] = boundary_thickness
    

    #shift the center
    if center:
        center_shift = (resolution//2*center[0], resolution//2*center[1])
        boundary_array = np.roll(boundary_array, center_shift, axis=(0,1))
        boundary_array = boundary_array + center[2]

    
    return boundary_array

def create_cylinder_boundary(resolution,
                             boundary_radius,
                             boundary_thickness):
    assert resolution > 1
    assert boundary_radius > 0 and boundary_thickness > 0, \
        "Boundary radius and thickness must be greater than 0"
    assert boundary_radius < 1, \
        "Boundary radius and thickness must be less than 0.5"

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32)
    x, y = np.ogrid[:resolution, :resolution]
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    boundary_array[mask, 0] = -boundary_thickness
    boundary_array[mask, 1] = boundary_thickness

    return boundary_array

def create_sphere_boundary(resolution, boundary_radius):
    assert resolution > 1
    assert boundary_radius > 0, "Boundary radius must be greater than 0"
    assert boundary_radius < 1, "Boundary radius must be less than 1"

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32)
    x, y, = np.ogrid[:resolution, :resolution]
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    h = (boundary_radius * resolution//2) ** 2 - (X[mask] - resolution / 2) ** 2 - (Y[mask] - resolution / 2) ** 2
    z_1 = -np.sqrt(h)
    z_2 = np.sqrt(h)
    for single_z1, single_z2 in zip(z_1, z_2):
        assert single_z1 <= single_z2, "z_1 must be less than z_2"
    # ensure 
    normalized_z_1 = z_1 / resolution
    normalized_z_2 = z_2 / resolution
    boundary_array[mask, 0] = normalized_z_1
    boundary_array[mask, 1] = normalized_z_2

    return boundary_array


def create_cone_boundary(resolution, boundary_radius, boundary_height):
    assert resolution > 1
    assert boundary_radius > 0 and boundary_height > 0, \
        "Boundary radius and height must be greater than 0"
    assert boundary_radius < 1 and boundary_height < 1, \
        "Boundary radius and height must be less than 1"

    boundary_array = np.zeros((resolution, resolution, 2), dtype=np.float32)
    x, y = np.ogrid[:resolution, :resolution]
    mask = (x - resolution / 2) ** 2 + (y - resolution / 2) ** 2 <= (boundary_radius * resolution//2) ** 2
    X, Y = np.meshgrid(np.arange(resolution), np.arange(resolution))
    h = (boundary_radius*resolution/2  - np.sqrt((X[mask] - resolution / 2) ** 2 + (Y[mask] - resolution / 2) ** 2)) * \
        boundary_height/resolution/boundary_radius*2

    top = h - boundary_height / 2
    bottom = -boundary_height/ 2
    boundary_array[mask, 0] = bottom
    boundary_array[mask, 1] = top

    return boundary_array


def combine_boundary(


if __name__ == "__main__":
    save_dir = "bounds"
    resolution = 1024
    boundary_width = 0.4
    boundary_height = 0.3
    boundary_thickness = 0.5
    rectangle_boundary = create_rectangle_boundary(resolution, boundary_width, boundary_height, boundary_thickness)
    print(rectangle_boundary.shape)
    np.save(os.path.join(save_dir, "rectangle_boundary.npy"), rectangle_boundary)

    boundary_radius = 0.4
    cylinder_boundary = create_cylinder_boundary(resolution, boundary_radius, boundary_thickness)
    print(cylinder_boundary.shape)
    np.save(os.path.join(save_dir, "cylinder_boundary.npy"), cylinder_boundary)

    sphere_boundary = create_sphere_boundary(resolution, boundary_radius)
    print(sphere_boundary.shape)
    np.save(os.path.join(save_dir, "sphere_boundary.npy"), sphere_boundary)

    boundary_height = 0.8
    boundary_radius = 0.7
    cone_boundary = create_cone_boundary(resolution, boundary_radius, boundary_height)
    print(cone_boundary.shape)
    np.save(os.path.join(save_dir, "cone_boundary.npy"), cone_boundary)


    