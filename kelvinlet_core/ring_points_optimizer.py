# import os
# import sys
# import random
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from kelvinlet_core import common

def linear(length, x):
    y = -x + length
    return y

def get_triangle_points(num_points_per_side, length):
    # 2D right triangle
    num_points = num_points_per_side + (num_points_per_side - 1) + (num_points_per_side - 2) + 1
    points = np.zeros((num_points, 2))
    
    # one leg aligned with x-axis
    points[:num_points_per_side, 0] = np.linspace(0, length, num_points_per_side)
    points[:num_points_per_side, 1] = np.zeros(num_points_per_side)
    
    # hypothenuse
    points[num_points_per_side:(2 * num_points_per_side - 1), 0] = np.copy(np.flip(points[:(num_points_per_side - 1), 0]))
    points[num_points_per_side:(2 * num_points_per_side - 1), 1] = linear(length, points[num_points_per_side:(2 * num_points_per_side - 1), 0])
    
    # one leg aligned with y-axis
    points[(2 * num_points_per_side - 1):-1, 0] = np.zeros(num_points_per_side - 2)
    points[(2 * num_points_per_side - 1):-1, 1] = np.copy(np.flip(points[num_points_per_side:(2 * num_points_per_side - 1 - 1), 1]))
    
    points[-1] = np.copy(points[0])
    
    return points

def get_circle_points(num_points, radius):
    points = np.zeros((num_points, 2))
    theta = np.linspace(0, 2 * np.pi, num_points)
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    
    points[-1] = np.copy(points[0])
    
    return points

def get_ellipse_points(num_points, radius_x, radius_y):
    points = np.zeros((num_points, 2))
    theta = np.linspace(0, 2 * np.pi, num_points)
    points[:, 0] = radius_x * np.cos(theta)
    points[:, 1] = radius_y * np.sin(theta)
    
    points[-1] = np.copy(points[0])
    
    return points

def get_noisy_circle_points(num_points, radius):
    points = np.zeros((num_points, 2))
    theta = np.linspace(0, 2 * np.pi, num_points)
    points[:, 0] = radius * np.cos(theta) + 0.2 * radius * np.random.rand(num_points)
    points[:, 1] = radius * np.sin(theta) + 0.2 * radius * np.random.rand(num_points)
    
    points[-1] = np.copy(points[0])
    
    return points

def get_noisy_ellipse_points(num_points, radius_x, radius_y):
    points = np.zeros((num_points, 2))
    theta = np.linspace(0, 2 * np.pi, num_points)
    points[:, 0] = radius_x * np.cos(theta) + 0.1 * radius_x * np.random.rand(num_points)
    points[:, 1] = radius_y * np.sin(theta) + 0.1 * radius_y * np.random.rand(num_points)
    
    points[-1] = np.copy(points[0])
    
    return points

def get_area(points):
    num_points = points.shape[0]
    assert(points.shape == (num_points, 2))
    # reference: https://stackoverflow.com/a/30408825
    area = 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
    return area

def objective_function(scale, original_points, prescribed_area):
    scaled_points = get_scaled_points(original_points, scale)
    objective = np.abs(get_area(scaled_points) - prescribed_area) # minimize this absolute difference
    return objective

def find_scale_to_get_prescribed_area(original_points, prescribed_area):
    print("note that I think this only works for convex-shapes (not non-convex shapes), but i think it is okay to assume that vessels usually have convex cross-sections")
    
    num_points = original_points.shape[0]
    num_spatial_dims = original_points.shape[1]
    assert(original_points.shape == (num_points, num_spatial_dims))
    np.testing.assert_equal(original_points[0, :], original_points[-1, :])
    # np.testing.assert_allclose(common.get_centroid(original_points), np.zeros((1, num_spatial_dims)), atol=1e-8)
    
    # reference: https://www.einblick.ai/python-code-examples/minimizing-function-scipy-optimize-minimize/
    
    result = opt.minimize(objective_function, x0 = prescribed_area / get_area(original_points), args = (original_points, prescribed_area), method = "Nelder-Mead")
    
    true_scale = (result.x)[0]
    
    # Print message indicating why the process terminated
    print("message = ", result.message)

    # Print the minimum value of the function
    # print("objective value = ", result.fun)

    # Print the x-value resulting in the minimum value
    print("true scale = ", true_scale)
    
    true_scaled_points = get_scaled_points(original_points, true_scale)
    
    print("percent difference = ", (get_area(true_scaled_points) - prescribed_area) / prescribed_area * 100)
    
    return true_scale

def get_scaled_points(points, scale):
    return points * scale

if __name__ == "__main__":
    ##########################################
    num_points_per_side = 11
    
    # shape = "triangle"
    # length = 1
    
    # shape = "circle"
    # radius = 1
    
    # shape = "ellipse"
    # radius = 1
    
    # shape = "noisy_circle"
    # radius = 1
    
    shape = "noisy_ellipse"
    radius = 1
    ##########################################
    
    if shape == "triangle":
        points = get_triangle_points(num_points_per_side, length)
    elif shape == "circle":
        points = get_circle_points(num_points_per_side * 2, radius)
    elif shape == "ellipse":
        points = get_ellipse_points(num_points_per_side * 2, radius, radius * 2)
    elif shape == "noisy_circle":
        points = get_noisy_circle_points(num_points_per_side * 2, radius)
    elif shape == "noisy_ellipse":
        points = get_noisy_ellipse_points(num_points_per_side * 2, radius, radius * 2)
    
    fig, axs = plt.subplots(1, 1)
    color = next(axs._get_lines.prop_cycler)['color']
    axs.plot(points[:, 0], points[:, 1], "-", color = color, label = "original")
    axs.plot(points[:, 0], points[:, 1], "o", color = color)
    
    num_points = points.shape[0]
    centroid = common.get_centroid(points)
    for ip in range(num_points):
        array = np.zeros((2, 2))
        array[0] = centroid
        array[1] = points[ip]
        plt.plot(array[:, 0], array[:, 1], "k-")
    axs.plot(centroid[:, 0], centroid[:, 1], "r*")
    true_centroid = np.broadcast_to(centroid, (num_points, centroid.shape[1]))
    
    for desired_area_percent_scale in [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2.5, 1]:
        print("------------------------------")
        prescribed_area = get_area(points) * desired_area_percent_scale / 100
        true_scale = find_scale_to_get_prescribed_area(points, prescribed_area)
        true_scaled_points = get_scaled_points(points, true_scale)
        center = common.get_centroid(true_scaled_points)
        center = np.broadcast_to(center, (num_points, center.shape[1]))
        true_scaled_points += true_centroid - center
        print("desired_area_percent_scale = ", desired_area_percent_scale)
        print("true area percent scale    = ", get_area(true_scaled_points) / get_area(points) * 100)
        color = next(axs._get_lines.prop_cycler)['color'] # https://stackoverflow.com/a/28779637
        axs.plot(true_scaled_points[:, 0], true_scaled_points[:, 1], "-", color = color, label = str(desired_area_percent_scale))
        axs.plot(true_scaled_points[:, 0], true_scaled_points[:, 1], "*", color = color)
    
    axs.legend()
    axs.axis('equal')
    fig.tight_layout()
    plt.show()