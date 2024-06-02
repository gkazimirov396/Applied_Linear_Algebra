import numpy as np
import matplotlib.pyplot as plt

from typing import Literal

type Axis2D = Literal['x', 'y'];
type Axis3D = Axis2D | Literal['z'];

# 1

object1 = np.array([[0, 0], [0, 1], [1, 0], [0, 0]])  # Triangle
object2 = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]]) # Batman
object3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0]]) # 3D Object

def plot_object(coords: np.ndarray, title: str):
    plt.figure()
    plt.plot(coords[:, 0], coords[:, 1], 'o-') 
    plt.fill(coords[:, 0], coords[:, 1], alpha=0.3) 
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_objects(original_coords: np.ndarray, transformed_coords: np.ndarray, title: str):
    plt.figure()
    plt.plot(original_coords[:, 0], original_coords[:, 1], 'o-', label='Original')
    plt.plot(transformed_coords[:, 0], transformed_coords[:, 1], 'o-', label='Transformed') 
    plt.fill(original_coords[:, 0], original_coords[:, 1], alpha=0.3)
    plt.fill(transformed_coords[:, 0], transformed_coords[:, 1], alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_object(object1, 'Object 1 (Triangle)')
plot_object(object2, 'Object 2 (Batman)')

# 2

def rotate_object(coords: np.ndarray, angle_degrees: int, title: str):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    print("Rotation Matrix:\n", rotation_matrix)

    transformed_coords = np.dot(coords, rotation_matrix)
    plot_objects(coords, transformed_coords, f'Rotated {title} by {angle_degrees} degrees')

    return transformed_coords

rotated_object = rotate_object(object1, 45, 'Triangle')
rotated_object = rotate_object(object2, 45, 'Batman')


def scale_object(coords: np.ndarray, sx: int, sy: int, title: str):
    scaling_matrix = np.array([
        [sx, 0],
        [0, sy]
    ])

    print("Scaling Matrix:\n", scaling_matrix)

    transformed_coords = np.dot(coords, scaling_matrix)
    plot_objects(coords, transformed_coords, f'Scaled {title} by factors {sx} and {sy}')

    return transformed_coords

scaled_object = scale_object(object1, 2, 1, 'Triangle')
scaled_object = scale_object(object2, 2, 1, 'Batman')

def reflect_object(coords: np.ndarray, axis: Axis2D, title: str):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])

    print("Reflection Matrix:\n", reflection_matrix)

    transformed_coords = np.dot(coords, reflection_matrix)
    plot_objects(coords, transformed_coords, f'Reflected {title} across {axis.upper()}-axis')

    return transformed_coords

reflect_object(object1, 'y', 'Triangle')
reflect_object(object2, 'x', 'Batman')


def shear_object(coords: np.ndarray, k: float, axis: Axis2D):
    if axis == 'x':
        shear_matrix = np.array([
            [1, k],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [k, 1]
        ])
    
    print("Shear Matrix:\n", shear_matrix)

    transformed_coords = np.dot(coords, shear_matrix)
    plot_objects(coords, transformed_coords, f'Sheared Object along {axis.upper()}-axis')

    return transformed_coords

shear_object(object1, 0.5, 'x')
shear_object(object2, 0.8, 'y')

def custom_transform(coords: np.ndarray, transformation_matrix: np.ndarray, title: str):
    transformed_coords = np.dot(coords, transformation_matrix)

    plot_objects(coords, transformed_coords, f'Custom Transformed {title}')

    return transformed_coords


example_matrix = np.array([
    [1, -1],
    [1, 1]
])

custom_transform(object1, example_matrix, 'Triangle')
custom_transform(object2, example_matrix, 'Batman')

# 4

def plot_object_3d(coords: np.ndarray, title='3D Object'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'o-')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.title(title)
    plt.show()

def rotate_object_3d(coords: np.ndarray, angle_degrees: int, axis: Axis3D):
    angle_radians = np.radians(angle_degrees)

    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])

    transformed_coords = np.dot(coords, rotation_matrix)
    plot_object_3d(transformed_coords, f'3D Rotated Object around {axis.upper()} by {angle_degrees} degrees')

plot_object_3d(object3d, '3D Object')

rotate_object_3d(object3d, 45, 'z')