import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import localthickness as lt
from skimage.measure import regionprops
import cc3d
from skimage.measure import euler_number
import vedo
import tifffile
from vedo import *
import time
import pandas as pd

os.system('cls')

start_time = time.time()

## LOCAL THICKNESS ##

def apply_local_thickness(B,voxel):

 
   array_thickness= lt.local_thickness(B,scale=0.5)*2
   array_separation= lt.local_thickness(~B,scale=0.5)*2
   
   thickness= np.mean(array_thickness)*voxel
   separation= np.mean(array_separation)*voxel
   std_thickness=np.std(array_thickness) 
   std_separation=np.std(array_separation) 


   return thickness, separation,std_thickness, std_separation

def remove_black_slices(slide_stack):
    
    slice_count = slide_stack.shape[0]
    total_pixels = slide_stack.shape[1] * slide_stack.shape[2]

    # Calculating the number of black pixels for each slice
    black_pixels_per_slice = np.sum(slide_stack == 0, axis=(1, 2))

    # Calculating the percentage of black pixels for each slice
    black_percentage_per_slice = black_pixels_per_slice / total_pixels * 100

    # Indices of slices to be removed
    indices_to_remove = np.where(black_percentage_per_slice > 80)[0]

    # Removing slices
    remaining_slide_stack = np.delete(slide_stack, indices_to_remove, axis=0)

    print(f"{len(indices_to_remove)} slices have been removed.")

    return remaining_slide_stack

## PURIFY ##

def purify_stack(stack):
    purified_image=[]
    stacked_purify_3d = np.zeros_like(stack)

    initial_info = extract_initial_info(stack)
    
    ## Foreground phase
    labeled_particles = cc3d.connected_components(stack, connectivity=26)
    reduced_image = reduce_particles_to_largest(labeled_particles)

    for i in range(stack.shape[0]):

        labeled_background_particles = cc3d.connected_components(reduced_image[i], connectivity=8)
        largest_background_particle_label = find_largest_background_particle(labeled_background_particles)
        relabel_border_particles(labeled_particles, largest_background_particle_label, initial_info['width'], initial_info['depth'], initial_info['slice'])
        image_without_small_background_particles = remove_small_background_particles(reduced_image, labeled_background_particles)
        purified_image.append(image_without_small_background_particles)
        stacked_purify_3d = np.stack(purified_image, axis=0)

    return stacked_purify_3d

def relabel_particles_across_slices(current_slice_labels, previous_slice_labels):
    unique_labels = np.unique(current_slice_labels)
    
    for label_val in unique_labels:
        if label_val != 0:
            corresponding_label = find_corresponding_label(label_val, current_slice_labels, previous_slice_labels)
            current_slice_labels[current_slice_labels == label_val] = corresponding_label

def find_corresponding_label(current_label, current_slice_labels, previous_slice_labels):
    current_particle_coords = np.argwhere(current_slice_labels == current_label)
    corresponding_labels = previous_slice_labels[tuple(current_particle_coords.T)]

    unique_corresponding_labels = np.unique(corresponding_labels[corresponding_labels > 0])

    if unique_corresponding_labels.size == 0:
        return 0

    return np.min(unique_corresponding_labels)

def extract_initial_info(imp):
    return {'width': imp.shape[2], 'depth': imp.shape[1], 'slice': imp.shape[0]}

def connected_component_labeling(image, connectivity):
    return cc3d.connected_components(image, connectivity=26)

def reduce_particles_to_largest(particle_labels):
    properties = regionprops(particle_labels)
    largest_particle_label = max(properties, key=lambda prop: prop.area).label
    reduced_image = np.where(particle_labels == largest_particle_label, 1, 0)
    return reduced_image

def connected_component_labeling_background(image, connectivity):
    return cc3d.connected_components(image, connectivity=8)

def find_largest_background_particle(background_particle_labels):
    properties = regionprops(background_particle_labels)
    largest_background_particle_label = max(properties, key=lambda prop: prop.area).label
    return largest_background_particle_label

def relabel_border_particles(particle_labels, largest_background_particle_label, w, h, d):
    borders = [(0, slice(None), slice(None)), 
               (slice(None), 0, slice(None)), 
               (slice(None), slice(None), 0), 
               (-1, slice(None), slice(None)), 
               (slice(None), -1, slice(None)), 
               (slice(None), slice(None), -1)]
    
    for border_slice in borders:
        for label_val in np.unique(particle_labels[border_slice]):
            if label_val != 0:
                particle_labels[particle_labels == label_val] = largest_background_particle_label

def remove_small_background_particles(imp, background_particle_labels):
    properties = regionprops(background_particle_labels)
    min_area_threshold = 100  # Adjust as needed
    for prop in properties:
        if prop.area < min_area_threshold:
            background_particle_labels[background_particle_labels == prop.label] = 0
    image_without_small_background_particles = np.where(background_particle_labels > 0, 1, 0)
    return image_without_small_background_particles

## CONNECTIVITY ##

def connect(stacked_purify_3d, volume):
    stack_rior = np.transpose(stacked_purify_3d, (1, 2, 0))
    sum_euler=euler_number(stack_rior)
    delta_chi = get_delta_chi(stacked_purify_3d, sum_euler)
    connectivity = get_connectivity(delta_chi)
    conn_density = get_conn_density(stacked_purify_3d, connectivity,volume)
    
    return sum_euler, delta_chi, connectivity, conn_density

def get_conn_density(stack, connectivity, volume):
    slice, height, width = stack.shape
    cal = 0.006  
    conn_density = connectivity / volume
    return conn_density


def get_delta_chi(stack, sum_euler):
    c= correct_for_edges(stack)
    delta_chi = sum_euler #- c
    return delta_chi

def get_connectivity(delta_chi):
    connectivity = 1 - delta_chi
    return connectivity

def get_dimension(stack):
    slice, height, width = stack.shape
    return slice, height, width

def get_pixel(stack, x, y, z):
    slice, height, width= get_dimension(stack)

    if 0 <= x < width and 0 <= y < height and 0 <= z < slice:
        #print(x, y, z, stack[z, y, x])

        return stack[z, y, x]

    return 0

def get_stack_vertices(stack):
    n_stack_vertices = 0
    slice, height, width= get_dimension(stack)
    x_inc = max(1, width - 1)
    y_inc = max(1, height - 1)
    z_inc = max(1, slice - 1)

    for z in range(0, slice, z_inc):
        for y in range(0, height, y_inc):
            for x in range(0, width, x_inc):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_vertices += 1

    return n_stack_vertices

def get_stack_edges(stack):
    slice, height, width= get_dimension(stack)

    n_stack_edges = 0
    
    w1 = width - 1
    h1 = height - 1
    slice1 = slice - 1
    
    x_inc = max(1, w1)
    y_inc = max(1, h1)
    z_inc = max(1, slice1)
    
    # left to right stack edges
    for z in range(0, slice, z_inc):
        for y in range(0, height, y_inc):
            for x in range(1, w1):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_edges += 1

    # back to front stack edges
    for z in range(0, slice, z_inc):
        for x in range(0, width, x_inc):
            for y in range(1, h1):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_edges += 1

    # top to bottom stack edges
    for y in range(0, height, y_inc):
        for x in range(0, width, x_inc):
            for z in range(1, slice1):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_edges += 1

    return n_stack_edges

def get_stack_faces(stack):
    slice, height, width= get_dimension(stack)

    w1 = width - 1
    h1 = height - 1
    slice1 = slice - 1
    
    x_inc = max(1, w1)
    y_inc = max(1, h1)
    z_inc = max(1, slice1)
    n_stack_faces = 0

    # top and bottom faces
    for z in range(0, slice, z_inc):
        for y in range(1, h1):
            for x in range(1, w1):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_faces += 1

    # back and front faces
    for y in range(0, height, y_inc):
        for z in range(1, slice1):
            for x in range(1, w1):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_faces += 1

    # left and right faces
    for x in range(0, width, x_inc):
        for y in range(1, h1):
            for z in range(1, slice1):
                if get_pixel(stack, x, y, z) == 1:
                    n_stack_faces += 1

    return n_stack_faces

def get_edge_vertices(stack):
    slice, height, width= get_dimension(stack)

    x_inc = max(1, width - 1)
    y_inc = max(1, height - 1)
    z_inc = max(1, slice - 1)
    n_edge_vertices = 0

    # left->right edges
    for z in range(0, slice, z_inc):
        for y in range(0, height, y_inc):
            for x in range(1, width):
                if get_pixel(stack, x, y, z) == 1:
                    n_edge_vertices += 1
                elif get_pixel(stack, x - 1, y, z) == 1:
                    n_edge_vertices += 1

    # back->front edges
    for z in range(0, slice, z_inc):
        for x in range(0, width, x_inc):
            for y in range(1, height):
                if get_pixel(stack, x, y, z) == 1:
                    n_edge_vertices += 1
                elif get_pixel(stack, x, y - 1, z) == 1:
                    n_edge_vertices += 1

    # top->bottom edges
    for x in range(0, width, x_inc):
        for y in range(0, height, y_inc):
            for z in range(1, slice):
                if get_pixel(stack, x, y, z) == 1:
                    n_edge_vertices += 1
                elif get_pixel(stack, x, y, z - 1) == 1:
                    n_edge_vertices += 1

    return n_edge_vertices

def get_face_vertices(stack):
    slice, height, width= get_dimension(stack)

    x_inc = max(1, width - 1)
    y_inc = max(1, height - 1)
    z_inc = max(1, slice - 1)
    n_face_vertices = 0

    # top and bottom faces (all 4 edges)
    for z in range(0, slice, z_inc):
        for y in range(0, height+1):
            for x in range(0, width+1):
                # if the voxel or any of its neighbors are foreground, the vertex is counted
                if get_pixel(stack, x, y, z) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x, y - 1, z) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x - 1, y - 1, z) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x - 1, y, z) == 1:
                    n_face_vertices += 1

    # left and right faces (2 vertical edges)
    for x in range(0, width, x_inc):
        for y in range(0, height+1):
            for z in range(1, slice):
                # if the voxel or any of its neighbors are foreground, the vertex is counted
                if get_pixel(stack, x, y, z) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x, y - 1, z) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x, y - 1, z - 1) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x, y, z - 1) == 1:
                    n_face_vertices += 1

    # back and front faces (0 vertical edges)
    for y in range(0, height, y_inc):
        for x in range(1, width):
            for z in range(1, slice):
                # if the voxel or any of its neighbors are foreground, the vertex is counted
                if get_pixel(stack, x, y, z) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x, y, z - 1) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x - 1, y, z - 1) == 1:
                    n_face_vertices += 1
                elif get_pixel(stack, x - 1, y, z) == 1:
                    n_face_vertices += 1

    return n_face_vertices

def get_face_edges(stack):
    slice, height, width= get_dimension(stack)

    x_inc = max(1, width - 1)
    y_inc = max(1, height - 1)
    z_inc = max(1, slice - 1)
    n_face_edges = 0

    # top and bottom faces (all 4 edges)
    # check 2 edges per voxel
    for z in range(0, slice, z_inc):
        for y in range(0, height+1):
            for x in range(0, width+1):
                # if the voxel or any of its neighbors are foreground, the vertex is counted
                if get_pixel(stack, x, y, z) == 1:
                    n_face_edges += 2
                else:
                    if get_pixel(stack, x, y - 1, z) == 1:
                        n_face_edges += 1
                    if get_pixel(stack, x - 1, y, z) == 1:
                        n_face_edges += 1

    # back and front faces, horizontal edges
    for y in range(0, height, y_inc):
        for z in range(1, slice):
            for x in range(0, width):
                if get_pixel(stack, x, y, z) == 1:
                    n_face_edges += 1
                elif get_pixel(stack, x, y, z - 1) == 1:
                    n_face_edges += 1

    # back and front faces, vertical edges
    for y in range(0, height, y_inc):
        for z in range(0, slice):
            for x in range(0, width + 1):
                if get_pixel(stack, x, y, z) == 1:
                    n_face_edges += 1
                elif get_pixel(stack, x - 1, y, z) == 1:
                    n_face_edges += 1

    # left and right stack faces, horizontal edges
    for x in range(0, width, x_inc):
        for z in range(1, slice):
            for y in range(0, height):
                if get_pixel(stack, x, y, z) == 1:
                    n_face_edges += 1
                elif get_pixel(stack, x, y, z - 1) == 1:
                    n_face_edges += 1

    # left and right stack faces, vertical voxel edges
    for x in range(0, width, x_inc):
        for z in range(0, slice):
            for y in range(1, height):
                if get_pixel(stack, x, y, z) == 1:
                    n_face_edges += 1
                elif get_pixel(stack, x, y - 1, z) == 1:
                    n_face_edges += 1

    return n_face_edges

def correct_for_edges(stack):
    f = get_stack_vertices(stack)
    e = get_stack_edges(stack) + 3 * f
    c = get_stack_faces(stack) + 2 * e - 3 * f
    d = get_edge_vertices(stack) + f
    a = get_face_vertices(stack)
    b = get_face_edges(stack)

    chi_zero = f
    chi_one = float(d - e)
    chi_two = float(a - b + c)

    edge_correction = chi_two / 2 + chi_one / 4 + chi_zero / 8

    return edge_correction



# Function to choose folder with STL files
def choose_files():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the folder containing STL files")
    
    if not folder_path:
        print("No folder selected.")
        return None
    
    stl_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".stl")]
    
    if not stl_files:
        print("No STL files found in the selected folder.")
        return None
    
    for stl_file in stl_files:
        yield stl_file

# STL files elaboration function
def process_stl(file_path):
    file_name = os.path.basename(file_path)
    mesh = vedo.Mesh(file_path)
    Area = mesh.area()
    volume_ = mesh.volume()
   
    h = 10.4
    l = 10.4
    p = 3.4
    vol_base = h * l * p
    porosity = (1 - (volume_ / vol_base)) * 100
    if volume_ < 0.5:
        return {
            "file_name": file_name,
            "BS_BV": None,
            "porosity": porosity,
            "conn_density": None,
            "thickness": None,
            "spacing": None
        }
    
    BS_BV = Area / volume_

    print(f"\nProcessing file: {file_name}")
    print("\nPorosity:", round(porosity, 3))
    print("\nArea/volume:", round(BS_BV, 3))

    vedo.settings.use_depth_peeling = True

    bounds = mesh.bounds()
    zmin, zmax = bounds[-2:]
    height = abs(zmin) + abs(zmax)
    xmin, xmax = bounds[:2]
    length = abs(xmin) + abs(xmax)
    slices = 100
    pixels = 850
    voxel = h / pixels  # (mm)
    spacing1 = [0, 0, 0]
    spacing1[0] = spacing1[1] = length / pixels
    spacing1[2] = height / slices
    volume = mesh.binarize(spacing=spacing1)
    volume.write(f"{file_name}_scaffold.tiff")
    #volume.show()
    tiff_volume = tifffile.imread(f"{file_name}_scaffold.tiff") > 0
    B_b = np.where(tiff_volume> 0, 1, 0).astype(np.uint8) 

    B = remove_black_slices(B_b)
    if B.shape[0] < 0.5 * B_b.shape[0]:
        print(f"Skipping detailed processing for file {file_name} because black slices {B.shape[0]} exceed 50% of total slices {B_b[0]}")
        # Save porosity and BS_BV even if the file is skipped
        return {
            "file_name": file_name,
            "BS_BV": BS_BV,
            "porosity": porosity,
            "conn_density": None,
            "thickness": None,
            "spacing": None
        }

    tifffile.imwrite(f"{file_name}_scaffold_binarized.tiff", B * 255)

    # TRABECULAR THICKNESS AND TRABECULAR SPACING
    thickness, separation, std_thickness, std_separation = apply_local_thickness(tiff_volume, voxel)

    # PURIFY
    stacked_purify_3d = purify_stack(B)
    # If you want visualize scaffold images:

    ## VISUALIZE ##
    num_slices = stacked_purify_3d.shape[0]

    #Create a figure and axes
    fig, ax = plt.subplots()

    # Update slice
    def update_slice(slice_index):
        ax.clear()
        current_slice = stacked_purify_3d[slice_index, :, :]
        ax.imshow(current_slice, cmap='gray')
        ax.set_title(f"Slice {slice_index+1}")
        plt.draw()


    # Press bottom to see slices
    def on_key(event):
        if event.key == 'right':
            # Next slice
            slice_index = (int(ax.get_title().split()[1]) + 1) % num_slices
            update_slice(slice_index)
        elif event.key == 'left':
            # Previous slice
            slice_index = (int(ax.get_title().split()[1]) - 1) % num_slices
            update_slice(slice_index)

    # Connect function to key_press_event 
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_slice(0)
    #plt.show()

    # Percentage of purified image compared to original one
    percentages = []

    # Calculate white pixels for each image
    for i in range(B.shape[0]):
        original_white_pixels = np.sum(B[i, :, :] == 1)
        purified_white_pixels = np.sum(stacked_purify_3d[i,:,:] == 1)
    
        # Calculate white pixels of purified image compared to original one
        percentage = (purified_white_pixels / original_white_pixels) * 100
        percentages.append(percentage)


    average_percentage = np.mean(percentages)


    print("\nPurify percentage is {:.1f}%".format(round(average_percentage,1)))


    sum_euler, delta_chi, connectivity, conn_density = connect(stacked_purify_3d, volume_)

    print("\nMean thickness_algorithm:", round(thickness, 3))

    print("\nMean spacing_algorithm:", round(separation, 3))

    print("ok")

    return {
        "file_name": file_name,
        "BS_BV": BS_BV,
        "porosity": porosity,
        "conn_density": conn_density,
        "thickness": thickness,
        "spacing": separation
    }

# List to collect results
results = []

# Iterate through all STL files in the selected folder
for file in choose_files():
    if file:
        result = process_stl(file)
        if result:
            results.append(result)


# Create a DataFrame with the results and save it to an Excel file
df = pd.DataFrame(results)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "results.xlsx")
df.to_excel(file_path, index=False)

print("Results saved to results.xlsx")

end_time = time.time()

elapsed_time = (end_time - start_time)/60
print("The program took", elapsed_time, "minutes to execute.")

