""" Gets the images with max width and max height from the image folder.
"""
from PIL import Image

def get_max_width_height(all_images_path, all_file_names):
    end = len(all_file_names)
    progress, max_width, max_height = 0, 0, 0
    for file in all_file_names:
        width, height = Image.open(f"{all_images_path}{file}", 'r').size
        if width > max_width:
            max_width = width
            wname = file
        if height > max_height:
            max_height = height
            hname = file
        progress += 1
        if progress % 1000 == 0:
            print(f"Progress: {progress}/{end}", end='\r')
    return (max_width, max_height, wname, hname)

max_dims = get_max_width_height(all_img_path, all_file_names)
print(max_dims)
