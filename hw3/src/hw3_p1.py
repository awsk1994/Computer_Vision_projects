options = {
    'img_dir': './img/p1',
    'output_dir': './result',
    'write_output': False,
    'show_res': False,
    'show_moment_res': True,
    'moment_debug': True,
    'debug': False,
    'show_which_step': True
}

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from hw3_p1_utils import *

def main():
	# 0. Original
	if options['show_which_step']:
		print("0. Getting image")

	open_full = cv2.imread(options['img_dir'] + "/" + 'open-bw-full.png')
	open_partial = cv2.imread(options['img_dir'] + "/" + 'open-bw-partial.png')
	open_fist = cv2.imread(options['img_dir'] + "/" + 'open_fist-bw.png')
	tumor = cv2.imread(options['img_dir'] + "/" + 'tumor-fold.png')

	if options['show_res']:
	    plt.imshow(open_full)
	    plt.show()
	    plt.imshow(open_partial)
	    plt.show()
	    plt.imshow(open_fist)
	    plt.show()
	    plt.imshow(tumor)
	    plt.show()

	# 1,2. Connected Components + Filter largest components
	if options['show_which_step']:
		print("1,2. Calculating Connected Components + Filter Largest Components")

	open_full_cc, open_full_colors = get_connected_components(open_full, preprocess_mode=0, n=2, debug=options['debug'], output_res=options['debug'])
	open_partial_cc, open_partial_colors = get_connected_components(open_partial, preprocess_mode=1, n=2, debug=options['debug'], output_res=options['debug'])
	open_fist_cc, open_fist_colors = get_connected_components(open_fist, preprocess_mode=2, debug=options['debug'], n=3, output_res=options['debug'])
	tumor_cc, tumor_colors = get_connected_components(tumor, preprocess_mode=3, debug=options['debug'], n=2, output_res=options['debug'])

	if options['show_res']:
	    plt.imshow(open_full_cc)
	    plt.show()
	    plt.imshow([open_full_colors])
	    plt.show()

	    plt.imshow(open_partial_cc)
	    plt.show()
	    plt.imshow([open_partial_colors])
	    plt.show()

	    plt.imshow(open_fist_cc)
	    plt.show()
	    plt.imshow([open_fist_colors])
	    plt.show()

	    plt.imshow(tumor_cc)
	    plt.show()
	    plt.imshow([tumor_colors])
	    plt.show()

	if options['write_output']:
	    cv2.imwrite(options['output_dir'] + '/open_full_cc.png', open_full_cc)
	    cv2.imwrite(options['output_dir'] + '/open_partial_cc.png', open_partial_cc)
	    cv2.imwrite(options['output_dir'] + '/open_fist_cc.png', open_fist_cc)
	    cv2.imwrite(options['output_dir'] + '/tumor_cc.png', tumor_cc)

	# 3. Boundary 
	if options['show_which_step']:
		print("3. Applying boundary tracing algorithm")

	open_full_boundaries, open_full_overlay, open_full_boundary_img = boundary_tracing(open_full_cc, open_full_colors, [0,0,0], debug=options['debug'])
	open_partial_boundary, open_partial_overlay, open_partial_boundary_img = boundary_tracing(open_partial_cc, open_partial_colors, [0,0,0], debug=options['debug'])

	open_fist_boundary_1, open_fist_overlay_1, open_fist_boundary_img_1 = boundary_tracing(open_fist_cc, [open_fist_colors[0]], [0,0,0], debug=options['debug'])
	open_fist_boundary_2, open_fist_overlay_2, open_fist_boundary_img_2 = boundary_tracing(open_fist_cc, [open_fist_colors[1]], [0,0,0], debug=options['debug'])

	# TODO: Merge into method below
	for i in range(open_fist_boundary_img_1.shape[1]):
	    for j in range(open_fist_boundary_img_1.shape[0]):
	        if np.all(open_fist_boundary_img_2[j][i] == [0,0,0]):
	            open_fist_boundary_img_1[j][i] = [0,0,0]

	tumor_boundary, tumor_overlay, tumor_boundary_img = boundary_tracing(tumor_cc, tumor_colors, [0,0,0], debug=options['debug'])

	if options['write_output']:
	    cv2.imwrite(options['output_dir'] + '/open_full_boundary_img.png', open_full_boundary_img)
	    cv2.imwrite(options['output_dir'] + '/open_partial_boundary_img.png', open_partial_boundary_img)
	    cv2.imwrite(options['output_dir'] + '/open_fist_boundary_img.png', open_fist_boundary_img_1)
	    cv2.imwrite(options['output_dir'] + '/tumor_boundary_img.png', tumor_boundary_img)

	if options['show_res']:
	    plt.imshow(open_full_boundary_img)
	    plt.show()

	    plt.imshow(open_partial_boundary_img)
	    plt.show()

	    plt.imshow(open_fist_boundary_img_1)
	    plt.show()

	    plt.imshow(tumor_boundary_img)
	    plt.show()

	# 4. Moment Calculations
	if options['show_which_step']:
		print("4. Calculating Moment")

	open_full_moment = calc_moment_numbers(open_full_cc,open_full_colors[0], open_full_boundary_img, debug=options['moment_debug'])
	open_partial_moment = calc_moment_numbers(open_partial_cc,open_partial_colors[0], open_partial_boundary_img, debug=options['moment_debug'])
	open_fist_moment_1 = calc_moment_numbers(open_fist_cc,open_fist_colors[0], open_fist_boundary_img_1, debug=options['moment_debug'])
	open_fist_moment_2 = calc_moment_numbers(open_fist_cc,open_fist_colors[1], open_fist_boundary_img_2, debug=options['moment_debug'])
	tumor_moment = calc_moment_numbers(tumor_cc,tumor_colors[0], tumor_boundary_img, debug=options['moment_debug'])

	if options['show_moment_res']:
	    print("open_full_moment")
	    pretty_print_obj(open_full_moment)
	    
	    print("\nopen_partial_moment")
	    pretty_print_obj(open_partial_moment)
	    
	    print("\nopen_fist_moment_1")
	    pretty_print_obj(open_fist_moment_1)
	    
	    print("\nopen_fist_moment_2")
	    pretty_print_obj(open_fist_moment_2)
	    
	    print("\ntumor_moment")
	    pretty_print_obj(tumor_moment)

	# 5. Skeleton Finding Algorithm
	if options['show_which_step']:
		print("5. Apply Skeleton Finding Algorithm")

	open_full_skeleton = skeletonize(open_full_cc)
	open_partial_skeleton = skeletonize(open_partial_cc)
	open_fist_skeleton = skeletonize(open_fist_cc, gray_then_thres=True)
	tumor_skeleton = skeletonize(tumor_cc)

	if options['show_res']:
	    plt.imshow(open_full_skeleton)
	    plt.show()
	    plt.imshow(open_partial_skeleton)
	    plt.show()
	    plt.imshow(open_fist_skeleton)
	    plt.show()
	    plt.imshow(tumor_skeleton)
	    plt.show()

	if options['write_output']:
	    cv2.imwrite(options['output_dir'] + '/open_full_skeleton.png', open_full_skeleton)
	    cv2.imwrite(options['output_dir'] + '/open_partial_skeleton.png', open_partial_skeleton)
	    cv2.imwrite(options['output_dir'] + '/open_fist_skeleton.png', open_fist_skeleton)
	    cv2.imwrite(options['output_dir'] + '/tumor_skeleton.png', tumor_skeleton)

if __name__ == "__main__":
	main()
