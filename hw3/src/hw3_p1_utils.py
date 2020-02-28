import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ========== UTILS ========== 
def get_unique_colors(img):
    return (np.unique(img.reshape(-1, img.shape[2]), axis=0))

def getNextNewColor(usedColors):    
    newColor = (np.random.choice(range(256), size=3))
    while np.any([np.all(uc == newColor) for uc in usedColors]): # if newColor matches any of the oldColors
        newColor = (np.random.choice(range(256), size=3))
    return newColor

def gen_color_key(color):
    return "_".join(str(channel) for channel in color)

def pretty_print_obj(obj):
    for (k,v) in obj.items():
        print(k,":",v)

# ========== Connected Components ========== 
def floodfill(surface, x, y, oldColors, usedColors):
    if surface[x][y] not in oldColors: # Has new color already. No need to look.
        return surface, usedColors

    colorOfFocus = surface[x][y].copy()
    newColor = getNextNewColor(usedColors)    
    usedColors = np.vstack([usedColors, newColor])

    # Add first coord into stack
    theStack = [(x, y)]
    
    while len(theStack) > 0:
        x, y = theStack.pop()
        
        if x < 0 or x > surface.shape[0]-1 or y < 0 or y > surface.shape[1]-1: # Out of Bounds
            continue
        
        if np.all(surface[x][y] == colorOfFocus):
            surface[x][y] = newColor
            theStack.append((x+1, y))  # right
            theStack.append((x-1, y))  # left
            theStack.append((x, y+1))  # down
            theStack.append((x, y-1))  # up

    return surface, usedColors

def flood_fill_multi(img, debug=False):
    oldColors = get_unique_colors(img)
    usedColors = get_unique_colors(img)
    
    if debug:
        print("Used Colors")
        plt.imshow(usedColors)
        plt.show()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img, usedColors = floodfill(img, i, j, oldColors, usedColors)

    return img, usedColors

def get_largest_components(img, usedColors, n=2):
    h = {}    
    # Get count of all connected components
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = img[i][j]
            color_key = gen_color_key(color)
            if color_key in h.keys():
                h[color_key] += 1
            else:
                h[color_key] = 1

    h_desc = [item[0] for item in sorted(h.items(), key = lambda kv:(kv[1], kv[0]))]    # filter
    h_desc_rev_filt = list(reversed(h_desc))[:n]
    top_n_components = [[int(ck) for ck in colorkey.split('_')] for colorkey in h_desc_rev_filt]
    return top_n_components

def filter_out_colors(img, colors, bgColor):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            curr_color = img[i][j]
            if not np.any([np.all(c == curr_color) for c in colors]):  #  if curr_color not in c ,then change current pixel to bgColor
                img[i][j] = bgColor
    return img

# Image preprocess
def img_preprocess_0(img):
    kernel_10x10 = np.ones((10,10),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_10x10) # Open. Fill in any holes.
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_10x10) # Open. Fill in any holes.
    img = cv2.GaussianBlur(img, (5,5), 0)

    _, img = cv2.threshold(img,250,255,cv2.THRESH_BINARY)
    return img

def img_preprocess_1(img):
    kernel_2x2 = np.ones((2,2),np.uint8)
    img = cv2.erode(img, kernel_2x2, iterations=1)
    img = cv2.GaussianBlur(img, (5,5), 0)    
    img = cv2.dilate(img, kernel_2x2, iterations=1)
    _, img = cv2.threshold(img,175,255,cv2.THRESH_BINARY)
    return img

def img_preprocess_2(img):
    kernel_5x5 = np.ones((5,5),np.uint8)
    kernel_3x3 = np.ones((3,3),np.uint8)

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_5x5) # Open. Fill in any holes.
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_3x3) # Close. Remove small blob
    _, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    return img

def img_preprocess_3(img):
    kernel_10x10 = np.ones((10,10),np.uint8)
    kernel_7x7 = np.ones((7,7),np.uint8)
    kernel_3x3 = np.ones((3,3),np.uint8)
    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_10x10) # Close. Remove small blob    
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_10x10) # Close. Remove small blob
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_7x7) # Close. Remove small blob
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_3x3) # Close. Remove small blob
    _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return img

def get_connected_components(img, preprocess_mode, n=2, debug=False, output_res=False):
    if debug:
        print("Original")
        plt.imshow(img)
        plt.show()

    if preprocess_mode == 0:
        img = img_preprocess_0(img)
    if preprocess_mode == 1:
        img = img_preprocess_1(img)
    elif preprocess_mode == 2:
        img = img_preprocess_2(img)
    else:
        img = img_preprocess_3(img)

    img, usedColors, = flood_fill_multi(img, debug=debug)
    
    if debug:
        print("After preprocess and floodfill multi")
        plt.imshow(img)
        plt.show()

    largest_colors = get_largest_components(img, usedColors, n=n)
    if debug:
        print("Largest_colors", largest_colors)
        plt.imshow(np.array([largest_colors]))
        plt.show()
    
    if n > 1:
        largest_colors = largest_colors[1:]
    
    img = filter_out_colors(img, largest_colors, [255, 255, 255])
    if debug or output_res:
        print("After filter out smallest colors")
        plt.imshow(img)
        plt.show()
    
    return img, largest_colors

# ========== Boundary Tracing ========== 

def get_next_cw_pos(center, curr): # Get next clockwise pixel based on curr and center.
#     '''
#     C is left of center. 
#     [[...],
#      [C center X]]
#     '''
    if curr[1] == center[1] and curr[0]+1 == center[0]: 
        return [curr[0], curr[1]-1]
#     '''
#     C is left-top of center. 
#     [[C X X],
#      [X center X]]
#      '''
    elif curr[1]+1 == center[1] and curr[0]+1 == center[0]: 
        return [curr[0]+1, curr[1]]
#     '''
#     C is top of center. 
#     [[X C X],
#      [X center X]]
#     '''
    elif curr[1]+1 == center[1] and curr[0] == center[0]: 
        return [curr[0]+1, curr[1]]
#     '''
#     C is top-right of center. 
#     [[X X C],
#      [X center X]]
#     '''
    elif curr[1]+1 == center[1] and curr[0]-1 == center[0]: 
        return [curr[0], curr[1]+1]
#     '''
#     C is right of center. 
#     [[X X X],
#      [X center C]]
#     '''
    elif curr[1] == center[1] and curr[0]-1 == center[0]: 
        return [curr[0], curr[1]+1]
#     '''
#     C is right-bot of center. 
#     [[X X X],
#      [X center X],
#      [X X C]]
#     '''
    elif curr[1]-1 == center[1] and curr[0]-1 == center[0]: 
        return [curr[0]-1, curr[1]]
#     '''
#     C is bot of center. 
#     [[X X X],
#      [X center X],
#      [X C X]]
#     '''
    elif curr[1]-1 == center[1] and curr[0] == center[0]: 
        return [curr[0]-1, curr[1]]
#     '''
#     C is left-bot of center. 
#     [[X X X],
#      [X center X],
#      [C X X]]
#     '''
    elif curr[1]-1 == center[1] and curr[0]+1 == center[0]: 
        return [curr[0], curr[1]-1]
#     '''
#     C is left of center. 
#     [[X X X],
#      [C center X],
#      [X X X]]
#     '''
    elif curr[1] == center[1] and curr[0]+1 == center[0]: 
        return [curr[0], curr[1]-1]
    else:
        print("ERROR")        

def boundary_tracing(img, target_colors, boundary_draw_color, debug=False):    
    # p = current pixel, c = pixel in consideration, b = pixel that is used to enter into p, B = boundaries
    B = []
    ptColor = [255,255,255]
    start = None
    
    #   From bottom to top and left to right scan the cells of T until a black pixel, s, of P is found.
    for j in range(img.shape[0]):
        if start is not None:
            break
        for i in range(img.shape[1]):

            if start is not None:
                break
            
            if np.any([np.all(img[j][i] == tc) for tc in target_colors]): # is ptColor
                start = [i,j]
                if debug:
                    print("Found first black pixel (i,j) = ({})".format(start))

    if start is None:
        print("ERROR | Start is None")
        return None, None

    B.append(start)
    p = start
    b = [start[0]-1, start[1]] # TODO: border handle cases.
    c = get_next_cw_pos(p, b)
    if debug:
        print("About to start. Next move is c={}, b={}".format(c,b))

    while not np.all(c == start): # while c != start
        if c[0] < 0 or c[0] > img.shape[1]-1 or c[1] < 0 or c[1] > img.shape[0]-1: # Out of bounds
            b = c
            c = get_next_cw_pos(p, b)
            if debug:
                print("out of bounds. Continue . Next move is c={}, b={}".format(c,b))
        elif np.any([np.all(img[c[1]][c[0]] == tc) for tc in target_colors]): # color at c is pointColor
            B.append(c)
            p = c
            c = get_next_cw_pos(p, b)
            if debug:
                print("Add c into B. Next move is B={}, c={}, b={}.".format(B,c,b))
        else:
            b = c
            c = get_next_cw_pos(p, b)
            if debug:
                print("No find black pixel. Next move is c={}, b={}".format(c,b))
                
    # Draw Boundary with orig
    boundary_overlay_img = img.copy()
    for b_coord in B:
        boundary_overlay_img[b_coord[1]][b_coord[0]] = boundary_draw_color
        
    # Draw just boundary
    boundary_img = np.ones(img.shape) * 255
    for boundary in B:
        boundary_img[boundary[1], boundary[0]] = (0,0,0)

    return B, boundary_overlay_img, boundary_img

# ========== Skeletonize ========== 
def skeletonize(img, gray_then_thres=False, debug=False):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray_then_thres:
        _, img = cv2.threshold(img,250,255,cv2.THRESH_BINARY)   # double threshold - 2 images
    img = cv2.bitwise_not(img)  # flip

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    ret,img = cv2.threshold(img,10,255,0)   # threshold

    img = img.copy()
    skel = img.copy()

    skel[:,:] = 0   # blank template
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    count = 0
    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel) # Opening
        temp  = cv2.subtract(img, temp) # Get (next) outer layer skeleton
        skel = cv2.bitwise_or(skel, temp)   # add onto skel

        img[:,:] = eroded[:,:]
        count += 1
        if debug:
            print("count=", count)
        if cv2.countNonZero(img) == 0:  # until image is 0 (cannot erode anymore)
            break

    return skel

# ========== Image Moments ========== 
# For each relevant region/object, compute the area, 
# orientation, and circularity (Emin/Emax). Also, identify 
# and count the boundary pixels of each region, and compute 
# compactness, the ratio of the area to the perimeter.
def calc_center_of_mass(img, obj_color):
    x_bar = 0
    y_bar = 0
    area = 0
    
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if np.any(np.all(img[j][i] == obj_color)):
                area += 1
                x_bar += i
                y_bar += j
    
    x_bar = x_bar/area
    y_bar = y_bar/area
    
    return area, x_bar, y_bar

def calc_perimeter(boundary_img, boundary_color=[0,0,0]):
    perimeter = 0

    for i in range(boundary_img.shape[1]):
        for j in range(boundary_img.shape[0]):
            if np.any(np.all(boundary_img[j][i] == boundary_color)):
                perimeter += 1
    return perimeter

def calc_second_moment(img, obj_color,x_bar, y_bar, area):
    a,b,c = 0,0,0
    m_11, m_20, m_02 = 0,0,0

    for i in range(img.shape[1]): # x
        for j in range(img.shape[0]): # y
            if np.any(np.all(img[j][i] == obj_color)):
                a += pow((i - x_bar), 2)
                b += 2 * (i-x_bar) * (j-y_bar)
                c += pow((j - y_bar), 2)
                m_20 += pow(i, 2)
                m_02 += pow(j, 2)
                m_11 += (i * j)

    h = pow(pow((a-c), 2) + pow(b,2), 0.5)
    
    E_min = (a+c)/2 - pow((a-c),2)/(2*h) - pow(b,2)/(2*h) if h>0 else (a+c)/2
    E_max = (a+c)/2 + pow((a-c),2)/(2*h) + pow(b,2)/(2*h) if h>0 else (a+c)/2
    circularity = E_min/E_max if (E_min > 0 and E_max > 0) else 0

    # Orientation

    # orientation = 0.5 * np.arctan((2*b)/(a+c)) if (a+c) > 0 else 0    # 1
    # arctan_num = (2 * (b - x_bar * y_bar))    # 2
    # arctan_denum = ((a - pow(x_bar, 2)) - (c - pow(y_bar, 2)))    # 2
    # orientation = np.rad2deg(0.5 * np.arctan(arctan_num/arctan_denum))    #2

    mu_bar_20 = (m_20/area) - pow(x_bar, 2)
    mu_bar_02 = (m_02/area) - pow(y_bar, 2)
    mu_bar_11 = (m_11/area) - (x_bar * y_bar)
    arctan_in = (2 * mu_bar_11) / (mu_bar_20 - mu_bar_02)
    orientation = np.rad2deg(0.5 * np.arctan(arctan_in))

    return a,b,c,h,E_min,E_max,circularity,orientation

# In: x,y + rgb channel image
def calc_moment_numbers(img, obj_color, boundary_img, debug=False):
    perimeter = calc_perimeter(boundary_img, boundary_color=[0,0,0])
    area, x_bar, y_bar = calc_center_of_mass(img, obj_color)
    compactness = pow(perimeter, 2)/area
    ratio_area_perim = area/perimeter
    a,b,c,h,E_min,E_max,circularity,orientation = calc_second_moment(img, obj_color, x_bar, y_bar, area)

    if debug:
        print("a={},b={},c={},h={},E_min={},E_max={}".format(a,b,c,h,E_min,E_max))
        print("x_bar={}, y_bar={}, perimeter={}".format(x_bar, y_bar, perimeter))
        print("Circularity={},orientation={}".format(circularity,orientation))
        print("Compactness={}, Ratio(area,perim)={}".format(compactness, ratio_area_perim))
    
    moment_res = {
        'area': area,
        'perimeter': perimeter,
        'ratio_area_perim': ratio_area_perim,
        'compactness': compactness,
        'orientation': orientation,
        'circularity': circularity
    }
    return moment_res
