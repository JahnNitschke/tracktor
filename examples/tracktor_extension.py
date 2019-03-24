import numpy as np
import pandas as pd
import cv2
import itertools
import os
import getXY as g
import kalman as kl
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def get_spaced_colors(n):
    """
    This function draws RGB values from the RGB space to create colours for plotting of tracked identities
    color_ids: list of tuples of RGB values
    """
    
    if n == 1:
        color_ids = [(0, 0, 255)]
        return color_ids
    else:
        max_value = 16581375 #255**3
        interval = int(max_value / n)
        colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
        color_ids = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors[0:-1]]
        return color_ids
    
def round_up_to_odd(f):
    """
    Function to round a number up to the next odd number
    """
    return int(np.ceil(f) // 2 * 2 + 1)

def detect_and_draw_contours(frame, thresh, meas_last, meas_now, erosion_dilation_count, min_area = 0, max_area = 10000):
    """
    This function detects contours, thresholds them based on area and draws them.
    
    Parameters
    ----------
    frame: ndarray, shape(n_rows, n_cols, 3)
        source image containing all three colour channels
    thresh: ndarray, shape(n_rows, n_cols, 1)
        binarised(0,255) image
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
    min_area: int
        minimum area threhold used to detect the object of interest
    max_area: int
        maximum area threhold used to detect the object of interest
        
    Returns
    -------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
    """
    # Detect contours and draw them based on specified area thresholds
    img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    meas_now = list(meas_now)
    meas_last = list(meas_last)
    
    final = frame.copy()

    i = 0
    
    if erosion_dilation_count == 0:
        meas_last = meas_now.copy()
    
    del meas_now[:]
    while i < len(contours):
        area = cv2.contourArea(contours[i])
        if area < min_area or area > max_area:
            del contours[i]
        else:
            cv2.drawContours(final, contours, i, (0,0,255), 1)
            M = cv2.moments(contours[i])
            if M['m00'] != 0:
            	cx = M['m10']/M['m00']
            	cy = M['m01']/M['m00']
            else:
            	cx = 0
            	cy = 0
            meas_now.append([cx,cy])
            i += 1
    return final, contours, meas_last, meas_now

def mouse_input(frame, n_inds, meas_last):
    """
    This function lets the user click on the objects positions in the initial frame.
    The function receives the frame, the number of objects n_inds and returns the positions of the objects
    for the first frame as meas_last, respectively as kl_estimate
    """
    meas_last = g.getXY(frame)[-n_inds:]
    kl_estimate = meas_last.copy()
    return meas_last, kl_estimate
    
def apply_dilation_b_on_w(frame):
    """
    This set of dilation and erosion functions erodes/dilates the background of the parsed frame
    with a (2,2) square. The functions are written out in a set of four to avoid confusion on what is eroded/dilated,
    since this depends on the type of image.
    """
    
    #dilates black blobs on white bg by eroding white bg    
    kernel = np.ones((2,2),np.uint8)
    manipulated_frame = cv2.erode(frame, kernel, iterations=1)
    return manipulated_frame

def apply_erosion_b_on_w(frame):
    #erodes black blobs on white bg by dilating white bg    
    kernel = np.ones((2,2),np.uint8)
    manipulated_frame = cv2.dilate(frame, kernel, iterations=1)
    return manipulated_frame   

def apply_dilation_w_on_b(frame):
    #erodes white blobs on black background by dilating white blobs
    kernel = np.ones((2,2),np.uint8)
    manipulated_frame = cv2.dilate(frame, kernel, iterations=1)
    return manipulated_frame

def apply_erosion_w_on_b(frame):
    #erodes white blobs in black bg by eroding white blobs
    kernel = np.ones((2,2),np.uint8)
    manipulated_frame = cv2.erode(frame, kernel, iterations=1)
    return manipulated_frame   
    
def apply_dilation_and_erosion(mask, final, contours, frame, thresh, meas_now, meas_last, n_inds, ero_dil_limit, ero_dil_count, min_area, max_area):
    """
    This function iteratively applies dilation and erosion to the thresholded image and then uses the 
    detect_and_draw_contours function to iteratively detect contours in the altered image and reiterate until the desired
    number of objects is detected or a user defined limit (ero_dil_limit) is exceeded. If the discrepance to the present
    object amount n_inds increases after a erosion/dilation, the erosion/dilation is also stopped.
    
    Parameters
    ----------
    -all arguments detect_and_draw_contours requires
    -ero_dil_limit and ero_dil_count
    
    Returns
    -------
    returns a new final frame, new contours and a new list of positions in meas_now
    
    """
    
    while len(meas_now) != n_inds and ero_dil_count < ero_dil_limit:
    #detected contours don't match expected animals and the ero_dil_limit is not exceeded -> erode/dilate frame
        if len(meas_now) < n_inds: #if too few contours were detected erode the contours
            #save frame before applying erosion in order to compare amount of detected contours
            meas_now_after_last_step = meas_now.copy()
            n_discrepance_after_last_step = abs(len(meas_now)-n_inds)
            final_after_last_step = final.copy()
            contours_after_last_step = contours.copy()
            thresh = apply_erosion_w_on_b(thresh)
            thresh[mask == 0] = 0 #sets colour outside of roi to black
            ero_dil_count += 1
            final, contours, meas_last, meas_now = detect_and_draw_contours(frame, thresh, meas_last, meas_now, ero_dil_count, min_area, max_area)
        elif len(meas_now) > n_inds: #if too many contours were detected dilate the contours
            #save frame before applying dilation in order to compare amount of detected contours
            meas_now_after_last_step = meas_now.copy()
            n_discrepance_after_last_step = abs(len(meas_now)-n_inds)
            final_after_last_step = final.copy()
            contours_after_last_step = contours.copy()
            thresh = apply_dilation_w_on_b(thresh)
            thresh[mask == 0] = 0 #sets colour outside of roi to black
            ero_dil_count += 1
            final, contours, meas_last, meas_now = detect_and_draw_contours(frame, thresh, meas_last, meas_now, ero_dil_count, min_area, max_area)
        
        n_discrepance = abs(len(meas_now)-n_inds)
        if n_discrepance >= n_discrepance_after_last_step:
            final = final_after_last_step
            contours = contours_after_last_step
            meas_now = meas_now_after_last_step
            break
    return final, contours, meas_now

def apply_simple_kalman(R, n_inds, kl_velocities, kl_Ps, kl_estimate, meas_last):
    """
    This function makes a kalman based estimate of object positions the current frame by using positions and velocities from 
    the last frame.
    
    Parameters
    ----------
    n_inds = number of objects
    kl_velocities = velocitiy estimate
    kl_estimate = position estimate
    kl_Ps = initial uncertainty convariance matrices
    R = measurement noise
    meas_last = positions of the last frame
    
    Returns
    -------
    returns kl_estimate, kl_velocities and kl_Ps. kl_estimates is subsequently used to try to assign positions of lost points,
    whereas the velocities and Ps are fed into the next kalman estimation.
    """
    
    for i in np.arange(0,n_inds,1): #iterate over objects
        #get kalman function arguments x and P from kl_estimate, kl_velocities and kl_Ps
        x = np.append(kl_estimate[i], kl_velocities[i]).reshape([4,1])
        P = kl_Ps[i]
            
        #apply simple kalman and store results
        x, P = kl.kalman_xy(x, P, meas_last[i], R)
        #decompose x into velocities and positions
        kl_estimate[i] = np.reshape(x[:2], [2,])
        kl_velocities[i] = np.reshape(x[2:], [2,])
        kl_Ps[i] = P    
    return kl_estimate, kl_velocities, kl_Ps

def find_lost_points1(meas_last, meas_now, n_inds):
    """
    This function tries to identify objects which were lost from the last frame to the current frame. First, a euclidean
    distance matrix for every object in meas_last (or the kalman estimate, if kalmen = True in "global variables") 
    and meas_now is generated. For every object in meas_last the minimum distance to an object in meas_now is taken and
    sorted descendingly. The top indices of meas_last objects are considered as lost.
    
    Parameters
    ----------
    meas_last = positions of the last frame or the simple kalman estimate
    meas_now = positions of contours in current frame, which are fewer than in the last frame.
    n_inds = amount of tracked objects
    
    Returns
    -------
    lost_points = list of points from meas_last which are considered as lost
    """
    
    meas_now = np.array(meas_now)
    meas_last = np.array(meas_last) 
 
    dist_matrix = cdist(meas_last, meas_now) 
    #euclidean distance matrix between contours from last/estimate and current frame 
    
    hi_dis2last = np.argsort(np.min(dist_matrix, axis = 1))[::-1]
    #find minimum distance for every object in meas_last, sort them ascendingly (np.argsort) and reverse order ([::-1])
    
    n_discrepance = len(meas_last)-len(meas_now)
    #contour count difference between frames
    
    lost_points = np.array(meas_last)[hi_dis2last[0:n_discrepance]] 
    #take n_discrepance contours from hi_dis2last, which are the putative individuals which were list in the current frame    
    return lost_points

def find_lost_points2(meas_last, meas_now, n_inds):
    """
    This function makes use of the hungarian algorithm to find the objects which fit worst to the positions in meas_now.
    First, a list of indices of all possible subsets from meas_last with the size of meas_now is created. Analogously a list of
    corresponding coordinates is created. For every subset an euclidean distance matrix is created, positions are assigned and
    the cost as the sum of the cost matrix is compared. Eventually the best subset with the lowest cost is kept. The points,
    which are not in the best subset are considered as lost points.
    
    Parameters
    ----------
    meas_last = positions of the last frame or the simple kalman estimate
    meas_now = positions of contours in current frame, which are fewer than in the last frame.
    n_inds = amount of tracked objects
    
    Returns
    -------
    lost_points = list of points from meas_last which are considered as lost
    """
    
    lowest_cost = np.inf #initial cost value
    
    #generate list of all subsets from meas_last with the length of meas_now containing indices of objects and their positions
    all_subsets = list(itertools.combinations(enumerate(meas_last), len(meas_now)))

    #split up all_subsets into a list of lists with the object indices per subset
    indices = []
    indices_subset = []
    for subset in all_subsets:
        for position in subset:
            indices_subset.append(position[0])
        indices.append(indices_subset)
        indices_subset = []
    
    #split up all_subsets into a list of lists with the object coordinates per subset
    positions = []
    positions_subset = []
    for subset in all_subsets:
        for position in subset:
            positions_subset.append(list(position)[1])
        positions.append(positions_subset)
        positions_subset = []
    
    #iterate over position subsets
    for i in np.arange(0, len(positions)):
        subset = positions[i]
        idx = indices[i]
        cost = cdist(meas_last, subset) #create cost function
        row_ind, col_ind = linear_sum_assignment(cost) #hungarian algorithm
        if cost[row_ind, col_ind].sum() < lowest_cost: #update best_idx if the costs are lower than before
            lowest_cost = cost[row_ind, col_ind].sum()
            best_idx = np.array(idx)

    #find points which are not in the best subset and store them in lost_points
    lost_points = []
    for i in np.arange(0, n_inds):
        if i not in best_idx:
            lost_points.append((meas_last[i]))
    return lost_points

def assign_lost_points1(meas_now, lost_points): 
    """
    This function takes the lost_points in the last frame identified previously and tries to make statements about them in
    the current frame. First, it assumes that two blobs merged and the lost_points are contained in the closest contours.
    To avoid identical positions for objects, the position of the closest contour, which possibly includes the lost object,
    is not simply copied. Instead the putative position for the lost object is calculated as the weighed mean between the
    lost point and the closest contour (0.1 lost point/ 0.9 closest contour)
    
    Parameters
    ----------
    meas_now = positions of contours in current frame, which are fewer than in the last frame.
    n_inds = amount of tracked objects
    
    Returns
    -------
    extended_meas_now = meas_now plus calculated positions for lost objects
    """
    
    #find closest contours for lost_points
    dist_matrix = cdist(lost_points, meas_now)
    closest_contour = np.argmin(dist_matrix, axis = 1) #indexes meas_now

    #extend meas_now with copies of the points in meas_now which are closest to putative lost points
    extended_meas_now = list(meas_now.copy())
    i = 0 #indexes lost_points in this for loop
    for element in closest_contour:
        extended_meas_now.append([(lost_points[i][0]*0.1)+(meas_now[element][0]*0.9), 
        (lost_points[i][1]*0.1)+(meas_now[element][1]*0.9)])
        #calculate weighed mean
        i += 1 #iterate not only over contours but also over lost points
    return extended_meas_now

def assign_lost_points2(meas_now, lost_points):
    """
    This function takes meas_now and the putative lost_points. It is designed as a failsave strategy and simply
    keeps the position of the lost point (which can be the actual position in the last frame or the kalman estimate)
    by copying it over to meas_now.    
    
    Parameters
    ----------
    meas_now = positions of contours in current frame, which are fewer than in the last frame.
    n_inds = amount of tracked objects
    
    Returns
    -------
    extended_meas_now = meas_now plus calculated positions for lost objects
    """
    extended_meas_now = np.append(meas_now, lost_points, axis = 0) 
    #append the centroid of the lost individuals from the last frame to the current frame
    return extended_meas_now

def reorder_and_draw(final, colours, n_inds, col_ind, meas_now, df, fr_no, colour_ignored_area):
    """
    This function reorders the measurements in the current frame to match
    identity from previous frame. This is done by using the results of the
    hungarian algorithm from the array col_inds.
    
    Parameters
    ----------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    colours: list, tuple
        list of tuples that represent colours used to assign individual identities
    n_inds: int
        total number of individuals being tracked
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.core.frame.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    mot: bool
        this boolean determines if we apply the alogrithm to a multi-object
        tracking problem
        
    Returns
    -------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    """
    # Reorder contours based on results of the hungarian algorithm
    equal = np.array_equal(col_ind, list(range(len(col_ind))))
    if equal == False:
        current_ids = col_ind.copy()
        reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
        meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]

    # Draw centroids
    #if mot == False:
    if n_inds == 1:
        for i in range(len(meas_now)):
            if colours[i%4] == (0,0,255):
                cv2.circle(final, tuple([int(x) for x in meas_now[i]]), 5, colours[i%4], -1, cv2.LINE_AA)
    else:
        for i in range(n_inds):
            cv2.circle(final, tuple([int(x) for x in meas_now[i]]), 5, colours[i%n_inds], -1, cv2.LINE_AA)
    
    #adjust text colour to background colour
    if colour_ignored_area > (255/2):
        text_colour = (0,0,0)
    else:
        text_colour = (255,255,255)
    
    # add frame number
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(final, str(int(fr_no)), (5,30), font, 1, text_colour, 2)
        
    return final, meas_now, df
