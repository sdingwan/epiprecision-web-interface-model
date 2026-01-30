import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path
from PIL import Image
import scipy.io as sio
from typing import Tuple, List, Dict, Any
import cv2
from scipy.signal import correlate2d

minPoints = 5
epsilon = 2

# Environment-configurable paths (fallback to legacy defaults)
DEFAULT_DATA_ROOT = Path("/home/local/ASUAD/abanerj3/Desktop/expertknowledge/DATA")
ENV_DATA_ROOT = os.environ.get("VOXELINFO_DATA_ROOT")
ENV_SUBJECT_DIR = os.environ.get("VOXELINFO_SUBJECT_DIR")
ENV_REPORT_DIR = os.environ.get("VOXELINFO_REPORT_DIR")
ENV_TEMPLATE_PATH = os.environ.get("VOXELINFO_TEMPLATE_PATH")
ENV_OUTPUT_DIR = os.environ.get("VOXELINFO_OUTPUT_DIR")
ENV_WORKSPACE_NAME = os.environ.get("VOXELINFO_WORKSPACE_NAME")
ENV_LABELS_PATH = os.environ.get("VOXELINFO_LABELS_PATH")


def normxcorr2(template: np.ndarray, image: np.ndarray) -> np.ndarray:
    template = template.astype(np.float64)
    image = image.astype(np.float64)

    h_t, w_t = template.shape
    n_pix = h_t * w_t

    # Zero-mean template
    t_mean = template.mean()
    t0 = template - t_mean
    t_norm = np.sqrt(np.sum(t0 ** 2))

    # Window of ones same size as template
    ones = np.ones_like(template, dtype=np.float64)

    # Sum of image pixels under each template-sized window (full mode)
    image_sum = correlate2d(image, ones, mode='full')
    image_sq_sum = correlate2d(image ** 2, ones, mode='full')

    # Cross-correlation term between image and zero-mean template
    num = correlate2d(image, t0, mode='full')

    # Local mean of image under each window
    mu_I = image_sum / n_pix

    # Local variance term of image
    var_I = image_sq_sum - n_pix * (mu_I ** 2)
    var_I[var_I < 0] = 0  # numerical safety

    denom = np.sqrt(var_I) * t_norm
    corr = np.zeros_like(num)
    mask = denom > 0
    corr[mask] = num[mask] / denom[mask]

    return corr


def automate_slicing(main_image: np.ndarray,
                     template_image: np.ndarray,
                     threshold: float = 0.7):

    # 1. Convert to grayscale if needed
    if main_image.ndim == 3:
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_RGB2GRAY)
    else:
        main_gray = main_image.copy()

    if template_image.ndim == 3:
        tmpl_gray = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)
    else:
        tmpl_gray = template_image.copy()

    # 2. normxcorr2
    corr = normxcorr2(tmpl_gray, main_gray)

    # 3. Threshold
    rows, cols = np.where(corr > threshold)
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError(
            f"No template matches found above threshold={threshold}. "
            "Check R.png and IC_1_thresh.png or lower the threshold."
        )

    unique_rows = np.unique(rows)
    unique_cols = np.unique(cols)

    numRow = int(unique_rows.size - 1)   # exactly as MATLAB logic
    numCol = int(unique_cols.size)

    h_t, w_t = tmpl_gray.shape[:2]

    # Starting coords (0-based), direct port of MATLAB formula:
    startY = int(round(rows[0] - h_t / 3.0))
    startX = int(round(cols[0] - w_t / 3.0))

    # Tile sizes (distance between unique peaks)
    if unique_cols.size >= 2:
        sorted_cols = np.sort(unique_cols)
        sizeX = int(sorted_cols[1] - sorted_cols[0])
    else:
        sizeX = int(w_t)

    if unique_rows.size >= 2:
        sorted_rows = np.sort(unique_rows)
        sizeY = int(sorted_rows[1] - sorted_rows[0])
    else:
        sizeY = int(h_t)

    return startX, startY, sizeX, sizeY, numRow, numCol


def dbscan_custom(image: np.ndarray, min_points: int = None, eps: float = None):

    if min_points is None:
        min_points = minPoints
    if eps is None:
        eps = epsilon

    H, W = image.shape[:2]
    is_3d = (image.ndim == 3)

    # MATLAB: Points = [Points; [j i]] with i=1..H, j=1..W
    points = []
    for i0 in range(H):
        for j0 in range(W):
            if is_3d:
                if (image[i0, j0, 0] > 0) or (image[i0, j0, 1] > 0) or (image[i0, j0, 2] > 0):
                    points.append([j0 , i0])  # [x,y] 1-based
            else:
                if image[i0, j0] > 0:
                    points.append([j0, i0])

    if not points:
        return []

    Points = np.asarray(points, dtype=float)

    # ----- Core point detection (exact MATLAB logic) -----
    corePoints = []
    coreIndices = []
    for k in range(Points.shape[0]):  # MATLAB k=1..size(Points,1)
        neighbor = 0
        for j in range(Points.shape[0]):  # MATLAB j=1..size(Points,1)
            if k != j:
                euDist = np.sqrt(np.sum((Points[k, :] - Points[j, :]) ** 2))
                if euDist <= eps:
                    neighbor += 1
        if neighbor > min_points:  # MATLAB strict >
            corePoints.append(Points[k, :])
            coreIndices.append(k)

    if len(corePoints) == 0:
        return []

    corePoints = np.asarray(corePoints, dtype=float)

    # MATLAB: remPoints = Points; remPoints(coreIndices,:) = [];
    remPoints = np.delete(Points, coreIndices, axis=0)

    # MATLAB: Cluster{1} = []; clusNum=1; remID=1:size(corePoints,1)
    Cluster = []
    clusNum = 1

    remID = list(range(corePoints.shape[0]))  # 0-based IDs of remaining core points
    indexR = []

    if len(remID) > 0:
        k = remID[0]
        Cluster.append(corePoints[k:k+1, :].copy())  # Cluster{1} = corePoints(k,:)
        indexR = [0]                                 # MATLAB indexR = 1 (position in remID)
        indexClus = 0                                # MATLAB indexClus = 1

        while len(remID) > 0:
            # MATLAB: for j = 1:size(corePoints(remID,:),1)
            for jpos in range(len(remID)):
                euDist = np.sqrt(np.sum((Cluster[clusNum-1][indexClus, :] - corePoints[remID[jpos], :]) ** 2))
                if euDist != 0:
                    if euDist <= eps:
                        Cluster[clusNum-1] = np.vstack([Cluster[clusNum-1], corePoints[remID[jpos], :]])
                        indexR.append(jpos)

            # MATLAB:
            # if(size(indexR,2)<=1 && indexClus == size(Cluster{clusNum},1))
            if (len(indexR) <= 1) and (indexClus == (Cluster[clusNum-1].shape[0] - 1)):
                clusNum += 1

                # remID(indexR) = [];
                remove_positions = set(indexR)
                remID = [rid for pos, rid in enumerate(remID) if pos not in remove_positions]

                indexR = []
                if len(remID) > 0:
                    k = remID[0]
                    indexClus = 0
                    indexR = [0]
                    Cluster.append(corePoints[k:k+1, :].copy())
            else:
                # remID(indexR) = [];
                remove_positions = set(indexR)
                remID = [rid for pos, rid in enumerate(remID) if pos not in remove_positions]

                indexR = []
                indexClus += 1

    # ----- Border points attachment (exact MATLAB loop behavior) -----
    borderIndex = []
    # MATLAB: for i=1:size(remPoints,1)
    for i in range(remPoints.shape[0]):
        # MATLAB: for k=1:clusNum-1
        for k in range(clusNum - 1):
            # MATLAB: for p=1:size(Cluster{k},1)
            for p in range(Cluster[k].shape[0]):
                euDist = np.sqrt(np.sum((remPoints[i, :] - Cluster[k][p, :]) ** 2))
                if euDist <= eps:
                    Cluster[k] = np.vstack([Cluster[k], remPoints[i, :]])
                    borderIndex.append(i)
                    # MATLAB does: k=clusNum; break;  (only breaks p-loop)
                    break

    # MATLAB: remPoints(borderIndex,:) = [];
    # (Deletion is done on original indexing; duplicates don't matter.)
    if borderIndex:
        borderIndex = sorted(set(borderIndex))
        remPoints = np.delete(remPoints, borderIndex, axis=0)

    return Cluster


def cluster_detect_v1(image: np.ndarray, offset: int, end_off: int):

    H, W = image.shape[:2]

    newI = np.zeros_like(image)

    # Copy interior region (same ROI as Allzeros)
    row_start = offset - 1
    row_end_excl = H - end_off
    col_start = offset - 1
    col_end_excl = W - end_off

    if row_start < row_end_excl and col_start < col_end_excl:
        newI[row_start:row_end_excl, col_start:col_end_excl, :] = \
            image[row_start:row_end_excl, col_start:col_end_excl, :]

    newIB = newI.copy()


    for i in range(H):
        for j in range(W):
            R = int(newI[i, j, 0])
            G = int(newI[i, j, 1])
            B = int(newI[i, j, 2])

            # White / gray: R == G == B
            if (R == G) and (G == B):
                newI[i, j, :] = 0
                newIB[i, j, :] = 0
            else:
                m = max(R, B)
                # ignore blue clusters in newI
                if m == B:
                    newI[i, j, :] = 0
                # ignore red clusters in newIB
                if m == R:
                    newIB[i, j, :] = 0
    print("nonzero newI:", np.count_nonzero(np.any(newI > 0, axis=2)))
    print("nonzero newIB:", np.count_nonzero(np.any(newIB > 0, axis=2)))

    clusters1 = dbscan_custom(newI,minPoints,epsilon)
    clusters2 = dbscan_custom(newIB,minPoints,epsilon)
    clusters_all = clusters1 + clusters2
    print("clusters1:", len(clusters1), "clusters2:", len(clusters2), "total:", len(clusters1)+len(clusters2))


    return clusters_all, newI, newIB


def cluster_width_computation(newI: np.ndarray, newIB: np.ndarray, Clusters: list) -> np.ndarray:
    """
    MATLAB-faithful behavior, but 0-based indexing.
    Convention: cluster points are [x, y] = [col, row] 0-based, write as modI[y, x].
    """
    y = []
    H, W = newI.shape[:2]

    # MATLAB: modI created ONCE and reused/accumulated across clusters
    modI = np.zeros((H, W), dtype=int)

    for i in range(len(Clusters)):
        if Clusters[i] is not None and len(Clusters[i]) > 0:
            pts = np.asarray(Clusters[i], dtype=int)

            # MATLAB: modI(Clusters{i}(:,1),Clusters{i}(:,2)) = 1
            # With our 0-based convention: points are [x,y] and arrays are [y,x]
            for x_coord, y_coord in pts:
                if 0 <= y_coord < H and 0 <= x_coord < W:
                    modI[y_coord, x_coord] = 1

            # widths (same logic you already had)
            xWidth = 0
            for r in range(H):
                cols = np.where(modI[r, :] > 0)[0]
                if cols.size > 0:
                    xWidth = max(xWidth, int(cols.max() - cols.min()))

            yWidth = 0
            for c in range(W):
                rows = np.where(modI[:, c] > 0)[0]
                if rows.size > 0:
                    yWidth = max(yWidth, int(rows.max() - rows.min()))

            y.append([xWidth, yWidth])
        else:
            y.append([0, 0])

    return np.array(y, dtype=float)



def inpoly2(vert: np.ndarray, node: np.ndarray, edge: np.ndarray = None, 
            ftol: float = None) -> np.ndarray:
    """
    Point-in-polygon test using ray-casting algorithm.
    
    Args:
        vert: (N, 2) array of test points [x, y]
        node: (M, 2) array of polygon vertices [x, y]
        edge: ignored (for API compatibility)
        ftol: ignored (for API compatibility)
    
    Returns:
        (N,) boolean array - True if point is inside polygon
    """
    if vert is None or node is None:
        return np.array([], dtype=bool)
    
    pts = np.asarray(vert, dtype=float)
    poly = np.asarray(node, dtype=float)
    
    if pts.size == 0:
        return np.array([], dtype=bool)
    
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    
    if poly.size == 0 or poly.shape[0] < 3:
        return np.zeros(pts.shape[0], dtype=bool)
    
    inside = np.zeros(pts.shape[0], dtype=bool)
    
    n = poly.shape[0]
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Ray-casting algorithm
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        
        # Check if ray from point crosses this edge
        cond1 = (yi > y) != (yj > y)
        
        if np.any(cond1):
            slope = (xj - xi) / (yj - yi + 1e-12)
            x_intersect = xi + slope * (y - yi)
            cond2 = x < x_intersect
            inside[cond1 & cond2] = ~inside[cond1 & cond2]
        
        j = i
    
    return inside


def inpoly2_simple(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    
    pts = np.asarray(points, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    
    if pts.size == 0:
        return np.array([], dtype=bool)
    
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    
    if poly.shape[0] < 3:
        return np.zeros(pts.shape[0], dtype=bool)
    
    inside = np.zeros(pts.shape[0], dtype=bool)
    
    n = poly.shape[0]
    x = pts[:, 0]
    y = pts[:, 1]
    
    
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        cond1 = (yi > y) != (yj > y)
        
        if np.any(cond1):
            
            slope = (xj - xi) / (yj - yi + 1e-12)
            x_intersect = xi + slope * (y - yi)
            
            cond2 = x < x_intersect
            inside[cond1 & cond2] = ~inside[cond1 & cond2]
        
        j = i
    
    return inside

def complete_inside(B: np.ndarray, A: np.ndarray) -> int:
    
    
    if B is None or A is None:
        return 0
    if B.size == 0 or A.size == 0:
        return 0
    
    
    if B.shape[0] == 2:
        B_pts = B.T  # (N, 2)
    else:
        B_pts = B
    
    if A.shape[0] == 2:
        A_pts = A.T  # (M, 2)
    else:
        A_pts = A
    
    
    p = inpoly2(B_pts, A_pts)
    
    
    if np.sum(p) == B_pts.shape[0]:  
        return 1
    else:
        return 0

def white_matter_noise_detect(Clusters: list, image: np.ndarray, epsilon: float,
                               perc: float, im_contour: np.ndarray,
                               image_pixels: np.ndarray):
    
    y = []
    clusOverStat = []
    whitePerc = 0.0
    whiteMatter = []
    
    if Clusters is None or len(Clusters) == 0:
        return np.array(y), np.array(clusOverStat), whitePerc, whiteMatter
    
    if im_contour is None or im_contour.size == 0:
        return np.array(y), np.array(clusOverStat), whitePerc, whiteMatter
    
    # MATLAB: [M,c] = contour(rgb2gray(imContour));
    if im_contour.ndim == 3:
        gray = 0.2989 * im_contour[:,:,0] + 0.5870 * im_contour[:,:,1] + 0.1140 * im_contour[:,:,2]
    else:
        gray = im_contour.astype(float)
    
    # Get all contours with their levels
    all_contours, all_levels, level_list = extract_contours_with_levels(gray)
    
    if len(level_list) < 2:
        return np.array(y), np.array(clusOverStat), whitePerc, whiteMatter
    
    # MATLAB: c.LevelList(end-1) - second to last level
    wmLevel = level_list[-2]
    
    # Get contours at white matter level
    whiteMatter = []
    for i, level in enumerate(all_levels):
        if abs(level - wmLevel) < 1e-6:
            # Convert to (2, N) format
            whiteMatter.append(all_contours[i].T)
    
    if len(whiteMatter) == 0:
        return np.array(y), np.array(clusOverStat), whitePerc, []
    
    # Find largest white matter region
    sizeMax = 0
    maxID = 0
    for i in range(len(whiteMatter)):
        if whiteMatter[i].shape[1] > sizeMax:
            sizeMax = whiteMatter[i].shape[1]
            maxID = i
    
    # Determine which contours are inside others
    yM = np.zeros(len(whiteMatter), dtype=int)
    for i in range(len(whiteMatter)):
        for j in range(len(whiteMatter)):
            if i != j:
                if whiteMatter[i] is not None and whiteMatter[j] is not None:
                    yM[i] = complete_inside(whiteMatter[i], whiteMatter[j])
                    if yM[i] == 1:
                        break
    
    # Calculate white matter percentage
    whitePerc = 0.0
    for j in range(len(whiteMatter)):
        if yM[j] != 1:
            wm_pts = whiteMatter[j].T  # (N, 2)
            pInside = inpoly2(image_pixels, wm_pts)
            whitePerc += np.sum(pInside)
    
    # Check each cluster
    for idx in range(len(Clusters)):
        cluster = Clusters[idx]
        y_val = 0
        clusOver_val = 0.0
        
        if cluster is None or len(cluster) == 0:
            y.append(0)
            clusOverStat.append(0.0)
            continue
        
        pts = np.asarray(cluster, dtype=float)
        
        # Check against white matter regions
        if len(whiteMatter) > 0 and maxID < len(whiteMatter):
            for j in range(len(whiteMatter)):
                if yM[j] != 1:
                    wm_pts = whiteMatter[j].T
                    isInside = inpoly2(pts, wm_pts)
                    
                    if 100.0 * (np.sum(isInside) / pts.shape[0]) > perc:
                        for k_idx in range(len(whiteMatter)):
                            if yM[k_idx] == 1:
                                wm_gray_pts = whiteMatter[k_idx].T
                                isInsideG = inpoly2(pts, wm_gray_pts)
                                clusOver_val = 100.0 * np.sum(isInsideG) / pts.shape[0]
                                
                                if clusOver_val > perc:
                                    y_val = 0
                                    break
                                else:
                                    y_val = 1
                        break
        
        # Final check against largest white matter
        if maxID < len(whiteMatter):
            wm_max_pts = whiteMatter[maxID].T
            inPolyI2 = inpoly2(pts, wm_max_pts)
            clusOver_val = 100.0 * np.sum(inPolyI2) / pts.shape[0]
            
            if np.sum(inPolyI2) > (perc / 100.0) * pts.shape[0]:
                y_val = 1
        
        y.append(y_val)
        clusOverStat.append(clusOver_val)
    
    return np.array(y, dtype=float), np.array(clusOverStat, dtype=float), whitePerc, whiteMatter

def periphery_noise_detect(Clusters: list, image: np.ndarray, epsilon: float,
                           perc1: float, perc2: float, im_contour: np.ndarray,
                           num_row: int, num_col: int, tile_idx: int):
   
    y = []
    yNum = []
    OuterContour = []
    
    if Clusters is None or len(Clusters) == 0:
        return np.array(y), np.array(yNum), OuterContour
    
    if im_contour is None or im_contour.size == 0:
        return np.array(y), np.array(yNum), OuterContour
    
    if im_contour.ndim == 3:
        gray = 0.2989 * im_contour[:,:,0] + 0.5870 * im_contour[:,:,1] + 0.1140 * im_contour[:,:,2]
    else:
        gray = im_contour.astype(float)
    
    
    outerPot, level_list = get_contours_at_level(gray, target_level=None)
    
    if len(outerPot) == 0:
        return np.array(y), np.array(yNum), OuterContour
    
    k = len(outerPot)
    
    yM = np.zeros(k, dtype=int)
    for i in range(k):
        for j in range(k):
            if i != j:
                # outerPot[i] is (2, N), transpose to (N, 2) for inpoly2
                test_pts = outerPot[i].T
                poly_pts = outerPot[j].T
                p = inpoly2(test_pts, poly_pts)
                if np.sum(p) > (perc1 / 100.0) * outerPot[i].shape[1]:
                    yM[i] = 1
                    break
    
    OuterContour = []
    outerContourSize = 0
    for i in range(k):
        if yM[i] == 0:
            OuterContour.append(outerPot[i])  # Keep (2, N) format
            outerContourSize += outerPot[i].shape[1]
    
    # Process each cluster
    for idx in range(len(Clusters)):
        cluster = Clusters[idx]
        y_val = 0
        yNum_val = 0
        
        if cluster is None or len(cluster) == 0:
            y.append(0)
            yNum.append(0)
            continue
        
        pts = np.asarray(cluster, dtype=float)
        
        for k_pt in range(pts.shape[0]):
            minDist = 100.0
            for g in range(len(OuterContour)):
                oc = OuterContour[g]  # (2, N)
                for m in range(oc.shape[1]):
                    dx = oc[0, m] - pts[k_pt, 0]
                    dy = oc[1, m] - pts[k_pt, 1]
                    euDist = np.sqrt(dx*dx + dy*dy)
                    if euDist < minDist:
                        minDist = euDist
            
            if minDist < epsilon:
                y_val += 1
        
        
        trueVal = 0
        for g in range(len(OuterContour)):
            oc_pts = OuterContour[g].T  # (N, 2)
            inP = inpoly2(pts, oc_pts)
            pct_inside = 100.0 * (np.sum(inP) / pts.shape[0]) if pts.shape[0] > 0 else 0
            if pct_inside <= perc1:
                yNum_val = y_val
            else:
                yNum_val = 100.0 * y_val / pts.shape[0]
                trueVal = 1
                break
        
        if trueVal != 1:
            y_val = -1
        
        if y_val != -1:
            yNum_val = 100.0 * y_val / pts.shape[0]
        else:
            yNum_val = y_val
        
        if y_val > (perc2 / 100.0) * pts.shape[0] or y_val > (perc2 / 100.0) * outerContourSize:
            y_val = -1
            yNum_val = -1
        
        y.append(y_val)
        yNum.append(yNum_val)
    
    return np.array(y, dtype=float), np.array(yNum, dtype=float), OuterContour

def artery_noise_detect(Clusters: list, image: np.ndarray, epsilon: float,
                        perc: float, im_contour: np.ndarray) -> np.ndarray:
    
    y = np.zeros(len(Clusters), dtype=float)
    
    if Clusters is None or len(Clusters) == 0:
        return y
    
    if im_contour is None or im_contour.size == 0:
        return y
    
    # MATLAB: [M,c] = contour(rgb2gray(imContour));
    if im_contour.ndim == 3:
        gray = 0.2989 * im_contour[:,:,0] + 0.5870 * im_contour[:,:,1] + 0.1140 * im_contour[:,:,2]
    else:
        gray = im_contour.astype(float)
    
    # MATLAB: midX = floor(size(image,2)/2);
    midX = float(image.shape[1] // 2)
    
    # Get contours at first level (like MATLAB's c.LevelList(1))
    outerPot, level_list = get_contours_at_level(gray, target_level=None)
    
    if len(outerPot) == 0:
        return y
    
    k = len(outerPot)
    
    # Determine which contours are inside others
    yM = np.zeros(k, dtype=int)
    for i in range(k):
        for j in range(k):
            if i != j:
                p = inpoly2(outerPot[i].T, outerPot[j].T)
                if np.sum(p) > (perc / 100.0) * outerPot[i].shape[1]:
                    yM[i] = 1
                    break
    
    # Separate outer and inner contours
    artOuter = []
    artInner = []
    for i in range(k):
        if yM[i] == 0:
            artOuter.append(outerPot[i])
        else:
            artInner.append(outerPot[i])
    
    if len(artOuter) == 0:
        return y
    
    # Find largest outer contour
    maxSize = 0
    maxID = 0
    for i in range(len(artOuter)):
        if artOuter[i].shape[1] > maxSize:
            maxSize = artOuter[i].shape[1]
            maxID = i
    
    # Define arterial region
    artContour = artOuter[maxID]  # (2, N)
    xs = artContour[0, :]
    ys = artContour[1, :]
    
    near_mid = np.where(np.abs(xs - midX) <= 5)[0]
    if len(near_mid) > 0:
        minY = float(np.min(ys[near_mid]))
    else:
        minY = float(np.min(ys))
    
    xL = midX - 5.0
    xM = midX + 5.0
    yL = minY
    yM_val = minY + 25.0
    arterialRegion = np.array([[xL, yL], [xL, yM_val], [xM, yM_val], [xM, yL]])
    
    # Check each cluster
    for idx in range(len(Clusters)):
        cluster = Clusters[idx]
        
        if cluster is None or len(cluster) == 0:
            continue
        
        pts = np.asarray(cluster, dtype=float)
        y_val = 0.0
        y0 = 0.0
        yI = 0
        
        # Check inner contours
        for mp in range(len(artInner)):
            inner = artInner[mp]
            if inner is not None and inner.shape[1] > 0:
                p = inpoly2(pts, inner.T)
                pI = inpoly2(inner.T, pts)
                y_val += np.sum(p)
                if np.sum(pI) > (perc / 100.0) * inner.shape[1]:
                    yI = 1000
        
        # Check outer contours
        for mp in range(len(artOuter)):
            outer = artOuter[mp]
            if outer is not None and outer.shape[1] > 0:
                p = inpoly2(pts, outer.T)
                y0 += np.sum(p)
        
        # Final decision
        if (y_val > (perc / 100.0) * pts.shape[0] or 
            y0 < ((1 - (perc / 100.0)) * pts.shape[0]) or 
            yI == 1000):
            y[idx] = -1
        else:
            y[idx] = 0
    
    return y


def extract_contours_with_levels(gray: np.ndarray):
    """
    Extract contours using threshold-based binary mask approach.
    This mimics MATLAB's contour behavior more accurately.
    
    Returns:
        all_contours: list of (N, 2) arrays with [x, y] coordinates
        all_levels: list of level values for each contour
        level_list: array of unique levels
    """
    gray = np.asarray(gray, dtype=np.float64)
    
    if gray.size == 0 or np.max(gray) == 0:
        return [], [], np.array([])
    
    # Convert to uint8 for cv2
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Use multiple threshold levels to mimic MATLAB's contour behavior
    max_val = np.max(gray_uint8[gray_uint8 > 0]) if np.any(gray_uint8 > 0) else 255
    levels = np.array([1, max_val // 4, max_val // 2, 3 * max_val // 4], dtype=float)
    levels = levels[levels > 0]
    levels = np.unique(levels)
    
    all_contours = []
    all_levels = []
    
    for level in levels:
        # Binary threshold at this level
        _, mask = cv2.threshold(gray_uint8, int(level), 255, cv2.THRESH_BINARY)
        
        # Find contours - RETR_LIST gets all contours (both outer and inner)
        contours_info = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_info) == 3:
            _, contours, _ = contours_info
        else:
            contours, _ = contours_info
        
        for contour in contours:
            contour = contour.reshape(-1, 2)
            if contour.shape[0] >= 3:
                # contour is [x, y] format from OpenCV
                all_contours.append(contour.astype(float))
                all_levels.append(float(level))
    
    level_list = np.unique(levels)
    
    return all_contours, all_levels, level_list





def get_contours_at_level(gray: np.ndarray, target_level: float = None):
    """
    Extract outer contours from grayscale image using threshold.
    Returns contours in (2, N) format like MATLAB.
    
    For periphery/artery detection, we want the OUTER brain boundary,
    so we use RETR_EXTERNAL to get only outermost contours.
    """
    gray = np.asarray(gray, dtype=np.float64)
    
    if gray.size == 0 or np.max(gray) == 0:
        return [], np.array([])
    
    # Convert to uint8
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Use low threshold to get brain boundary (threshold=1 catches any non-zero pixel)
    threshold_val = 1 if target_level is None else int(target_level)
    _, mask = cv2.threshold(gray_uint8, threshold_val, 255, cv2.THRESH_BINARY)
    
    # RETR_EXTERNAL gets only outermost contours (brain boundary)
    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_info) == 3:
        _, contours, _ = contours_info
    else:
        contours, _ = contours_info
    
    result = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if contour.shape[0] >= 3:
            # Transpose to (2, N) format: row 0 = x coords, row 1 = y coords
            result.append(contour.astype(float).T)
    
    # Level list for compatibility
    level_list = np.array([float(threshold_val)])
    
    return result, level_list




def cluster_detect(image: np.ndarray, offset: int, end_off: int, 
                   im_contour: np.ndarray, image_pixels: np.ndarray,
                   num_row: int, num_col: int, index: int, size_im: np.ndarray):
    
    factor_m = 1.0  
    
    H, W = image.shape[:2]
    newI = np.zeros_like(image)
    
    
    row_start = offset - 1                    
    row_end = H - end_off                     
    col_start = offset - 1
    col_end = W - end_off
    
    for i in range(row_start, row_end):       
        for j in range(col_start, col_end):   
            newI[i, j, :] = image[i, j, :]    
    
    
    newIB = newI.copy()
    
    
    for i in range(H):                        
        for j in range(W):                   
            
            R = int(newI[i, j, 0])           
            G = int(newI[i, j, 1])         
            B = int(newI[i, j, 2])           
            
            if (R == G) and (G == B):        
                
                newI[i, j, :] = [0, 0, 0]
                newIB[i, j, :] = [0, 0, 0]
            else:
                
                if max(R, B) == B:            
                    newI[i, j, :] = [0, 0, 0]
                
                
                R_IB = int(newIB[i, j, 0])
                B_IB = int(newIB[i, j, 2])
                if max(R_IB, B_IB) == R_IB:  
                    newIB[i, j, :] = [0, 0, 0]
    
    
    Clusters = dbscan_custom(newI)
    Clusters2 = dbscan_custom(newIB)
    Clusters = Clusters + Clusters2           # Concatenate lists
    
    factorX = 1.0
    maxThresh = min((50.0 / factorX), 90.0)
    maxThresh2 = 20.0 / factorX
    thresh2 = 99.0
    threshNew = 60.0
    
    peri, periNum, OuterContour = periphery_noise_detect(
        Clusters, image, 2, threshNew, thresh2, im_contour, num_row, num_col, index
    )
    
    
    grayInside = 0
    for ko in range(len(OuterContour)):       # MATLAB: 1:size(OuterContour,2)
        
        oc = OuterContour[ko]
        if oc.shape[0] == 2:                  # If (2, N) format
            oc_pts = oc.T                     # Transpose to (N, 2)
        else:
            oc_pts = oc
        pInside = inpoly2(image_pixels, oc_pts)
        grayInside = grayInside + np.sum(pInside)
    
    
    art = artery_noise_detect(Clusters, image, 2, 40.0 / factorX, im_contour)
    
    whiteM, clusOverStat, whitePerc, _ = white_matter_noise_detect(
        Clusters, image, epsilon, 40.0 / factorX, im_contour, image_pixels
    )
    
    if grayInside > 0:
        percWhite = 100.0 * whitePerc / grayInside
    else:
        percWhite = 0.0  # Avoid division by zero
    
    
    maxID = 0
    maxSize = 0
    for i in range(len(Clusters)):            
        if Clusters[i] is not None and len(Clusters[i]) > 0:
            cluster_size = Clusters[i].shape[0]  
            if maxSize < cluster_size:
                maxSize = cluster_size
                maxID = i
    clusterWidth = cluster_width_computation(newI, newIB, Clusters)
    print(Clusters, clusterWidth, peri, periNum, art, whiteM, maxID, clusOverStat, percWhite)
    return Clusters, clusterWidth, peri, periNum, art, whiteM, maxID, clusOverStat, percWhite



def all_zeros(image: np.ndarray, offset: int, end_off: int) -> int:

    H, W = image.shape[:2]

    row_start = offset - 1              # inclusive
    row_end_excl = H - end_off          # exclusive
    col_start = offset - 1
    col_end_excl = W - end_off

    if row_start >= row_end_excl or col_start >= col_end_excl:
        # empty ROI => all zeros
        return 1

    if image.ndim == 3:
        region_sum = np.sum(image[row_start:row_end_excl,
                                  col_start:col_end_excl, :])
    else:
        region_sum = np.sum(image[row_start:row_end_excl,
                                  col_start:col_end_excl])

    return 1 if region_sum == 0 else 0


def remove_r(image: np.ndarray, offset: int, end_off: int) -> np.ndarray:
    y = image.copy()
    H, W = y.shape[:2]

    for i in range(offset):
        for j in range(offset):
            y[i, j, :] = 0

    row_start = max(H - end_off - 1, 0)
    for i in range(row_start, H):
        for j in range(offset):
            y[i, j, :] = 0

    return y


import numpy as np


def color_detection(im_cropped: np.ndarray, voxel_size: int = 3) -> np.ndarray:
    
    if im_cropped is None or im_cropped.size == 0:
        return np.array([], dtype=int)
    
    if im_cropped.ndim != 3 or im_cropped.shape[2] != 3:
        raise ValueError("color_detection expects an RGB image (H, W, 3)")
    
    H, W = im_cropped.shape[:2]
    
    
    num_rows = H // voxel_size  # Number of complete voxel rows
    num_cols = W // voxel_size  # Number of complete voxel columns
    
    if num_rows == 0 or num_cols == 0:
        return np.array([], dtype=int)
    
    VR = np.zeros((num_rows, num_cols), dtype=int)
    
    for j in range(num_cols):  # Column index
        startX = j * voxel_size
        endX = startX + voxel_size
        
        for i in range(num_rows):  # Row index
            startY = i * voxel_size
            endY = startY + voxel_size
            
            # Extract tile
            tile = im_cropped[startY:endY, startX:endX, :]
            
            
            red_mask = (
                (tile[:, :, 0] >= 130) &  # R >= 130
                (tile[:, :, 1] <= 60) &   # G <= 60
                (tile[:, :, 2] <= 100)    # B <= 100
            )
            
            # MATLAB: VR(i,j) = size(G,1) where [G,H] = find(mask)
            VR[i, j] = int(np.sum(red_mask))
    
    return VR

import numpy as np


def compute_assymetry(VR: np.ndarray) -> float:
    
    if VR is None or VR.size == 0:
        return 1000.0
    
    # Ensure 2D array
    if VR.ndim == 1:
        VR = VR.reshape(1, -1)
    
    num_cols = VR.shape[1]
    midPoint = (num_cols // 2) + 1  
    
    first_val = VR[0, 0]  
    if np.sum(VR - first_val) == 0:
        return 1000.0
    
    
    y = 0.0
    num_rows = VR.shape[0]
    
    for i in range(num_rows):
        
        left_end = midPoint - 1  
        right_start = midPoint   
        
        left_half = VR[i, 0:left_end]          
        right_half = VR[i, right_start:]       
        min_len = min(len(left_half), len(right_half))
        
        if min_len > 0:
            
            assymDiff = np.abs(left_half[:min_len] - right_half[:min_len])
            y += float(np.sum(assymDiff))
    
    return y

import numpy as np
from typing import Tuple


def compute_confusion_metrics(num: np.ndarray, labelV: np.ndarray) -> Tuple[float, float, float, float, float]:
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    num_flat = np.asarray(num).flatten()
    labelV_flat = np.asarray(labelV).flatten()
    
    for km in range(len(num_flat)):
        
        if num_flat[km] != 2:
            
            if num_flat[km] >= 1 and labelV_flat[km] == 1:
                TP += 1
            
            elif num_flat[km] >= 1 and labelV_flat[km] == 0:
                FN += 1
            
            elif num_flat[km] == 0 and labelV_flat[km] == 1:
                FP += 1
            
            else:
                TN += 1
    
    
    if (TP + FP) > 0:
        Precision = TP / (TP + FP)
    else:
        Precision = 0.0
    
    
    if (TP + FN) > 0:
        Recall = TP / (TP + FN)
    else:
        Recall = 0.0
    
    if Precision > 0 and Recall > 0:
        F1Score = 2.0 * (1.0 / (1.0 / Precision + 1.0 / Recall))
       
    else:
        F1Score = 0.0
    
    total = TP + TN + FP + FN
    if total > 0:
        Accuracy = (TP + TN) / total
    else:
        Accuracy = 0.0
    
   
    if (TN + FP) > 0:
        Specificity = TN / (TN + FP)
    else:
        Specificity = 0.0
    
    return Precision, Recall, F1Score, Accuracy, Specificity

def crop_image(im: np.ndarray, start_x: int, start_y: int, 
               size_x: int, size_y: int) -> np.ndarray:
    """Crop image region."""
    return im[start_y:start_y + size_y, start_x:start_x + size_x]


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_patient() -> Dict[str, Any]:
    """
    Main processing function for extracting voxel information.
    
    Returns:
        Dictionary with all results
    """
    # Global parameters (from original MATLAB)
    data_root = Path(ENV_DATA_ROOT) if ENV_DATA_ROOT else DEFAULT_DATA_ROOT
    output_dir = Path(ENV_OUTPUT_DIR) if ENV_OUTPUT_DIR else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subject folder
    if ENV_SUBJECT_DIR:
        subject_dir = Path(ENV_SUBJECT_DIR)
        if not subject_dir.exists():
            raise ValueError(f"Subject folder not found: {subject_dir}")
    else:
        candidates = sorted(data_root.glob("ASUAI_0*"))
        if len(candidates) == 0:
            raise ValueError(f"No patient folder found under {data_root}")
        subject_dir = candidates[0]

    folder_name = subject_dir.name
    print(f"Processing folder: {folder_name}", flush=True)
    
    # factorM = 1 for most patients (exceptions were 15, 22, 31 in multi-patient version)
    factorM = 1
    
    # Resolve report folder
    if ENV_REPORT_DIR:
        report_dir = Path(ENV_REPORT_DIR)
    else:
        report_dir = subject_dir / "MO" / "report"
        if not report_dir.exists():
            report_dir = subject_dir / "report"
    if not report_dir.exists():
        raise ValueError(f"Report folder not found: {report_dir}")

    # Read first threshold image for scaling
    imgName = 'IC_1_thresh.png'
    img_path = report_dir / imgName
    if not img_path.exists():
        thresh_candidates = sorted(report_dir.glob("*_thresh*"))
        if not thresh_candidates:
            raise ValueError(f"No threshold images found in {report_dir}")
        img_path = thresh_candidates[0]
    im = np.array(Image.open(img_path))

    # Read template image
    template_path = Path(ENV_TEMPLATE_PATH) if ENV_TEMPLATE_PATH else (data_root / "R.png")
    if not template_path.exists():
        raise FileNotFoundError(f"Template image not found: {template_path}")
    templateImage = np.array(Image.open(template_path))
    
    # Compute slicing parameters
    X, Y, size_X, size_Y, num_Row, num_Col = automate_slicing(im, templateImage)
    Scalings = np.array([X, Y, size_X, size_Y])
    sio.savemat(str(output_dir / 'scalings.mat'), {'NewScalings': Scalings})
    
    numRow = num_Row
    numCol = num_Col
    
    # Auto-detect: compute contours if file doesn't exist, otherwise load it
    python_contour_path = str(output_dir / f'ContourImagesP{folder_name}Init_python.mat')
    if os.path.exists(python_contour_path):
        need = 0  # Load existing contour file
        print(f"Found existing contour file: {python_contour_path}", flush=True)
    else:
        need = 1  # Compute contours since file doesn't exist
        print(f"Contour file not found: {python_contour_path}. Will compute contours.", flush=True)
    
    Offset = 1
    endOff = 6
    problem = [8, 9]
    
    # Get files
    files = sorted(str(p) for p in report_dir.glob("*_thresh*"))
    n = len(files)
    listing = sorted(str(p) for p in report_dir.glob("t*.txt"))
    
    sigLength = 295
    freq = np.arange(1, sigLength + 1) * 0.08475
    tThresh = 70
    fThresh = 15
    sThresh = int(np.ceil(fThresh / 0.08475))
    closenessCluster = 20
    
    minClusterIndex = np.zeros(numCol * numRow, dtype=int)
    maxClusterSizeI = 1000 * np.ones(numCol * numRow)
    
    imContour = {}
    
    # =========================================================================
    # FIRST PHASE: Initial contour detection
    # =========================================================================
    if need:
        print(f"First pass: processing {n} files...", flush=True)
        # First pass: find minimum cluster sizes
        for fileNum in range(n):
            icaData_temp = np.loadtxt(listing[fileNum])
            filename = os.path.basename(files[fileNum])
            fileNameTemp = int(filename.split('_')[1])
            
            if fileNameTemp == 18:
                print('here')
            
            im = np.array(Image.open(files[fileNum]))
            
            startX = int(Scalings[0])
            s_value = startX
            startY = int(Scalings[1])
            sizeX = int(Scalings[2])
            sizeY = int(Scalings[3])
            BrainSize = 1200000 ** (1/3)
            VoxelSize = 3
            
            imCropped = {}
            k = 0
            fileInfo = []
            
            for ii in range(numRow):
                for jj in range(numCol):
                    imContour[k] = None
                    imCropped[k] = crop_image(im, startX, startY, sizeX, sizeY)
                    fileInfo.append([ii, jj, fileNum])
                    startX += sizeX
                    k += 1
                startY += sizeY
                startX = s_value
            
            Offset = 1
            endOff = 6
            sizeLimit = 5 * VoxelSize
            pixelLimit = VoxelSize * VoxelSize * 15
            mismatchFactor = 1.3
            
            for i in range(len(imCropped)):
                if imCropped[i] is not None and len(imCropped[i]) > 0:
                    if not all_zeros(imCropped[i], Offset, endOff):
                        Clusters_temp, newI_temp, newIB_temp = cluster_detect_v1(imCropped[i], Offset, endOff)
                        
                        maxClusterSize = 0
                        for p in range(len(Clusters_temp)):
                            if maxClusterSize <= len(Clusters_temp[p]):
                                maxClusterSize = len(Clusters_temp[p])
                        
                        if maxClusterSizeI[i] > maxClusterSize:
                            maxClusterSizeI[i] = maxClusterSize
                            minClusterIndex[i] = fileNum
        
        print("Second pass: extracting contours...", flush=True)
        # Second pass: extract contour images
        for fileNum in range(n):
            icaData_temp = np.loadtxt(listing[fileNum])
            filename = os.path.basename(files[fileNum])
            fileNameTemp = int(filename.split('_')[1])
            

            im = np.array(Image.open(files[fileNum]))
            
            startX = int(Scalings[0])
            s_value = startX
            startY = int(Scalings[1])
            sizeX = int(Scalings[2])
            sizeY = int(Scalings[3])
            BrainSize = 1200000 ** (1/3)
            VoxelSize = 3
            
            imCropped = {}
            k = 0
            fileInfo = []
            
            for ii in range(numRow):
                for jj in range(numCol):
                    imCropped[k] = crop_image(im, startX, startY, sizeX, sizeY)
                    fileInfo.append([ii, jj, fileNum])
                    startX += sizeX
                    k += 1
                startY += sizeY
                startX = s_value
            
            Offset = 6
            endOff = 5
            sizeLimit = 5 * VoxelSize
            pixelLimit = VoxelSize * VoxelSize * 15
            mismatchFactor = 1.3
            
            for i in range(len(imCropped)):
                if imCropped[i] is not None and len(imCropped[i]) > 0:
                    if not all_zeros(imCropped[i], Offset, endOff):
                        if minClusterIndex[i] == fileNum:
                            imContour[i] = imCropped[i]
        
        # Save contour images - convert dict to object array for MATLAB compatibility
        num_tiles = numRow * numCol
        imContour_array = np.empty((1, num_tiles), dtype=object)
        for i in range(num_tiles):
            # scipy.io.savemat cannot serialize None, use empty array instead
            imContour_array[0, i] = imContour.get(i) if imContour.get(i) is not None else np.empty((0, 0))
        contour_path = output_dir / f'Contour{folder_name}.mat'
        sio.savemat(str(contour_path), {'imContour': imContour_array})
    else:
        # Load existing Python contour file
        print(f"Loading Python contour file: {python_contour_path}", flush=True)
        data = sio.loadmat(python_contour_path)
        print(f"Loaded contour file successfully", flush=True)
            
        imContour_loaded = data['imContour']
        imContour = {}
        for i in range(imContour_loaded.shape[1]):
            item = imContour_loaded[0, i]
            if item is not None and hasattr(item, 'size') and item.size > 0:
                imContour[i] = item
            else:
                imContour[i] = None
    
    # Remove R from contour images
    Offset = 6
    endOff = 5
    for i in range(numRow * numCol):
        if i in imContour and imContour[i] is not None:
            imContour[i] = remove_r(imContour[i], Offset, endOff)
    
    # =========================================================================
    # SECOND PHASE: After contour images are produced
    # =========================================================================
    print(f"Contour images loaded: {sum(1 for v in imContour.values() if v is not None)} non-empty tiles", flush=True)
    
    # Build image pixels array
    imagePixels = []
    if 20 in imContour and imContour[20] is not None:
        for j in range(imContour[20].shape[0]):
            for i in range(imContour[20].shape[1]):
                imagePixels.append([i, j])
    imagePixels = np.array(imagePixels)
    print(f"Image pixels array: {imagePixels.shape}", flush=True)
    
    # Storage
    icaData = {}
    maxScore = {}
    sigEnergy = {}
    energyPercent = np.zeros(n)
    classSign = np.zeros(n)
    filenameString = []
    label = np.zeros(n)
    label2 = np.zeros(n)
    label3 = np.zeros(n)
    label4 = np.zeros(n)
    label5 = np.zeros(n)
    cvFile = np.zeros(n)
    ClusterSizePat = {}
    OverallPeriFile = {}
    OverallArtFile = {}
    OverallWhiteFile = {}
    percWhiteArrayFile = {}
    percBrainArrayFile = {}
    
    # Process each IC file
    print(f"Starting main processing loop for {n} files...", flush=True)
    for fileNum in range(n):
        if fileNum % 10 == 0:
            print(f"Processing file {fileNum + 1}/{n}", flush=True)
        
        icaData[fileNum] = np.loadtxt(listing[fileNum])
        filename = os.path.basename(files[fileNum])
        fileNameTemp = int(filename.split('_')[1])
        
        (G_prob,) = np.where(np.array(problem) == fileNameTemp)
        if len(G_prob) > 0:
            print('Here: Problem found')
        
        im = np.array(Image.open(files[fileNum]))
        
        startX = int(Scalings[0])
        s_value = startX
        startY = int(Scalings[1])
        sizeX = int(Scalings[2])
        sizeY = int(Scalings[3])
        BrainSize = 1200000 ** (1/3)
        VoxelSize = 3
        
        imCropped = {}
        sizeIM = {}
        k = 0
        fileInfo = []
        
        for ii in range(numRow):
            for jj in range(numCol):
                imCropped[k] = crop_image(im, startX, startY, sizeX, sizeY)
                gray = np.mean(imCropped[k], axis=2) if len(imCropped[k].shape) == 3 else imCropped[k]
                G_temp, H_temp = np.where(gray > 0)
                sizeIM[k] = max(len(G_temp), 1)
                fileInfo.append([ii, jj, fileNum])
                startX += sizeX
                k += 1
            startY += sizeY
            startX = s_value
        
        # Image Processing
        Offset = 6
        endOff = 5
        sizeLimit = 5 * VoxelSize
        pixelLimit = factorM * VoxelSize * VoxelSize * 15
        mismatchFactor = 1.3
        
        numCluster = np.zeros(len(imCropped))
        sizeMatch = np.zeros(len(imCropped))
        bigClusterID = []
        clusterInfo = np.zeros((len(imCropped), 4))
        clusterInfo2 = {}
        Clusters = {}
        clusOverStat = {}
        peri = {}
        periNum = {}
        periMax = np.zeros(len(imCropped))
        ClusterWidth = {}
        art = {}
        artMax = np.zeros(len(imCropped))
        whiteM = {}
        maxID = np.zeros(len(imCropped), dtype=int)
        percWhite = np.zeros(len(imCropped))
        ClusterCentroid = np.zeros((len(imCropped), 2))
        majorClusID = []
        
        for i in range(len(imCropped)):
            clusterInfo2[i] = []
            clusterInfo[i, :] = [0, 0, 0, 0]
            
            
            numCluster[i] = 0
            sizeMatch[i] = 0
            
            if imCropped[i] is not None and len(imCropped[i]) > 0:
                if not all_zeros(imCropped[i], Offset, endOff):
                    imCropped[i] = remove_r(imCropped[i], Offset, endOff)
                    
                    im_contour_i = imContour.get(i, None)
                    
                    (Clusters[i], ClusterWidth[i], peri[i], periNum[i], 
                     art[i], whiteM[i], maxID[i], clusOverStat[i], 
                     percWhite[i]) = cluster_detect(
                        imCropped[i], Offset, endOff,
                        im_contour_i, imagePixels, numRow, numCol, i,
                        np.array(list(sizeIM.values()))
                    )
                    print(Clusters[i], ClusterWidth[i], peri[i], periNum[i], 
                     art[i], whiteM[i], maxID[i], clusOverStat[i], 
                     percWhite[i])
                    
                    ClusterCentroid[i, :] = [0, 0]
                    periMax[i] = 0
                    artMax[i] = 0
                    
                    if maxID[i] != 0:
                        if len(Clusters[i][maxID[i]]) >= pixelLimit:
                            ClusterCentroid[i, 0] = np.mean([c[0] for c in Clusters[i][maxID[i]]])
                            ClusterCentroid[i, 1] = np.mean([c[1] for c in Clusters[i][maxID[i]]])
                            periMax[i] = peri[i][maxID[i]]
                            artMax[i] = art[i][maxID[i]]
                    
                    maxSizeClus = 0
                    maxClusID = 0
                    
                    for kM in range(len(Clusters[i])):
                        if len(Clusters[i][kM]) >= pixelLimit and 8 <= i <= 36:
                            numCluster[i] += 1
                            clusterInfo2[i].append([
                                len(Clusters[i][kM]),
                                peri[i][kM],
                                art[i][kM],
                                whiteM[i][kM]
                            ])
                            if maxSizeClus < len(Clusters[i][kM]):
                                maxSizeClus = len(Clusters[i][kM])
                                maxClusID = kM
                    
                    if len(Clusters[i]) > 0:
                        kM = len(Clusters[i]) - 1
                        clusterInfo[i, :] = [
                            len(Clusters[i][kM]),
                            peri[i][kM],
                            art[i][kM],
                            whiteM[i][kM]
                        ]
                    
                    for kM in range(len(Clusters[i])):
                        if len(Clusters[i][kM]) >= pixelLimit and 8 <= i <= 36:
                            majorClusID.append(i)
                            if ClusterWidth[i][kM][0] > sizeLimit or ClusterWidth[i][kM][1] > sizeLimit:
                                if peri[i][kM] == -1:
                                    sizeMatch[i] = -1
                                    bigClusterID.append(0)
                                    break
                                else:
                                    if art[i][kM] == -1:
                                        sizeMatch[i] = -1
                                        break
                                    else:
                                        if whiteM[i][kM] == 1:
                                            sizeMatch[i] = -1
                                            break
                                        else:
                                            sizeMatch[i] = 1
                                            bigClusterID.append(1)
        
        (G,) = np.where(numCluster > 0)
        (G,) = np.where((numCluster > 0) & (numCluster <= 3))
        (G4,) = np.where(sizeMatch > 0)
        (G1,) = np.where(sizeMatch == -1)
        
        tThresh = 90
        label[fileNum] = 0
        label2[fileNum] = 0
        label3[fileNum] = 0
        label4[fileNum] = 0
        label5[fileNum] = 0
        cvFile[fileNum] = np.std(icaData[fileNum]) / np.mean(icaData[fileNum]) if np.mean(icaData[fileNum]) != 0 else 0
        arteryCVThresh = 2.5
        
        G_max, H_max = np.max(clusterInfo[:, 0]), np.argmax(clusterInfo[:, 0])
        (G3,) = np.where(clusterInfo[:, 1] == -1)
        
        if clusterInfo[H_max, 1] == -1:
            label3[fileNum] = 0
        elif clusterInfo[H_max, 2] == -1:
            label3[fileNum] = 0
        elif clusterInfo[H_max, 3] == 1:
            label3[fileNum] = 0
        else:
            label3[fileNum] = 1
        
        if len(G4) > 0:
            label2[fileNum] = 1
            label4[fileNum] = 1
            if len(G) >= len(G1):
                label[fileNum] = 1
            else:
                label[fileNum] = 0
            if len(G3) > 0:
                label4[fileNum] = 0
        
        # Aggregate cluster information
        k = 1
        OverallClusters = []
        ClusterSize = []
        OverallPeri = []
        OverallArt = []
        OverallWhite = []
        OverallWhiteOverlap = []
        OverallPeriNum = []
        percBrainArray = []
        percWhiteArray = []
        
        max_sizeIM = max(sizeIM.values()) if sizeIM else 1
        
        # Iterate over all tiles (0 to numRow*numCol-1), not just len(Clusters)
        for i in range(numRow * numCol):
            if i in Clusters and Clusters[i] is not None and len(Clusters[i]) > 0:
                OverallClusters.extend(Clusters[i])
                for lk in range(len(Clusters[i])):
                    ClusterSize.append(len(Clusters[i][lk]))
                OverallPeri.extend(peri[i])
                OverallPeriNum.extend(periNum[i])
                OverallArt.extend(art[i])
                OverallWhite.extend(whiteM[i])
                OverallWhiteOverlap.extend(clusOverStat[i])
                percWhiteArray.extend([percWhite[i]] * len(Clusters[i]))
                percBrainArray.extend([100 * (sizeIM[i] / max_sizeIM)] * len(Clusters[i]))
        
        ClusterSize = np.array(ClusterSize)
        OverallPeri = np.array(OverallPeri)
        OverallArt = np.array(OverallArt)
        OverallWhite = np.array(OverallWhite)
        percBrainArray = np.array(percBrainArray)
        percWhiteArray = np.array(percWhiteArray)
        
        if len(ClusterSize) > 0:
            B = np.sort(ClusterSize)[::-1]
            I = np.argsort(ClusterSize)[::-1]
            
            ClusterSizePat[fileNum] = ClusterSize
            OverallPeriFile[fileNum] = OverallPeri
            OverallArtFile[fileNum] = OverallArt
            OverallWhiteFile[fileNum] = OverallWhite
            percWhiteArrayFile[fileNum] = percWhiteArray
            percBrainArrayFile[fileNum] = percBrainArray
            
            majorityDec = []
            lk = 0
            while lk < len(ClusterSize):
                if ClusterSize[I[lk]] >= pixelLimit:
                    if (OverallPeri[I[lk]] != -1 and OverallArt[I[lk]] != -1 and 
                        OverallWhite[I[lk]] != 1 and percBrainArray[I[lk]] > 10):
                        majorityDec.append(1)
                    elif (OverallPeri[I[lk]] != -1 and OverallArt[I[lk]] != -1 and 
                          OverallWhite[I[lk]] == 1 and percWhiteArray[I[lk]] <= 10):
                        majorityDec.append(0)
                    else:
                        majorityDec.append(-1)
                else:
                    break
                lk += 1
            
            majorityDec = np.array(majorityDec)
            
            (G_md,) = np.where(majorityDec == 1)
            (G2_md,) = np.where(majorityDec != 0)
            
            scoreM = np.sum(ClusterSize[I[:len(majorityDec)]] * majorityDec) / np.mean(ClusterSize) if np.mean(ClusterSize) != 0 else 0
            ClusterNumLimit = 3
            (GSize,) = np.where(ClusterSize > pixelLimit)
            
            if len(G2_md) > 0 and len(G_md) / len(G2_md) > 0.5:
                label5[fileNum] = 1
            else:
                label5[fileNum] = 0
        
        # Compute voxel representation and scores
        VR = {}
        score = np.zeros(numRow * numCol)
        scoreCluster = np.zeros(numRow * numCol)
        scoreAssym = np.zeros(numRow * numCol)
        assymScore = np.zeros(numRow * numCol)
        
        for i in range(numRow * numCol):
            if i in imCropped and imCropped[i] is not None:
                if imCropped[i].shape[0] < 160 or imCropped[i].shape[1] < 160:
                    VR[i] = None
                else:
                    VR[i] = color_detection(imCropped[i], VoxelSize)
                    
                    dataOrganization = []
                    if VR[i] is not None:
                        for k_vr in range(VR[i].shape[0]):
                            for j_vr in range(VR[i].shape[1]):
                                if VR[i][k_vr, j_vr] != 0:
                                    dataOrganization.append([k_vr, j_vr, VR[i][k_vr, j_vr]])
                    
                    if len(dataOrganization) > 0:
                        dataOrganization = np.array(dataOrganization)
                        from sklearn.cluster import DBSCAN
                        IDX_db = DBSCAN(eps=2, min_samples=5).fit(dataOrganization)
                        A = len(set(IDX_db.labels_)) - (1 if -1 in IDX_db.labels_ else 0)
                        
                        mu = 2
                        sigma = 2
                        x = A
                        scoreCluster[i] = np.exp(-(x - mu)**2 / sigma**2)
                    else:
                        scoreCluster[i] = 0
                    
                    if VR[i] is not None:
                        assymScore[i] = compute_assymetry(VR[i])
                    stdAssymScore = 40
                    scoreAssym[i] = np.exp(-assymScore[i]**2 / stdAssymScore**2)
                    score[i] = 0.5 * scoreCluster[i] + 0.5 * scoreAssym[i]
        
        # Compute signal energy
        for j in range(sigLength):
            if fileNum not in sigEnergy:
                sigEnergy[fileNum] = np.zeros(sigLength)
            total_sum = np.sum(icaData[fileNum])
            if total_sum != 0:
                sigEnergy[fileNum][j] = 100 * np.sum(icaData[fileNum][:j+1]) / total_sum
        
        stdThresh = 17.7
        energyPercent[fileNum] = sigEnergy[fileNum][sThresh - 1] if sThresh <= sigLength else 0
        classSign[fileNum] = 2 - 1 / np.exp((sigEnergy[fileNum][sThresh - 1] - tThresh) / stdThresh) if sThresh <= sigLength else 0
        
        # Max scores
        top3_cluster = np.sort(scoreCluster)[-3:][::-1]
        top3_assym = np.sort(assymScore)[:3]
        maxScore[fileNum] = np.concatenate([top3_cluster, top3_assym, [sigEnergy[fileNum][sThresh - 1] if sThresh <= sigLength else 0]])
        
        (G_score,) = np.where(score > 0.8)
        filenameString.append(int(filename.split('_')[1]))
    
    # Sort by filename
    filenameString = np.array(filenameString)
    B_fn, I_fn = np.sort(filenameString), np.argsort(filenameString)
    
    labelV = label[I_fn]
    labelV2 = label2[I_fn]
    labelV3 = label3[I_fn]
    labelV4 = label4[I_fn]
    labelV5 = label5[I_fn]
    
    # Create tables (store as dict for Python)
    T = {'filenameString': filenameString, 'label': label}
    T2 = {'filenameString': filenameString, 'label2': label2}
    T3 = {'filenameString': filenameString, 'label3': label3}
    T4 = {'filenameString': filenameString, 'label4': label4}
    T5 = {'filenameString': filenameString, 'label5': label5}
    
    # Summary of labels
    print(f"\n=== Processing Complete ===", flush=True)
    print(f"Total files processed: {n}", flush=True)
    print(f"Label V1 summary: {np.sum(labelV == 1)} signal, {np.sum(labelV == 0)} noise", flush=True)
    print(f"Label V5 summary: {np.sum(labelV5 == 1)} signal, {np.sum(labelV5 == 0)} noise", flush=True)
    
    # Load ground truth
    labels_path = ENV_LABELS_PATH or str(data_root / "labels" / f"{folder_name}.csv")
    
    try:
        data = np.genfromtxt(labels_path, delimiter=',', skip_header=1)
        num = data[:, 1] if data.ndim > 1 else data
        
        # Compute metrics
        Precision1, Recall1, F1Score1, Accuracy1,_ = compute_confusion_metrics(num, labelV)
        Precision2, Recall2, F1Score2, Accuracy2,_ = compute_confusion_metrics(num, labelV2)
        Precision3, Recall3, F1Score3, Accuracy3,_ = compute_confusion_metrics(num, labelV3)
        Precision4, Recall4, F1Score4, Accuracy4,_ = compute_confusion_metrics(num, labelV4)
        Precision5, Recall5, F1Score5, Accuracy5,_ = compute_confusion_metrics(num, labelV5)
        
        print(f'Precision V1 = {Precision1}')
        print(f'Recall V1 = {Recall1}')
        print(f'F1Score V1 = {F1Score1}')
        
        print(f'Precision V2 = {Precision2}')
        print(f'Recall V2 = {Recall2}')
        print(f'F1Score V2 = {F1Score2}')
        
        print(f'Precision V3 = {Precision3}')
        print(f'Recall V3 = {Recall3}')
        print(f'F1Score V3 = {F1Score3}')
        
        print(f'Precision V4 = {Precision4}')
        print(f'Recall V4 = {Recall4}')
        print(f'F1Score V4 = {F1Score4}')
        
        print(f'Precision V5 = {Precision5}')
        print(f'Recall V5 = {Recall5}')
        print(f'F1Score V5 = {F1Score5}')
        
    except Exception as e:
        print(f"Could not load ground truth: {e}")
    
    # Convert dicts to MATLAB-compatible cell arrays (1, n) object arrays
    def dict_to_cell_array(d, n):
        """Convert dict to (1, n) object array for MATLAB cell array compatibility."""
        arr = np.empty((1, n), dtype=object)
        for i in range(n):
            arr[0, i] = d.get(i, np.array([]))
        return arr
    
    # Prepare results
    results = {
        'labelV': labelV,
        'labelV2': labelV2,
        'labelV3': labelV3,
        'labelV4': labelV4,
        'labelV5': labelV5,
        'filenameString': filenameString,
        'icaData': dict_to_cell_array(icaData, n),
        'sigEnergy': dict_to_cell_array(sigEnergy, n),
        'maxScore': dict_to_cell_array(maxScore, n),
        'ClusterSizePat': dict_to_cell_array(ClusterSizePat, n),
        'OverallPeriFile': dict_to_cell_array(OverallPeriFile, n),
        'OverallArtFile': dict_to_cell_array(OverallArtFile, n),
        'OverallWhiteFile': dict_to_cell_array(OverallWhiteFile, n),
        'percWhiteArrayFile': dict_to_cell_array(percWhiteArrayFile, n),
        'percBrainArrayFile': dict_to_cell_array(percBrainArrayFile, n),
    }
    
    # Save workspace
    workspace_name = ENV_WORKSPACE_NAME or f'NEW_Workspace-{folder_name}V4.mat'
    workspace_path = output_dir / workspace_name
    sio.savemat(str(workspace_path), results)
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Extracting voxel information")
    print("=" * 50)
    
    results = process_patient()
    
    print("\nProcessing complete!")
    print(f"Labels (V5): {results['labelV5']}")
