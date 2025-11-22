
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib

if os.environ.get('MPLBACKEND') is None:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def detect_brain_grid(main_image, template_image, threshold=0.7):
    """Detect all brain slices in the grid layout using template matching"""
    if len(main_image.shape) == 3:
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    else:
        main_gray = main_image.copy()

    if len(template_image.shape) == 3:
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_image.copy()

    corr_result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(corr_result >= threshold)
    rows, cols = loc[0], loc[1]

    if len(rows) == 0:
        raise ValueError("No template matches found. Try lowering threshold.")

    # Cluster nearby matches
    unique_positions = []
    positions = list(zip(rows, cols))
    used = set()
    
    for i, (r, c) in enumerate(positions):
        if i in used:
            continue
        cluster = [(r, c)]
        used.add(i)
        for j, (r2, c2) in enumerate(positions[i+1:], start=i+1):
            if abs(r - r2) < template_gray.shape[0]//2 and abs(c - c2) < template_gray.shape[1]//2:
                cluster.append((r2, c2))
                used.add(j)
        mean_r = int(np.mean([x[0] for x in cluster]))
        mean_c = int(np.mean([x[1] for x in cluster]))
        unique_positions.append((mean_r, mean_c))
    
    unique_positions = sorted(unique_positions, key=lambda x: (x[0], x[1]))
    
    rows_list = [p[0] for p in unique_positions]
    cols_list = [p[1] for p in unique_positions]
    unique_rows = sorted(set(rows_list))
    unique_cols = sorted(set(cols_list))
    
    if len(unique_cols) >= 2:
        sizeX = int(np.median(np.diff(sorted(unique_cols))))
    else:
        sizeX = template_gray.shape[1] * 2
    
    if len(unique_rows) >= 2:
        sizeY = int(np.median(np.diff(sorted(unique_rows))))
    else:
        sizeY = template_gray.shape[0] * 2
    
    print(f"[DEBUG] Found grid: {len(unique_rows)} rows x {len(unique_cols)} columns")
    print(f"[DEBUG] Grid cell size: {sizeX}x{sizeY}")
    
    return unique_positions, unique_rows, unique_cols, sizeX, sizeY, template_gray.shape


def detect_brain_slices_by_contours(main_image, min_area=500, max_area=15000):
    """Detect brain slices directly using contour detection"""
    if len(main_image.shape) == 3:
        gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = main_image.copy()
    
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    brain_slices = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.5 < aspect_ratio < 2.0:
                brain_slices.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })
    
    brain_slices.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
    print(f"[DEBUG] Detected {len(brain_slices)} brain slices using contours")
    
    return brain_slices


def extract_all_slices(main_image, unique_rows, unique_cols, sizeX, sizeY, 
                       template_positions, template_size, expansion_factor=1.3):
    """Extract complete individual brain slices with adjustable expansion"""
    h, w = main_image.shape[:2]
    slices = []
    
    BASE_WIDTH = sizeX
    BASE_HEIGHT = sizeY
    SLICE_WIDTH = int(BASE_WIDTH * expansion_factor)
    SLICE_HEIGHT = int(BASE_HEIGHT * expansion_factor)
    
    print(f"[DEBUG] Grid spacing: {BASE_WIDTH}x{BASE_HEIGHT} pixels")
    print(f"[DEBUG] Extraction size: {SLICE_WIDTH}x{SLICE_HEIGHT} pixels (expansion={expansion_factor})")
    
    for row_idx, row_y in enumerate(unique_rows[:-1]):
        for col_idx, col_x in enumerate(unique_cols):
            corrected_x = col_x - template_size[1] // 3
            corrected_y = row_y - template_size[0] // 3
            
            center_offset_x = (SLICE_WIDTH - BASE_WIDTH) // 2
            center_offset_y = (SLICE_HEIGHT - BASE_HEIGHT) // 2
            
            startX = max(0, int(corrected_x - center_offset_x))
            startY = max(0, int(corrected_y - center_offset_y))
            endX = min(w, startX + SLICE_WIDTH)
            endY = min(h, startY + SLICE_HEIGHT)
            
            slice_img = main_image[startY:endY, startX:endX].copy()
            
            if slice_img.shape[0] < 20 or slice_img.shape[1] < 20:
                continue
            
            gray_slice = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY) if len(slice_img.shape) == 3 else slice_img
            non_black = np.sum(gray_slice > 10)
            
            if non_black < 100:
                continue
            
            # Remove templates
            for (temp_r, temp_c) in template_positions:
                rel_x = temp_c - startX
                rel_y = temp_r - startY
                
                if (0 <= rel_x < slice_img.shape[1] and 0 <= rel_y < slice_img.shape[0]):
                    padding = 3
                    mask_startX = max(0, rel_x - padding)
                    mask_startY = max(0, rel_y - padding)
                    mask_endX = min(slice_img.shape[1], rel_x + template_size[1] + padding)
                    mask_endY = min(slice_img.shape[0], rel_y + template_size[0] + padding)
                    slice_img[mask_startY:mask_endY, mask_startX:mask_endX] = 0
            
            slices.append({
                'image': slice_img,
                'position': (row_idx, col_idx),
                'coordinates': (startX, startY, endX, endY),
                'row_y': row_y,
                'col_x': col_x
            })
    
    print(f"[DEBUG] Extracted {len(slices)} complete brain slices")
    return slices


def extract_slices_from_contours(main_image, brain_slices, padding=5):
    """Extract brain slices using detected contours"""
    h, w = main_image.shape[:2]
    slices = []
    
    for idx, slice_info in enumerate(brain_slices):
        x, y, w_box, h_box = slice_info['bbox']
        
        startX = max(0, x - padding)
        startY = max(0, y - padding)
        endX = min(w, x + w_box + padding)
        endY = min(h, y + h_box + padding)
        
        slice_img = main_image[startY:endY, startX:endX].copy()
        
        row_idx = idx // 11
        col_idx = idx % 11
        
        slices.append({
            'image': slice_img,
            'position': (row_idx, col_idx),
            'coordinates': (startX, startY, endX, endY),
            'bbox': slice_info['bbox'],
            'area': slice_info['area']
        })
    
    print(f"[DEBUG] Extracted {len(slices)} slices from contours")
    return slices


def quantify_activation(slice_img, color_threshold=30):
    """Calculate activation score"""
    if len(slice_img.shape) != 3:
        return 0, 0, 0, 0
    
    hsv = cv2.cvtColor(slice_img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(slice_img)
    gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
    
    mean_val = (b.astype(float) + g.astype(float) + r.astype(float)) / 3
    color_variance = np.sqrt(
        ((b.astype(float) - mean_val) ** 2 + 
         (g.astype(float) - mean_val) ** 2 + 
         (r.astype(float) - mean_val) ** 2) / 3
    )
    
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    
    red_mask = ((hue < 15) | (hue > 165)) & (saturation > 40) & (value > 40)
    orange_mask = (hue >= 5) & (hue <= 25) & (saturation > 40) & (value > 40)
    blue_mask = (hue > 100) & (hue < 130) & (saturation > 50) & (value > 50)
    color_mask = ((saturation > 30) | (color_variance > 20)) & (gray > color_threshold)
    
    final_mask = red_mask | orange_mask | blue_mask | color_mask
    activation_count = np.sum(final_mask)
    red_orange_count = np.sum(red_mask | orange_mask)
    blue_count = np.sum(blue_mask)
    activation_score = red_orange_count * 2 + blue_count
    
    return activation_score, activation_count, red_orange_count, blue_count

def find_max_activation_slice(slices):
    """Find slice with maximum activation"""
    max_score = 0
    best_slice = None
    
    for slice_data in slices:
        score, total, red_orange, blue = quantify_activation(slice_data['image'])
        slice_data['activation_score'] = score
        slice_data['red_orange'] = red_orange
        
        if score > max_score:
            max_score = score
            best_slice = slice_data
    
    if best_slice is None:
        raise ValueError("No activation found")
    
    print(f"\n[INFO] Selected Row {best_slice['position'][0]}, Col {best_slice['position'][1]} "
          f"(Score={max_score}, Red/Orange={best_slice['red_orange']})")
    
    return best_slice


def dbscan_cluster(image, epsilon=3, min_samples=10, color_threshold=30, prioritize_red=True):
    """Apply DBSCAN and return cluster with bounding box"""
    if len(image.shape) != 3:
        raise ValueError("Image must be in color")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    
    mean_val = (b.astype(float) + g.astype(float) + r.astype(float)) / 3
    color_variance = np.sqrt(
        ((b.astype(float) - mean_val) ** 2 + 
         (g.astype(float) - mean_val) ** 2 + 
         (r.astype(float) - mean_val) ** 2) / 3
    )
    
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    
    color_mask = ((saturation > 30) | (color_variance > 20)) & (gray > color_threshold)
    red_mask = ((hue < 15) | (hue > 165)) & (saturation > 40) & (value > 40)
    orange_mask = (hue >= 5) & (hue <= 25) & (saturation > 40) & (value > 40)
    blue_mask = (hue > 100) & (hue < 130) & (saturation > 50) & (value > 50)
    
    final_mask = (red_mask | orange_mask | blue_mask | color_mask).astype(np.uint8) * 255
    
    y, x = np.where(final_mask > 0)
    points = np.column_stack((x, y))
    
    print(f"[DEBUG] Found {len(points)} activation pixels")

    if len(points) == 0:
        raise ValueError("No activation pixels found")

    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = db.labels_

    cluster_info = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        cluster_points = points[labels == lbl]
        
        cluster_colors = [image[py, px] for (px, py) in cluster_points[:100]]
        avg_color = np.mean(cluster_colors, axis=0)
        
        cluster_type = "RED/ORANGE" if avg_color[2] > avg_color[0] else "BLUE"
        
        cluster_info.append({
            'points': cluster_points,
            'size': len(cluster_points),
            'type': cluster_type
        })

    print(f"[DEBUG] Found {len(cluster_info)} clusters")

    if len(cluster_info) == 0:
        raise ValueError("No clusters found")

    # Select best cluster
    if prioritize_red:
        red_clusters = [c for c in cluster_info if c['type'] == 'RED/ORANGE']
        biggest_cluster = max(red_clusters, key=lambda x: x['size'])['points'] if red_clusters else max(cluster_info, key=lambda x: x['size'])['points']
    else:
        biggest_cluster = max(cluster_info, key=lambda x: x['size'])['points']

    print(f"[DEBUG] Selected cluster: {len(biggest_cluster)} points")

    # Calculate bounding box
    min_x = np.min(biggest_cluster[:, 0])
    max_x = np.max(biggest_cluster[:, 0])
    min_y = np.min(biggest_cluster[:, 1])
    max_y = np.max(biggest_cluster[:, 1])
    bbox = (min_x, min_y, max_x, max_y)

    # Visualization
    cluster_image = image.copy()
    for (x, y) in biggest_cluster:
        cv2.circle(cluster_image, (x, y), 1, (0, 255, 255), -1)

    return cluster_image, biggest_cluster, bbox

def soz_localization_pipeline(main_image_path, template_path=None, epsilon=3, min_samples=5, 
                               threshold=0.7, color_threshold=30, prioritize_red=True,
                               expansion_factor=1.3, use_contour_detection=False,
                               output_path=None, show_plot=False):
    """
    Complete pipeline with two detection methods
    
    Args:
        use_contour_detection: If True, detect slices directly (no template needed)
                              If False, use R template matching
    """
    main_image = cv2.imread(main_image_path)
    
    if use_contour_detection:
        print("[INFO] Using contour-based slice detection")
        brain_slices_info = detect_brain_slices_by_contours(main_image, min_area=500, max_area=15000)
        all_slices = extract_slices_from_contours(main_image, brain_slices_info, padding=5)
    else:
        print("[INFO] Using template-based slice detection")
        template_image = cv2.imread(template_path)
        template_positions, unique_rows, unique_cols, sizeX, sizeY, template_size = \
            detect_brain_grid(main_image, template_image, threshold)
        
        all_slices = extract_all_slices(main_image, unique_rows, unique_cols, 
                                         sizeX, sizeY, template_positions, template_size,
                                         expansion_factor)
    
    best_slice = find_max_activation_slice(all_slices)
    brain_slice = best_slice['image']
    
    clustered_image, biggest_cluster, bbox = dbscan_cluster(brain_slice, epsilon, min_samples, 
                                                             color_threshold, prioritize_red)
    
    print(f"[INFO] Cluster bounding box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})")

    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Panel 1: Full image with ALL slice bounding boxes
    plt.subplot(1, 3, 1)
    plt.title("SOZ IC with Detected Slices", fontsize=12)
    plt.imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))

    # Highlight ONLY the selected slice in green
    coords = best_slice['coordinates']
    rect = plt.Rectangle(
        (coords[0], coords[1]),
        coords[2] - coords[0],
        coords[3] - coords[1],
        fill=False,
        edgecolor='lime',
        linewidth=2
    )
    plt.gca().add_patch(rect)

    plt.axis('off')

    # Panel 2: Extracted single slice
    plt.subplot(1, 3, 2)
    plt.title(f"Single Slice (Row {best_slice['position'][0]}, Col {best_slice['position'][1]})", fontsize=11)
    plt.imshow(cv2.cvtColor(brain_slice, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Panel 3: Cluster with bounding box
    plt.subplot(1, 3, 3)
    plt.title("SOZ Cluster (DBSCAN)", fontsize=12)
    plt.imshow(cv2.cvtColor(clustered_image, cv2.COLOR_BGR2RGB))
    bbox_rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                              fill=False, edgecolor='lime', linewidth=2)
    plt.gca().add_patch(bbox_rect)
    plt.axis('off')
    
    plt.tight_layout()
    output_file = Path(output_path) if output_path else Path("SOZ_localization_result.png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Result saved to {output_file}")

    if show_plot:
        plt.show()

    plt.close(fig)

    return clustered_image, biggest_cluster, best_slice


def generate_dbscan_visual(main_image_path, output_path, template_path=None, **kwargs):
    """Helper to run localization and persist visualization to disk."""
    kwargs.setdefault('use_contour_detection', True)
    soz_localization_pipeline(
        main_image_path=main_image_path,
        template_path=template_path,
        output_path=output_path,
        show_plot=False,
        **kwargs
    )
    return str(Path(output_path))

if __name__ == "__main__":
    # grid_image_path = '/home/local/ASUAD/abanerj3/Desktop/expertknowledge/DATA/ASUAI_001/MO/report/IC_9_thresh.png'
    grid_image_path = '/home/local/ASUAD/abanerj3/Desktop/expertknowledge/final/patient_results/ASUAI_001_result/ic_soz/IC_34_thresh.png'
    template_path = '/home/local/ASUAD/abanerj3/Desktop/expertknowledge/DATA/R.png'

    clustered_img, cluster_pts, best_slice_info = soz_localization_pipeline(
        main_image_path=grid_image_path,
        epsilon=3,
        min_samples=5,
        color_threshold=30,
        prioritize_red=True,
        use_contour_detection=True,
        show_plot=True
    )



