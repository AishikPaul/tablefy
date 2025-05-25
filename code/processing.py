import cv2
import numpy as np
import easyocr
import pandas as pd
import matplotlib.pyplot as plt

def process_table_image(input_image_path):
    # --------- Your entire image processing pipeline here ---------
    # Replace hardcoded paths with dynamic usage based on input_image_path
    # Save intermediate images in "files/" as before
    # Save final CSV to "result_files/extracted_table.csv"
    
    # For brevity, I'm wrapping your existing code below, 
    # with adjustments for input and output paths

    # Step 1: binary conversion
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    _, binary_image = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_output_path = "files/binary_output.png"
    cv2.imwrite(binary_output_path, binary_image)

    # Step 2: shearing and rotation correction
    image_bin = cv2.imread(binary_output_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image_bin, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best_rect = approx

    if best_rect is None:
        raise ValueError("No suitable rectangular contour found!")

    def order_points(pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect_pts = order_points(best_rect)
    (dx, dy) = rect_pts[1] - rect_pts[0]
    rotation_angle = np.degrees(np.arctan2(dy, dx))

    width = np.linalg.norm(rect_pts[1] - rect_pts[0])
    height = np.linalg.norm(rect_pts[2] - rect_pts[1])
    if height > width:
        if rotation_angle < 0:
            rotation_angle += 90
        else:
            rotation_angle -= 90
    if abs(rotation_angle) < 1:
        rotation_angle = 0

    shear_x_angle = np.degrees(np.arctan2(rect_pts[1][1] - rect_pts[0][1], rect_pts[1][0] - rect_pts[0][0]))
    shear_y_angle = np.degrees(np.arctan2(rect_pts[3][0] - rect_pts[0][0], rect_pts[3][1] - rect_pts[0][1]))

    if abs(shear_x_angle) < 5:
        shear_x_angle = 0
    if abs(shear_y_angle) < 5:
        shear_y_angle = 0

    (h, w) = image_bin.shape
    shear_x = -np.tan(np.radians(shear_x_angle))
    shear_y = -np.tan(np.radians(shear_y_angle))

    shear_matrix = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])

    sheared_image = cv2.warpAffine(image_bin, shear_matrix, (w, h))
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
    final_image = cv2.warpAffine(sheared_image, rotation_matrix, (w, h))

    final_corrected_path = "files/final_corrected.png"
    cv2.imwrite(final_corrected_path, final_image)

    # Step 3: cropping table
    image_crop = cv2.imread(final_corrected_path, cv2.IMREAD_GRAYSCALE)
    binary = cv2.adaptiveThreshold(image_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    edges_crop = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges_crop, 1, np.pi/180, threshold=100, minLineLength=180, maxLineGap=80)
    mask = np.zeros_like(image_crop)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    table_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(table_contour)
    table_image = image_crop[y:y+h, x:x+w]
    table_image_path = "files/test_table_only.png"
    cv2.imwrite(table_image_path, table_image)

    # Step 4: detecting cells
    image_table = cv2.imread(table_image_path, cv2.IMREAD_GRAYSCALE)
    edges_table = cv2.Canny(image_table, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges_table, 1, np.pi/180, threshold=100, minLineLength=180, maxLineGap=80)

    bounding_boxes = []
    if lines is not None:
        x_coords, y_coords = [], []
        vertical_lines, horizontal_lines = [], []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])

            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float("inf")

            if abs(slope) > 5:
                vertical_lines.append((x1 + x2) // 2)
            elif abs(slope) < 0.2:
                horizontal_lines.append((y1 + y2) // 2)

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        vertical_lines.extend([x_min, x_max])
        horizontal_lines.extend([y_min, y_max])

        def merge_lines(lines, threshold=15):
            merged = []
            lines.sort()
            for coord in lines:
                if not merged or abs(merged[-1] - coord) > threshold:
                    merged.append(coord)
            return merged

        vertical_lines = merge_lines(vertical_lines, threshold=30)
        horizontal_lines = merge_lines(horizontal_lines, threshold=30)

        for i in range(len(vertical_lines) - 1):
            for j in range(len(horizontal_lines) - 1):
                x1, x2 = vertical_lines[i], vertical_lines[i + 1]
                y1, y2 = horizontal_lines[j], horizontal_lines[j + 1]
                bounding_boxes.append((x1, y1, x2, y2))

    # Step 5: OCR and CSV creation
    reader = easyocr.Reader(['en'])
    bounding_boxes.sort(key=lambda b: b[1])

    table_data = []
    current_row = []
    prev_y1 = None

    for (x1, y1, x2, y2) in bounding_boxes:
        if prev_y1 is not None and y1 != prev_y1:
            table_data.append(current_row)
            current_row = []

        cell_img = image_table[y1:y2, x1:x2]
        detected_text = reader.readtext(cell_img, detail=0)
        current_row.append(" ".join(detected_text) if detected_text else "N/A")
        prev_y1 = y1

    if current_row:
        table_data.append(current_row)

    csv_output_path = "result_files/extracted_table.csv"
    df = pd.DataFrame(table_data)
    df.to_csv(csv_output_path, index=False, header=False)

    return csv_output_path
