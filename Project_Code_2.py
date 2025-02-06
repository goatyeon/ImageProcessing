#!/usr/bin/env python
# coding: utf-8

# ### Base

# In[3]:


import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFile
from scipy.ndimage import label, find_objects
from sklearn.cluster import DBSCAN
import cv2

# Truncation error ignore setting
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(image_path):
    return Image.open(image_path).convert('RGB')

def to_grayscale(image):
    return image.convert('L')

def gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def sobel_edge_detection(image):
    image = np.array(image, dtype=np.float32)
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    sobel_x = np.zeros_like(image)
    sobel_y = np.zeros_like(image)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sobel_x[i, j] = np.sum(gx * image[i-1:i+2, j-1:j+2])
            sobel_y[i, j] = np.sum(gy * image[i-1:i+2, j-1:j+2])
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = (magnitude / magnitude.max()) * 255
    return Image.fromarray(magnitude.astype(np.uint8))

def hough_transform(image):
    image = np.array(image)
    height, width = image.shape
    rho_max = int(np.hypot(height, width))
    accumulator = np.zeros((2 * rho_max, 180), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            if image[y, x] > 128:
                for theta in range(180):
                    theta_rad = np.deg2rad(theta)
                    rho = int(x * np.cos(theta_rad) + y * np.sin(theta_rad))
                    accumulator[rho + rho_max, theta] += 1
    
    return accumulator

def find_peaks(accumulator, threshold=100):
    peaks = []
    rho_max, theta_max = accumulator.shape
    for rho in range(rho_max):
        for theta in range(theta_max):
            if accumulator[rho, theta] > threshold:
                peaks.append((rho - rho_max // 2, theta))
    return peaks

def draw_lines(image, peaks):
    draw = ImageDraw.Draw(image)
    for rho, theta in peaks:
        theta_rad = np.deg2rad(theta)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)
    return image

def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.linalg.det(A) != 0:
        x, y = np.linalg.solve(A, b)
        return int(np.round(x)), int(np.round(y))
    else:
        return None

def cluster_intersections(intersections):
    clustering = DBSCAN(eps=20, min_samples=2).fit(intersections)
    labels = clustering.labels_
    unique_labels = set(labels)
    clustered_points = []
    for k in unique_labels:
        if k == -1:
            continue
        class_members = (labels == k)
        xy = intersections[class_members]
        centroid = xy.mean(axis=0).astype(int)
        clustered_points.append(centroid)
    return clustered_points

def calculate_slope(line):
    rho, theta = line
    return np.tan(np.deg2rad(theta))

def group_lines_by_similarity(lines, threshold=0.85):
    grouped_lines = []
    while len(lines) > 0:
        current_line = lines.pop(0)
        group = [current_line]
        i = 0
        while i < len(lines):
            similarity = calculate_similarity(current_line, lines[i])
            if similarity >= threshold:
                group.append(lines.pop(i))
            else:
                i += 1
        grouped_lines.append(group)
    return grouped_lines

def calculate_similarity(line1, line2):
    vector1 = np.array([np.cos(np.deg2rad(line1[1])), np.sin(np.deg2rad(line1[1]))])
    vector2 = np.array([np.cos(np.deg2rad(line2[1])), np.sin(np.deg2rad(line2[1]))])
    dot_product = np.dot(vector1, vector2)
    return dot_product

def find_intersections_between_groups(grouped_lines1, grouped_lines2):
    intersections = []
    for line1 in grouped_lines1:
        for line2 in grouped_lines2:
            pt = line_intersection(line1, line2)
            if pt is not None:
                intersections.append(pt)
    return intersections

def perspective_transform(pts, img):
    img_array = np.array(img)
    pts1 = np.array(pts, dtype=np.float32)
    pts1 = order_points(pts1)

    w1 = np.linalg.norm(pts1[1] - pts1[0])
    w2 = np.linalg.norm(pts1[2] - pts1[3])
    h1 = np.linalg.norm(pts1[2] - pts1[1])
    h2 = np.linalg.norm(pts1[3] - pts1[0])
    width = int(max(w1, w2))
    height = int(max(h1, h2))

    pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    H = cv2.getPerspectiveTransform(pts1, pts2)
    warped_image = cv2.warpPerspective(img_array, H, (width, height))

    return warped_image

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def draw_intersections(image, intersections):
    if len(intersections) == 0:
        print("No intersections to draw.")
        return image

    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    for point in intersections:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(255, 0, 0))
    return drawn_image

def adjust_image_size(image, warped_image):
    original_width, original_height = image.size
    warped_width, warped_height = warped_image.shape[1], warped_image.shape[0]
    if warped_width < original_width or warped_height < original_height:
        resized_warped_image = cv2.resize(warped_image, (original_width, original_height))
        return resized_warped_image
    else:
        return warped_image

from scipy.spatial import ConvexHull

def find_optimal_rectangle_points(intersections):
    if len(intersections) < 4:
        print("Not enough intersections to find optimal rectangle points.")
        return intersections

    hull = ConvexHull(intersections)
    points = [intersections[vertex] for vertex in hull.vertices]
    return points[:4]

def draw_optimal_rectangle(image, optimal_points):
    optimal_rectangle_image = image.copy()
    draw = ImageDraw.Draw(optimal_rectangle_image)
    for i in range(len(optimal_points)):
        draw.line([tuple(optimal_points[i]), tuple(optimal_points[(i + 1) % len(optimal_points)])], fill=(0, 255, 0), width=2)
    return optimal_rectangle_image

def main(image_path):
    image = read_image(image_path)
    grayscale_image = to_grayscale(image)
    blurred_image = gaussian_blur(grayscale_image)
    edge_image = sobel_edge_detection(blurred_image)

    lines = hough_transform(edge_image)
    peaks = find_peaks(lines)

    if not peaks:
        print("No peaks found in Hough Transform.")
        return image

    line_image = draw_lines(image.copy(), peaks)
    grouped_lines = group_lines_by_similarity(peaks)

    intersections = []
    for i in range(len(grouped_lines)):
        for j in range(i + 1, len(grouped_lines)):
            intersections += find_intersections_between_groups(grouped_lines[i], grouped_lines[j])

    if not intersections:
        print("No intersections found.")
        return image

    intersections = np.array(intersections)
    clustered_intersections = cluster_intersections(intersections)

    intersections_image = draw_intersections(image.copy(), clustered_intersections)
    optimal_points = find_optimal_rectangle_points(clustered_intersections)

    if len(optimal_points) < 4:
        print("Not enough optimal points found.")
        return intersections_image

    optimal_rectangle_image = draw_optimal_rectangle(image.copy(), optimal_points)
    warped_image = perspective_transform(optimal_points, image)
    warped_image_image = Image.fromarray(warped_image)

    adjusted_warped_image = adjust_image_size(image, warped_image)
    adjusted_warped_image_image = Image.fromarray(adjusted_warped_image)
    
    return line_image

image_path = 'businesscard/BC2.jpg'
result_image = main(image_path)
result_image.show()


# ### 배경 제거 추가

# In[113]:


import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFile
from scipy.ndimage import label, find_objects
from sklearn.cluster import DBSCAN
import cv2

# 트렁케이션 오류 무시 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def to_grayscale(image):
    return image.convert('L')

def gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def sobel_edge_detection(image):
    image = np.array(image, dtype=np.float32)
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    sobel_x = np.zeros_like(image)
    sobel_y = np.zeros_like(image)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sobel_x[i, j] = np.sum(gx * image[i-1:i+2, j-1:j+2])
            sobel_y[i, j] = np.sum(gy * image[i-1:i+2, j-1:j+2])
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = (magnitude / magnitude.max()) * 255
    return Image.fromarray(magnitude.astype(np.uint8))

def hough_transform(image):
    image = np.array(image)
    if len(image.shape) == 3:  # 컬러 이미지인 경우
        height, width, _ = image.shape
    else:  # 그레이스케일 이미지인 경우
        height, width = image.shape
    rho_max = int(np.hypot(height, width))
    accumulator = np.zeros((2 * rho_max, 180), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            if len(image.shape) == 3:
                if np.all(image[y, x] > 128):  # 흰색 픽셀만 고려
                    for theta in range(180):
                        theta_rad = np.deg2rad(theta)
                        rho = int(x * np.cos(theta_rad) + y * np.sin(theta_rad))
                        accumulator[rho + rho_max, theta] += 1
            else:
                if image[y, x] > 128:
                    for theta in range(180):
                        theta_rad = np.deg2rad(theta)
                        rho = int(x * np.cos(theta_rad) + y * np.sin(theta_rad))
                        accumulator[rho + rho_max, theta] += 1
    
    return accumulator


def find_peaks(accumulator, threshold=100):
    peaks = []
    rho_max, theta_max = accumulator.shape
    for rho in range(rho_max):
        for theta in range(theta_max):
            if accumulator[rho, theta] > threshold:
                peaks.append((rho - rho_max // 2, theta))
    return peaks

def draw_lines(image, peaks):
    draw = ImageDraw.Draw(image)
    for rho, theta in peaks:
        theta_rad = np.deg2rad(theta)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)
    return image

def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.linalg.det(A) != 0:
        x, y = np.linalg.solve(A, b)
        return int(np.round(x)), int(np.round(y))
    else:
        return None

def cluster_intersections(intersections):
    clustering = DBSCAN(eps=20, min_samples=2).fit(intersections)
    labels = clustering.labels_
    unique_labels = set(labels)
    clustered_points = []
    for k in unique_labels:
        if k == -1:
            continue
        class_members = (labels == k)
        xy = intersections[class_members]
        centroid = xy.mean(axis=0).astype(int)
        clustered_points.append(centroid)
    return clustered_points

def calculate_slope(line):
    rho, theta = line
    return np.tan(np.deg2rad(theta))

def group_lines_by_similarity(lines, threshold=0.85):
    grouped_lines = []
    while len(lines) > 0:
        current_line = lines.pop(0)  # lines 리스트에서 직선을 하나씩 추출
        group = [current_line]  # 현재 직선을 포함한 새로운 그룹 생성
        i = 0
        while i < len(lines):
            # 현재 직선과 비교 직선 간의 유사도 계산
            similarity = calculate_similarity(current_line, lines[i])
            # 유사도가 임계값 이상인 경우 그룹에 추가
            if similarity >= threshold:
                group.append(lines.pop(i))  # 그룹에 추가하고 lines에서 해당 직선 제거
            else:
                i += 1
        grouped_lines.append(group)  # 완성된 그룹을 그룹 리스트에 추가
    return grouped_lines

def calculate_similarity(line1, line2):
    # 직선을 벡터로 표현
    vector1 = np.array([np.cos(np.deg2rad(line1[1])), np.sin(np.deg2rad(line1[1]))])
    vector2 = np.array([np.cos(np.deg2rad(line2[1])), np.sin(np.deg2rad(line2[1]))])
    # 내적 계산
    dot_product = np.dot(vector1, vector2)
    # 두 벡터의 방향이 얼마나 유사한지를 나타내는 유사도 반환
    return dot_product

def find_intersections_between_groups(grouped_lines1, grouped_lines2):
    intersections = []
    for line1 in grouped_lines1:
        for line2 in grouped_lines2:
            pt = line_intersection(line1, line2)
            if pt is not None:
                intersections.append(pt)
    return intersections

def perspective_transform(pts, img):
    img_array = np.array(img)

    # 변환 전 좌표 정렬
    pts1 = np.array(pts, dtype=np.float32)
    pts1 = order_points(pts1)

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = np.linalg.norm(pts1[1] - pts1[0])
    w2 = np.linalg.norm(pts1[2] - pts1[3])
    h1 = np.linalg.norm(pts1[2] - pts1[1])
    h2 = np.linalg.norm(pts1[3] - pts1[0])
    width = int(max(w1, w2))  # 두 좌우 거리간의 최대값이 서류의 폭
    height = int(max(h1, h2))  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # 원근 변환 행렬 계산
    H = cv2.getPerspectiveTransform(pts1, pts2)

    # 원근 변환 적용
    warped_image = cv2.warpPerspective(img_array, H, (width, height))

    return warped_image

def order_points(pts):
    # 좌표를 초기화합니다.
    rect = np.zeros((4, 2), dtype="float32")

    # 좌상단과 우하단의 합이 가장 작은 좌표를 찾습니다.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 좌하단과 우상단의 차이가 가장 작은 좌표를 찾습니다.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # 정렬된 좌표를 반환합니다.
    return rect

def draw_intersections(image, intersections):
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    for point in intersections:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(255, 0, 0))
    return drawn_image

def adjust_image_size(image, warped_image):
    # 원본 이미지와 변환된 이미지의 크기를 비교
    original_width, original_height = image.size
    warped_width, warped_height = warped_image.shape[1], warped_image.shape[0]

    # 변환된 이미지가 더 작으면 이미지 크기를 조정
    if warped_width < original_width or warped_height < original_height:
        resized_warped_image = cv2.resize(warped_image, (original_width, original_height))
        return resized_warped_image
    else:
        return warped_image

    
'''    
from scipy.spatial import ConvexHull

def find_optimal_rectangle_points(intersections):
    hull = ConvexHull(intersections)
    points = [intersections[vertex] for vertex in hull.vertices]
    return points[:4]  # Convex Hull을 이루는 꼭지점 중에서 처음 4개만 선택하여 반환합니다.

def draw_optimal_rectangle(image, optimal_points):
    optimal_rectangle_image = image.copy()
    draw = ImageDraw.Draw(optimal_rectangle_image)
    for i in range(len(optimal_points)):
        draw.line([tuple(optimal_points[i]), tuple(optimal_points[(i + 1) % len(optimal_points)])], fill=(0, 255, 0), width=2)
    return optimal_rectangle_image
'''


from scipy.spatial import ConvexHull

def find_optimal_rectangle_points(intersections):
    # Convex Hull 알고리즘을 사용하여 꼭지점을 찾습니다.
    hull = ConvexHull(intersections)
    points = [intersections[vertex] for vertex in hull.vertices]
    
    # Convex Hull의 점들과 이웃한 점들을 추가합니다.
    additional_points = []
    for vertex in hull.vertices:
        for neighbor in hull.neighbors[vertex]:
            if neighbor != -1 and neighbor not in hull.vertices:
                additional_points.append(intersections[neighbor])
    
    # 선택된 점들을 합쳐 최대 8개의 점을 반환합니다.
    points.extend(additional_points)
    
    # 꼭지점의 개수가 8개보다 많으면 8개까지만 선택합니다.
    return points[:min(len(points), 8)]

def draw_optimal_rectangle(image, optimal_points):
    optimal_rectangle_image = image.copy()
    draw = ImageDraw.Draw(optimal_rectangle_image)
    for i in range(len(optimal_points)):
        draw.line([tuple(optimal_points[i]), tuple(optimal_points[(i + 1) % len(optimal_points)])], fill=(0, 255, 0), width=2)
    return optimal_rectangle_image


import numpy as np

def remove_background(image, threshold=128):
    # Convert the image to numpy array
    img_array = np.array(image)
    
    # Create an empty mask to mark the background pixels
    background_mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    # Iterate through each pixel in the image
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            # Check if the pixel intensity is below the threshold (background)
            if np.all(img_array[y, x] <= threshold):
                background_mask[y, x] = 255  # Mark the pixel as background
    
    # Invert the background mask to mark the foreground pixels
    foreground_mask = 255 - background_mask
    
    # Apply the foreground mask to the original image
    foreground_image = cv2.bitwise_and(img_array, img_array, mask=foreground_mask)
    
    # Convert the resulting image back to PIL format
    return Image.fromarray(foreground_image)

'''
def color_objects(image, color=(255, 255, 255)):
    # Convert the image to numpy array
    img_array = np.array(image)
    
    # Convert color space to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the blurred image to create a mask of foreground pixels
    _, mask = cv2.threshold(blurred_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Create a mask for coloring the objects
    color_mask = np.zeros_like(img_array)
    color_mask[:] = color  # 모든 픽셀을 지정된 색으로 설정합니다.

    # Apply the color mask to the original image for the objects
    colored_image = cv2.bitwise_and(color_mask, color_mask, mask=mask)

    # Convert the background mask to RGB format for consistency
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Apply the object mask to make the background black
    final_image = cv2.bitwise_or(colored_image, mask_rgb)

    # Convert the resulting image back to PIL format
    return Image.fromarray(final_image)
'''




def main(image_path):
    image = read_image(image_path)
    
    # 배경 제거
    segmented_image = remove_background(image, 200)
    #segmented_image.show()
    
    th_image = color_objects(segmented_image)
    #th_image.show()
    
    # 그레이스케일 및 가우시안 블러 적용
    grayscale_image = to_grayscale(th_image)
    blurred_image = gaussian_blur(grayscale_image, 2)
    
    # 엣지 검출
    edge_image = sobel_edge_detection(th_image)
    #edge_image.show()

    lines = hough_transform(edge_image)
    peaks = find_peaks(lines)

    line_image = draw_lines(segmented_image.copy(), peaks)
    #line_image.show()

    # 선들을 유사도에 따라 그룹화
    grouped_lines = group_lines_by_similarity(peaks)
    
    # 교차점 찾기
    intersections = []
    for i in range(len(grouped_lines)):
        for j in range(i + 1, len(grouped_lines)):
            intersections += find_intersections_between_groups(grouped_lines[i], grouped_lines[j])
    
    clustered_intersections = cluster_intersections(np.array(intersections))
    
    # 교차점 그리기
    intersections_image = draw_intersections(segmented_image.copy(), clustered_intersections)
    #intersections_image.show()

    # 최적의 사각형 점들 찾기
    optimal_points = find_optimal_rectangle_points(np.array(clustered_intersections))
    
    # 최적의 사각형 그리기
    optimal_rectangle_image = draw_optimal_rectangle(segmented_image.copy(), optimal_points)
    #optimal_rectangle_image.show()

    # 원근 변환
    warped_image = perspective_transform(np.array(optimal_points), segmented_image)
    warped_image_image = Image.fromarray(warped_image)
    #warped_image_image.show()

    # 크기 조정
    adjusted_warped_image = adjust_image_size(segmented_image, warped_image)
    adjusted_warped_image_image = Image.fromarray(adjusted_warped_image)
    #adjusted_warped_image_image.show()
    return segmented_image, intersections_image, optimal_rectangle_image, adjusted_warped_image_image

image_path = 'businesscard/BC6.jpg'
seg, inter, opti, result = main(image_path)

seg.show()
inter.show()
opti.show()
result.show()


# In[ ]:





# ### Homography 사용

# In[61]:


import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFile
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import cv2

# 트렁케이션 오류 무시 설정
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def to_grayscale(image):
    return image.convert('L')

def gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def sobel_edge_detection(image):
    image = np.array(image, dtype=np.float32)
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    sobel_x = np.zeros_like(image)
    sobel_y = np.zeros_like(image)
    
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            sobel_x[i, j] = np.sum(gx * image[i-1:i+2, j-1:j+2])
            sobel_y[i, j] = np.sum(gy * image[i-1:i+2, j-1:j+2])
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = (magnitude / magnitude.max()) * 255
    return Image.fromarray(magnitude.astype(np.uint8))

def hough_transform(image):
    image = np.array(image)
    height, width = image.shape
    rho_max = int(np.hypot(height, width))
    accumulator = np.zeros((2 * rho_max, 180), dtype=np.int32)
    
    for y in range(height):
        for x in range(width):
            if image[y, x] > 128:
                for theta in range(180):
                    theta_rad = np.deg2rad(theta)
                    rho = int(x * np.cos(theta_rad) + y * np.sin(theta_rad))
                    accumulator[rho + rho_max, theta] += 1
    
    return accumulator

def find_peaks(accumulator, threshold=100):
    peaks = []
    rho_max, theta_max = accumulator.shape
    for rho in range(rho_max):
        for theta in range(theta_max):
            if accumulator[rho, theta] > threshold:
                peaks.append((rho - rho_max // 2, theta))
    return peaks

def draw_lines(image, peaks):
    draw = ImageDraw.Draw(image)
    for rho, theta in peaks:
        theta_rad = np.deg2rad(theta)
        a = np.cos(theta_rad)
        b = np.sin(theta_rad)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)
    return image

def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.linalg.det(A) != 0:
        x, y = np.linalg.solve(A, b)
        return int(np.round(x)), int(np.round(y))
    else:
        return None

def cluster_intersections(intersections):
    clustering = DBSCAN(eps=20, min_samples=2).fit(intersections)
    labels = clustering.labels_
    unique_labels = set(labels)
    clustered_points = []
    for k in unique_labels:
        if k == -1:
            continue
        class_members = (labels == k)
        xy = intersections[class_members]
        centroid = xy.mean(axis=0).astype(int)
        clustered_points.append(centroid)
    return clustered_points

def calculate_slope(line):
    rho, theta = line
    return np.tan(np.deg2rad(theta))

def group_lines_by_similarity(lines, threshold=0.85):
    grouped_lines = []
    while len(lines) > 0:
        current_line = lines.pop(0)
        group = [current_line]
        i = 0
        while i < len(lines):
            similarity = calculate_similarity(current_line, lines[i])
            if similarity >= threshold:
                group.append(lines.pop(i))
            else:
                i += 1
        grouped_lines.append(group)
    return grouped_lines

def calculate_similarity(line1, line2):
    vector1 = np.array([np.cos(np.deg2rad(line1[1])), np.sin(np.deg2rad(line1[1]))])
    vector2 = np.array([np.cos(np.deg2rad(line2[1])), np.sin(np.deg2rad(line2[1]))])
    dot_product = np.dot(vector1, vector2)
    return dot_product

def find_intersections_between_groups(grouped_lines1, grouped_lines2):
    intersections = []
    for group1 in grouped_lines1:
        for group2 in grouped_lines2:
            for line1 in group1:
                for line2 in group2:
                    pt = line_intersection(line1, line2)
                    if pt is not None:
                        intersections.append(pt)
    return intersections

def draw_optimal_rectangle(image, optimal_points):
    optimal_rectangle_image = image.copy()
    draw = ImageDraw.Draw(optimal_rectangle_image)
    for i in range(len(optimal_points)):
        draw.line([tuple(optimal_points[i]), tuple(optimal_points[(i + 1) % len(optimal_points)])], fill=(0, 255, 0), width=2)
    return optimal_rectangle_image, optimal_points

def order_points(pts):
    # 입력 점의 좌표를 시계 방향으로 정렬
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)  # 리스트를 numpy 배열로 변환
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(pts, img):
    # 이미지 배열로 변환
    img_array = np.array(img)

    # 입력 점 좌표를 정렬
    pts1 = np.array(pts, dtype=np.float32)
    pts1 = order_points(pts1)
    
    # 사각형의 너비와 높이를 계산
    w1 = np.linalg.norm(pts1[1] - pts1[0])
    w2 = np.linalg.norm(pts1[2] - pts1[3])
    h1 = np.linalg.norm(pts1[2] - pts1[1])
    h2 = np.linalg.norm(pts1[3] - pts1[0])
    
    width = int(max(w1, w2))
    height = int(max(h1, h2))
    
    # 변환 후의 좌표 설정
    pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    
    ####
    # 변환 행렬 계산 및 원근 변환 적용
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    warped_image = cv2.warpPerspective(img_array, H, (width, height))
    
    return Image.fromarray(warped_image)

def draw_intersections(image, intersections):
    drawn_image = image.copy()
    draw = ImageDraw.Draw(drawn_image)
    for point in intersections:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(255, 0, 0))
    return drawn_image

def adjust_image_size(image, warped_image):
    original_width, original_height = image.size
    warped_width, warped_height = warped_image.size
    if (warped_width < original_width or warped_height < original_height):
        resized_warped_image = warped_image.resize((original_width, original_height))
        return resized_warped_image
    else:
        return warped_image

def find_optimal_rectangle_points(intersections):
    if len(intersections) < 4:
        raise ValueError("Not enough points to construct a rectangle")
    
    hull = ConvexHull(intersections)
    points = [intersections[vertex] for vertex in hull.vertices]
    
    if len(points) < 4:
        raise ValueError("Not enough points to construct a rectangle after convex hull")

    # 가장 바깥쪽 4개의 점을 선택하여 반환 (사각형의 네 꼭짓점)
    return points[:4]

def main(image_path):
    # 이미지 읽기
    image = read_image(image_path)
    
    # 그레이스케일로 변환
    gray_image = to_grayscale(image)
    
    # 가우시안 블러 적용
    blurred_image = gaussian_blur(gray_image)
    
    # Sobel 엣지 검출 적용
    edges_image = sobel_edge_detection(blurred_image)
    
    # 허프 변환 적용
    accumulator = hough_transform(np.array(edges_image))
    
    # 피크 찾기
    peaks = find_peaks(accumulator, threshold=100)
    
    # 교차점 그리기
    intersections_image = draw_lines(image.copy(), peaks)
    
    # 선들을 그룹화하고 교차점을 찾기
    grouped_lines = group_lines_by_similarity(peaks)
    intersections = find_intersections_between_groups(grouped_lines, grouped_lines)
    
    # 교차점들을 클러스터링하여 중앙점을 찾기
    clustered_intersections = cluster_intersections(np.array(intersections))
    
    # 최적의 사각형 점 찾기
    optimal_points = find_optimal_rectangle_points(np.array(clustered_intersections))
    
    # 최적의 사각형 그리기
    optimal_rectangle_image, optimal_points = draw_optimal_rectangle(image.copy(), optimal_points)
    
    # 원근 변환
    warped_image = perspective_transform(optimal_points, image)
    
    # 이미지 크기 조정
    adjusted_warped_image = adjust_image_size(image, warped_image)
    
    return intersections_image, optimal_rectangle_image, adjusted_warped_image

# 이미지 경로를 설정하고 메인 함수 실행
image_path = 'businesscard/BC3.jpg'
intersections_image, optimal_rectangle_image, adjusted_warped_image = main(image_path)

# 결과 이미지 보여주기
intersections_image.show()
optimal_rectangle_image.show()
adjusted_warped_image.show()


# In[ ]:





# In[ ]:





# In[30]:


import os

image_dir = 'businesscard/'
result_dir = 'result2_images/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

image_filenames = [filename for filename in os.listdir(image_dir) if filename.startswith('BC')]

for filename in image_filenames:
    image_path = os.path.join(image_dir, filename)
    
    clustered_points_image, optimal_rectangle_image, _ = main(image_path)
    
    base_filename = filename.split('.')[0]
    
    #image_with_lines.save(os.path.join(result_dir, f'{base_filename}_lines.jpg'))
    clustered_points_image.save(os.path.join(result_dir, f'{base_filename}_intersections.jpg'))
    optimal_rectangle_image.save(os.path.join(result_dir, f'{base_filename}_rectangle.jpg'))
    #Image.fromarray(image).save(os.path.join(result_dir, f'{base_filename}_warped.jpg'))

