import cv2

image = cv2.imread("normal_faces/0000b86e2fd18333.jpg")


# def handle_mouse_click(e, x, y, flags, param):

#     if e == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
#         cv2.imshow("Image", image)
#         print("X: " + str(x))
#         print("Y: " + str(y))

# points1 = [[229.59984, 260.40552],
#            [292.75168, 243.79257],
#            [345.43906, 258.6347],
#            [342.63953, 294.09265],
#            [295.93002, 320.3842],
#            [241.45047, 304.22742]]

# points2 = [[548.568, 149.71957],
#            [613.12213, 131.57373],
#            [670.7916, 151.63983],
#            [662.09875, 193.59552],
#            [611.0434, 217.80844],
#            [560.1271, 197.22649]]

# points3 = [[705.8756, 298.95682],
#            [747.1592, 288.6646],
#            [791.6697, 306.06146],
#            [779.6672, 334.3117],
#            [743.137, 342.89935],
#            [713.72723, 327.10797]]

# points4 = [[476.02835, 433.00665],
#            [532.4949, 434.4976],
#            [578.76514, 456.88272],
#            [569.63574, 484.41766],
#            [517.46655, 492.923],
#            [473.2537, 465.87402]]

# all_points = [points1, points2, points3, points4]

# c = []
# for points in all_points:
#     for i in range(len(points)):
#         point = points[i]
#         cv2.circle(image, (int(point[0]), int(point[1])), 3, (0, 0, i*40), -1)
cv2.circle(image, (733, 413), 3, (0, 0, 255), -1)
cv2.circle(image, (759, 411), 3, (0, 0, 255), -1)
cv2.circle(image, (850, 428), 3, (0, 0, 255), -1)
cv2.circle(image, (841, 467), 3, (0, 0, 255), -1)
cv2.circle(image, (758, 488), 3, (0, 0, 255), -1)
cv2.circle(image, (741, 464), 3, (0, 0, 255), -1)

# 384, 348, 705, 669

# 597,182,726,311


# cv2.circle(image, (357, 348), 3, (0, 0, 255), -1)
# cv2.circle(image, (400, 391), 3, (0, 0, 255), -1)

# cv2.circle(image, (549, 295), 3, (0, 0, 255), -1)
# cv2.circle(image, (592, 338), 3, (0, 0, 255), -1)

# cv2.circle(image, (636, 285), 3, (0, 0, 255), -1)
# cv2.circle(image, (688, 337), 3, (0, 0, 255), -1)

# cv2.circle(image, (717, 349), 3, (0, 0, 255), -1)
# cv2.circle(image, (769, 401), 3, (0, 0, 255), -1)

# 338	434
# 382	477

# 348	357
# 391	400

# 295	549
# 338	592

# 285	636
# 337	688

# 349	717
# 401	769


# cv2.circle(image, (554, 82), 3, (0, 0, 255), -1)
# cv2.circle(image, (683, 211), 3, (0, 0, 255), -1)

# cv2.circle(image, (703, 255), 3, (0, 0, 255), -1)
# cv2.circle(image, (792, 345), 3, (0, 0, 255), -1)

# cv2.circle(image, (454, 383), 3, (0, 0, 255), -1)
# cv2.circle(image, (583, 512), 3, (0, 0, 255), -1)

cv2.imshow("Image", image)

# cv2.setMouseCallback("Image", handle_mouse_click)

cv2.waitKey(0)

cv2.destroyAllWindows()
