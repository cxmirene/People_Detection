import cv2
import numpy as np

left_camera_matrix = np.array([[344.7276859356455, 0, 164.4723132019533],
                               [0, 345.233999204332, 117.0978313248346],
                               [0,0,1]])
left_distortion = np.array([[0.3923933857791951, -2.4965717801875, -0.001251113426094487, -0.002417217283140779, 5.919737923990053]])



right_camera_matrix = np.array([[344.851405322161, 0, 164.9609008751307],
                                [0, 344.1841900157629, 128.163384469266],
                                [0,0,1]])
right_distortion = np.array([[0.3680668885911778, -2.423536458387395, -0.005405995970075847, -0.001687283685940221, 6.676235127032458]])

R = np.array([[0.9995741080625237, 0.00299727863363381, 0.02902789713022038],
              [-0.003585584807437043, 0.9997887991276778, 0.02023612414531704],
              [-0.02896111311049648, -0.02033158773013974, 0.9993737441356826]])
T = np.array([-4.002256239463035, 0.0272588795406389, 0.04774078902329625]) # 平移关系向量

size = (320, 240) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_32FC1)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_32FC1)

