import numpy as np
import cv2
import camera_configs
import math

class SGBM():
    stereo = 0

    def Init_SGBM(self):
        window_size = 6
        min_disp = 0
        num_disp = 320 - min_disp
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=32,  # max_disp has to be dividable by 16 f. E. HH 192, 256
            blockSize=6,
            P1=8 * window_size ** 2,
            # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
            P2=32 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=32,
            preFilterCap=63,
            mode=0
        )

    def Coordination(self, frame1, frame2, rect):
        # 根据更正map对图片进行重构
        img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

        # 将图片置为灰度图，为StereoBM作准备
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        # 根据Semi-Global Block Matching方法生成差异图
        disparity = self.stereo.compute(imgL, imgR)
        disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., camera_configs.Q)

        count = 0
        z = 0
        result_distance = []
        for r in rect:
            r_x = int((r[0]+r[2])/2)
            r_y = int((r[1]+r[3])/2)
            # cv2.rectangle(disp, (r_x-10, r_y-10), (r_x+10, r_y+10), (255, 255, 0), 4)
            for i in range(r_x-10,r_x+10):
                for j in range(r_y-10,r_y+10):
                    z1 = threeD[j][i][2]
                    if math.isnan(z1) or math.isnan(z1) or z1>=160000 or z1<=0:
                        continue
                    count += 1
                    z += z1
            if count==0:
                result_distance.append(-1)
            else:
                result_distance.append(round(z/count,2))

        # for i in range(rect[0,])
        # cv2.imshow("depth", disp)
        # cv2.imwrite("./SGBM_depth.jpg", disp)
        return result_distance, disp


# sgbm = SGBM()
# sgbm.Init_SGBM()
# frame1 = cv2.imread("/home/cxm-irene/文档/Two-eye/Image-Collect/Picture/left_1.jpg",1)
# frame2 = cv2.imread("/home/cxm-irene/文档/Two-eye/Image-Collect/Picture/right_1.jpg",1)
# rect = [[56,26,229,236]]
# distance = sgbm.Coordination(frame1, frame2, rect)
# print(distance)