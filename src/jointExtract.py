import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import MeanShift, estimate_bandwidth


# 함수 모아놓음
# 이미지 보여주는 함수####################
def img_show(title='image', img=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)

    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []

            for i in range(len(img)):
                titles.append(title)

        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)

            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])


# hough 변환 알고리즘##############################
def hough(red_points, houghThreshold, minLength, maxGap, onImg):
    lines = cv2.HoughLinesP(red_points, 1, np.pi / 180, threshold=houghThreshold, minLineLength=minLength,
                            maxLineGap=maxGap)
    return lines


# 허프 변환으로 추출한 선분들 (lines) 의 각도 구하기 / 범위는 0도~180도##############################
def calculate_angle(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    # 두 점의 좌표를 이용하여 각도를 계산합니다.
    angle = math.atan2(y2 - y1, x2 - x1)  # 라디안 단위로 계산됩니다.
    angle = math.degrees(angle)  # 각도로 변환합니다.

    # 결과값을 0 ~ 180도 범위로 조정합니다.
    if angle < 0:
        angle += 180

    return int(angle)


# 길이 구하기#####################################
def distance(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
    dis = round(math.sqrt(dis), 2)

    return dis


# 이미지 끝 점 구하기##############################
def halfPoints(points, half_points):

    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]

    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    
    half_points.append([x,y])
    return


# 카메라에서 실제 거리 구하기##############################
class Lens:
    # 0.25, 0.4, 0.65, 1.2mm
    def __init__(self, distance=1000, fl=12, siah=6.287, srph=4050):
        self.FL = fl  # FL: focalLength(초점길이)
        self.WD = distance * 0.001  # WD: workingDistance 0.34 물체와 카메라 사이의 거리
        self.SIAH = siah  # SIAH: sensorImageArea 14.0 센서 이미지 영역
        self.SRPH = srph  # SRPH: sensorResolßßßution 9248 센서 해상도

        # if distance <= 2000 : distance += 1000

        self.PMAG = self.FL / (distance)  # PMAG
        self.HFOV = self.SIAH / self.PMAG  # HFOV
        self.SPP = self.HFOV / self.SRPH  # SPP: Size Per Pixel

        # print("SPP(0.39):", self.SPP, ",  HFOV(1572):", self.HFOV, ", PMAG(0.004):", self.PMAG)
        self.R = 0.045 * ((self.WD) ** 2) - 0.355 * (self.WD) + 0.82  # 감소계수

        # ~~~~~~~~~~~~~~~~ REMOVABLE ~~~~~~~~~~~~~~~~~~~~~~
        self.R *= 1.2

    def real_length(self, pixel_length_list):
        real_length_list = []
        # print(pixel_length_list[0] * self.SPP * self.R )
        for i in range(len(pixel_length_list)):
            real_length_list.append(pixel_length_list[i] * self.SPP * self.R)
        # real_length_list = [pixels * self.SPP * self.R * 10000 for pixels in pixel_length_list]
        return real_length_list


# 선분을 극좌표로 변경
def getpolar(x1, y1, x2, y2):
    # 선분의 기울기 계산
    if (x2 == x1):
        theta = 0
        r = abs(x1)
    else:
        m = (y2 - y1) / (x2 - x1)
        # 라디안 단위 각도 계산
        theta = (np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi) + 90
        if theta > 180: theta = theta - 360
        # r 계산
        b = y1 - m * x1
        A = -m
        B = 1
        C = -b
        r = abs(C) / np.sqrt(A ** 2 + B ** 2)
    return theta, r


red = (0, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)

# red-black 이미지로 변환
def redImage(image): 
    img = image.copy()
    height = img.shape[0]
    width = img.shape[1]

    # red 좌표들만 따서 list 에 넣기
    points = []

    # red 좌표 아닌 부분은 black 으로 변환
    for i in range(height):
        for j in range(width):
            if not (np.array_equal(img[i, j], red)):
                img[i, j] = white
            else:
                points.append((i, j))
    redImg = img.copy()
    cv2.imwrite('redImg.jpg', redImg)
    return img


# 이미지에서 라인 가져와서 각도별로 clustering
def getLine(redImage) :
    red_points = redImage.copy()

    # canny 필터 적용 후 hough 알고리즘을 이용하여 후보 직선의 시작점과 끝점을 추출
    edges = cv2.Canny(red_points, 100, 150,apertureSize = 3)
    lines = hough(edges, 40, 150, 50, red_points)
    for points in lines:
        cv2.line(red_points, (points[0][0],points[0][1]), (points[0][2],points[0][3]), black, 3)
    lineImg = red_points.copy()
    cv2.imwrite('resultLine.jpg', lineImg)

    hough_angle = []
    for line in lines:
        hough_angle.append(round(calculate_angle([line[0][0], line[0][1]],[line[0][2], line[0][3]]) / 10 ))

    # 리스트 내의 값은 해당 각도의 lines의 index 값
    # clustering 한 각도들 중에서 최소 3개 이상인 각도의 index 를 이용해서 양 끝점 저장하기
    angleCnt = [ 0 for i in range(19)]
    angleLines = [[] for i in range(19)]

    i = 0
    # 같은 각도를 가진 절리의 개수 카운트
    for ind in hough_angle:
        # ind = 각 절리 각도값 / 10
        angleCnt[ind] += 1
        angleLines[ind].append(lines[i][0])
        i += 1
    print("angleCnt: ", angleCnt)
    cluster_points = []
    # 카운트한 값이 5 초과인 경우 해당 절리들은 하나의 절리군으로 인정
    for i in range(19):
        # i * 10 도 인 선분들의 개수가 10개 이상이면 count
        if (angleCnt[i] >= 5):
            cluster_points.append([angleLines[i]])
    # print("clusterPoints\n")
    # for point in cluster_points:
    #     print(point, "\n")
    
    cnt = 0
    half_points = [[] for i in range(len(cluster_points))]
    
    for pointSet in cluster_points :
        for points in pointSet[0] :
            halfPoints(points, half_points[cnt])
        cnt += 1
            
    count = 0
    result = []
    for half, cluster in zip(half_points, cluster_points):
        # print("half & clsuter\nhalf: ", half, "\ncluster: ",cluster , "\n")
        img = redImage.copy()
        # k means shift##########
        n_samples = 10
        side_points = np.array(half)
        bandwidth = estimate_bandwidth(half, quantile=0.6, n_samples=n_samples) 
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(half)
        cluster_centers = ms.cluster_centers_
        labels = ms.labels_
        #########################
        numCluster = len(cluster_centers)
        print('clsuter-center: ', cluster_centers)
        
        print("label:", labels)
        
        result_point = [[] for i in range(numCluster)]
        for labelValue in range(numCluster):
            index = 0
            point = [] # 해당 라벨값의 모든 절리 양 끝값
            for value in labels:
                if (value == labelValue):
                    point.append(cluster[0][index])
                index += 1
            # print("labelpoint: ", point)
            
            length = len(point)
            point.sort(key=lambda x:x[0])
            result_point[labelValue].append((point[0][0], point[0][1], point[length - 1][2], point[length - 1][3]))
            # print("resultPoint\n", result_point[labelValue][0], "\n")
            cv2.line(img, (result_point[labelValue][0][0], result_point[labelValue][0][1]), (result_point[labelValue][0][2], result_point[labelValue][0][3]), green, 3)
        
        imgUrl = "resultJoint" + str(count) + ".jpg"
        cv2.imwrite(imgUrl, img)
        result.append(result_point)

        count += 1
    return result
        
# joint 길이 구하기
def joint_length(joint_points, lens) :
    print('length joint_points: ', joint_points)

    jointset_length_list = []  
    for i, jointset in enumerate(joint_points[0]):#jointset 별로 길이를 다른 배열에 넣게 수정하였습니다
        joint_length_list = []
        for point in jointset:
            joint_angle = calculate_angle((point[0], point[1]), (point[2], point[3]))
            joint_length = distance((point[0], point[1]), (point[2], point[3]))
            # joint_length_list.append(joint_length)
            print('points : ', point)
            print('joint angle: ', joint_angle)
            print('joint length: ', joint_length)
            print()
        jointset_length_list.append(joint_length)
        
    print('jointset length list',jointset_length_list)    
    return jointset_length_list
    

    #joint set 간격 구하기
def joint_spacing(joint_points, lens) :
    print('get rho & spacing\n')
    jointset_spacings = []
    sorted_joint_points = []
    for i, jointset in enumerate(joint_points):
        # print("joint_spacing jointset: ", jointset)
        jointset_rho = []
        print("joint set",i)
 
        for point in jointset:
            print(point[0])
            _,joint_rho = getpolar(point[0][0], point[0][1], point[0][2], point[0][3])
            jointset_rho.append(joint_rho)
            print('rho : ', joint_rho)
            
        sorted_arg = np.argsort(np.array(jointset_rho))
        print(sorted_arg)
        jointset_rho = np.sort(np.array(jointset_rho))
        sortedpoints = []
        for i in range (0,len(jointset)):
            sortedpoints.append(jointset[sorted_arg[i]])
        print('jointset',i,'rho:',jointset_rho)
        sorted_joint_points.append(sortedpoints)
        jointset_spacings.append(np.diff(jointset_rho))
        print()

    print('jointset spacinig', jointset_spacings)
    real_length_list = []

    for i in range(len(jointset_spacings)):
        real_length = lens.real_length(jointset_spacings[i])
        # print('real joinset',i,'spacings', real_length)
        real_length_list.append(real_length)

    return real_length_list, sorted_joint_points


def make_data(joint_points, realDistance) :
    data = []
    lens = Lens(realDistance * 10,12,6.287,4050)
    spacing,sorted_joint = joint_spacing(joint_points, lens)
    
    # print(length[0][0])
    for num  in range(len(joint_points)) :
        # print("spacing: ", spacing)
        # print("sorted)joint: ", sorted_joint)
        # print("sorterd_joint: " , sorted_joint , "\n")
        length = joint_length(joint_points, lens)
        print(length[0])
        jointSetData = {
            "lines": [],
            "angles": [],
            "spacing": [],
            "length":  []
        }
        i = 0
        for point in sorted_joint[num]:
            jointSetData['lines'].append([[int(point[0][0]), int(point[0][1])], [int(point[0][2]),int(point[0][3])]])
            # jointSetData['length'].append(distance((point[0][0], point[0][1]), (point[0][2], point[0][3])))
            jointSetData['angles'].append(calculate_angle([point[0][0], point[0][1]], [point[0][2],point[0][3]]))
            i += 1
        jointSetData['spacing']=spacing[num]
        jointSetData['length']=length
        

        data.append(jointSetData)

    return data
