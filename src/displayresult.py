#!/usr/bin/env python
# coding: utf-8


import math

import cv2
import mplstereonet
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageColor, ImageDraw, ImageFont


def imgshow(img,title = None):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()
def drawLines(img, lines, color=(0, 0, 255)):
    lineOnImg = img.copy()
    for line in lines:
        if line is not None:
            x1, y1 = line[0]
            x2, y2 = line[1]
            cv2.line(lineOnImg, (x1, y1), (x2, y2), color, 8)
    return lineOnImg



# 무지개색 목록 (빨간색부터 보라색까지)
#color = ['r','b','g','y','c','m', 'k']
#tableau color
# colors = [
#     (40,39,214), # tab:red
#     (180,119,31), # tab:blue
#     (44,160,44), #tab:green
#     (189,103,148), #tab:purple
#     (14,127,255), #tab:orange
#     (127,127,127), #tab:gray
#     (207,190,23), #tab:cyan
#     (75,86,140), #tab:brown
#     (194,119,227), #tab:pink
#     (34,189,188) #tab:olive
    
# ]
colors = ['#7FFF00','#FFD700','#FF69B4','#778899','#FF4500','#8B008B','#008B8B']


#mpl 기본 색상
# colors = [#(0,255,128)
#     (0, 0, 255),    # 빨간색
#     (255, 0, 0),    # 파란색
#     (0, 255, 0),    # 초록색
#     (0, 255, 255),  # 노란색
#     (255,255,0),    #시안
#     (255, 0, 255)   # 마젠타
# ]
# colors = [
#     (176,232,59),
#     (0, 185, 255),
#     (252, 99, 107),
#     (208, 175, 26),
#     (206, 103, 106)
    
# ]

#검출된 모든 절리들을 원본 이미지에 그림
def drawJoints(img,data,jointNum = None):
    lineimg = img.copy()
    for i, jointset in enumerate(data):
        if jointNum is not None:
            rgb = ImageColor.getcolor(colors[jointNum%len(colors)],"RGB")
            color = tuple(reversed(rgb))
        else:
            rgb = ImageColor.getcolor(colors[i % len(colors)],"RGB")
            color = tuple(reversed(rgb))
        lineimg = drawLines(lineimg,jointset["lines"],color)
#     imgshow(lineimg)
    return lineimg

font_path = "arialbd.ttf" #폰트는 더 나은것으로 바꿀수 있다.
fontsize = 20

def dist(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def rotateText(img,text_img, text, center, color, angle):
    height,width,_ = img.shape
    #line중심에 따라 text 왼쪽 시작위치 설정
    #800, 450 사이즈 기준, 이미지를 다양한 크기로 받는다면 adaptable변경
    text_location = (center[0]-40,center[1]-34) 
    #텍스트 폰트 설정
    font = ImageFont.truetype(font_path, fontsize)
    
    
    #유니코드를 쓰기위해 PIL Image로 변경
    pil_image = Image.fromarray(cv2.cvtColor(text_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # 배경 색상으로 사각형 그리기
    bbox = draw.textbbox(text_location,text, font=font)
    draw.rectangle(bbox, fill=color)
    
    #텍스트 쓰고 다시 cv numpyarray로 변경
    draw.text(text_location, text, font=font, fill=(255,255,255))
    text_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    #글씨와 선 그린 이미지 회전시키기
    rotateAngle = -(angle)+180
    if angle<90:
        rotateAngle +=180
    M = cv2.getRotationMatrix2D(center, rotateAngle, 1)
    text_img = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]))
    
    if np.any(text_img[:height,:] != 0) or np.any(text_img[height*2:,:,:] != 0) or np.any(text_img[:,:width,:] != 0) or np.any(text_img[:,width*2:,:] != 0):
        M = cv2.getRotationMatrix2D(center, 180, 1)
        text_img2 = cv2.warpAffine(text_img, M, (text_img.shape[1], text_img.shape[0]))
        text_img2 = text_img2[height:height*2, width:width*2, :]
        text_img = text_img[height:height*2, width:width*2, :]
        if np.count_nonzero(text_img2)>np.count_nonzero(text_img):
            text_img = text_img2
    else:
        text_img = text_img[height:height*2, width:width*2, :]
            
    
    #회전 시킨거 마스크 따고 원본이미지 전처리하고 그위에 그리기
    gray_text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray_text_img,1,255,cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bg_img = cv2.bitwise_and(img,mask)
    combined_img = cv2.add(bg_img, text_img)
    
    return combined_img
    


def drawLength(img, text, line, angle, color):
    #선 중심 좌표 보정값 구하고 눈금 선 좌표 구하기
    height,width,_ = img.shape
    x1,y1 = line[0]
    x2,y2 = line[1]
    center = (int(width + (x1 + x2)//2),int(height + (y1 + y2)//2))
    linedist = dist(x1,y1,x2,y2)
    rulerx1 = int(center[0]-linedist//2)
    rulerx2 = int(center[0]+linedist//2)
    rulery1 = int(center[1]-20)
    rulery2 = int(center[1]-20)
    
    #원본의 3*3 크기의 빈 이미지 만들어서 위에 그리기
    text_img = np.zeros((height*3,width*3,3), dtype=np.uint8)
    cv2.line(text_img, (rulerx1, rulery1), (rulerx2, rulery2), color, 8)
    cv2.line(text_img,(rulerx1,rulery1),(rulerx1,center[1]), color, 8)
    cv2.line(text_img,(rulerx2,rulery2),(rulerx2,center[1]), color, 8)
    
    
    
    result = rotateText(img,text_img, text, center, (color[2],color[1],color[0]), angle)
    
    return result


#이미지에 선분 및 길이, 각도 데이터 표시
def drawInfo(img,data):
    text_img = img.copy()
    # text_img = drawJoints(text_img,data)
    for jointset in data:
        angles = jointset["angles"]
        color = (0,0,255)#(0, 185, 255) #길이 표시선 컬러
        for i in range(0,len(jointset["lines"])):
            angle = angles[i]
            line = jointset["lines"][i]
            text = f'{round(jointset["length"][i])}cm, {angle}°'
            #선 중심 좌표, 보정값으로 눈금 선 좌표 구하기
            height,width,_ = text_img.shape
            x1,y1 = line[0]
            x2,y2 = line[1]
            center = (width + (x1 + x2)//2,height + (y1 + y2)//2)
            linedist = dist(x1,y1,x2,y2)
            rulerx1 = int(center[0]-linedist//2)
            rulerx2 = int(center[0]+linedist//2)
            rulery1 = int(center[1]-20)
            rulery2 = int(center[1]-20)

            #원본의 3*3 크기의 빈 이미지 만들어서 위에 그리기
            canvas_img = np.zeros((height*3,width*3,3), dtype=np.uint8)
            cv2.line(canvas_img, (rulerx1, rulery1), (rulerx2, rulery2), color, 8)
            cv2.line(canvas_img, (rulerx1,rulery1),(rulerx1,center[1]), color, 8)
            cv2.line(canvas_img,(rulerx2,rulery2),(rulerx2,center[1]), color, 8)

            text_img = rotateText(text_img,canvas_img,text, center, (color[2],color[1],color[0]), angle)
#     imgshow(text_img,"continunity, degree")
    return text_img


def nearest_points_on_line_segments2(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = _A[0] * _B[1] - _A[1] * _B[0]
    denom = cross**2

    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)

    t = (b0 - a0)
    detA = _A[0] * t[1] - _A[1] * t[0]
    detB = _B[0] * t[1] - _B[1] * t[0]

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0)
    pB = b0 + (_B * t1)

    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA.astype(int), pB.astype(int)#,np.linalg.norm(pA-pB)


def calculate_angle(x1, y1, x2, y2):
    # 두 점 사이의 상대적인 x, y 좌표 계산
    dx = x2 - x1
    dy = y2 - y1
    if(dy==0):
        return 0
    
    # 아크탄젠트 계산
    angle_rad = math.atan2(dy, dx)
    
    # 라디안 값을 도 단위로 변환
    angle_deg = math.degrees(angle_rad)
    
    # 각도가 음수일 경우 양수로 변환
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg



#lines의 간격, spacing값 텍스트로 표시하는 함수
def drawspacing(img,data):
    spacing_img = img.copy()
    spacingcolor = (255,0,0)#(176,232,59)
    for jointset in data:
        if(len(jointset["lines"])==1):continue
        for i,spacing in enumerate(jointset["spacing"]):
            p1 = np.array(jointset["lines"][i][0])
            p2 = np.array(jointset["lines"][i][1])
            p3 = np.array(jointset["lines"][i+1][0])
            p4 = np.array(jointset["lines"][i+1][1])
            #두 선분의 최단점을 point1, point2로
            point1, point2 = nearest_points_on_line_segments2(p1,p2,p3,p4,clampAll=True)
            spacingtxt = f'{round(spacing)}cm'
            cv2.arrowedLine(spacing_img, point1, point2, spacingcolor, 6, tipLength = 0.07)
            cv2.arrowedLine(spacing_img, point2, point1, spacingcolor, 6, tipLength = 0.07)
            height,width,_ = spacing_img.shape
            center = (int(width+(point1[0]+point2[0])//2),int(height+(point1[1]+point2[1])//2))
            angle = calculate_angle(point1[0],point1[1],point2[0],point2[1])
            text_img = np.zeros((height*3,width*3,3), dtype=np.uint8)
            spacing_img=rotateText(spacing_img,text_img, spacingtxt, center, (spacingcolor[2],spacingcolor[1],spacingcolor[0]), angle)

#     imgshow(spacing_img,"spacing")
    return spacing_img

#모든 Joints line들 그리고 각도, 길이 표시, 
def drawAll(img,data, title = None):
    resultimg = img.copy()
    resultimg = drawJoints(resultimg,data)
    resultimg = drawInfo(resultimg,data)
    resultimg = drawspacing(resultimg,data)
    return resultimg


#Joinset마다 가져올 수 있다. setnum이 None인 경우 순서대로 출력
def drawJointset(img,data,title=None, setnum=None):
    if setnum is None:
        for i in range(0,len(data)):
            resultimg = img.copy()
            setData = []
            setData.append(data[i])
            resultimg = drawJoints(resultimg,setData,i)
            resultimg = drawInfo(resultimg,setData)
            resultimg = drawspacing(resultimg,setData)
            if title is None:
                imgshow(resultimg,"Jointset"+str(i))
            else:
                imgshow(resultimg,title+" - Jointset"+str(i))
    elif setnum <len(data):
        resultimg = img.copy()
        setData = []
        setData.append(data[setnum])
        resultimg = drawJoints(resultimg,setData,setnum)
        resultimg = drawInfo(resultimg,setData)
        resultimg = drawspacing(resultimg,setData)
    else: return img
    return resultimg


import pandas as pd


def calculate_average_spacing(spacings):
    if len(spacings) > 0:
        inverse_sum = sum(1 / spacing for spacing in spacings)
        average_spacing = 1 / inverse_sum
        return round(average_spacing, 1), round(inverse_sum, 3)
    else:
        return 0, 0

    
def getDataFrame(data):
    # Set the display options to show all columns and rows without truncation
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    
    # 데이터 리스트의 길이 확인
    data_length = len(data)

    # data의 평균 간격과 밀도 계산
    averages_data = []
    for d in data:
        spacing = d.get("spacing", [])
        average_spacing, inverse_sum = calculate_average_spacing(spacing)
        averages_data.append({"spacing_mean": average_spacing, "density_mean": inverse_sum})

    averages_data = [
        {
            "angles_mean": round(sum(joint.get("angles", [])) / len(joint.get("angles", [])), 1) if joint.get("angles") else None,
            "length_mean": round(sum(joint.get("length", [])) / len(joint.get("length", [])), 1) if joint.get("length") else None,
            "spacing_mean": averages_data[i]["spacing_mean"],
            "density_mean": averages_data[i]["density_mean"]
        }
        for i, joint in enumerate(data)
    ]

    # 데이터프레임 생성
    df = pd.DataFrame({
        "평균 각도": [data["angles_mean"] for data in averages_data],
        "평균 길이": [data["length_mean"] for data in averages_data],
        "평균 간격": [data["spacing_mean"] for data in averages_data],
        "평균 밀도": [data["density_mean"] for data in averages_data],
    })

    # 데이터프레임의 index 이름을 설정합니다.
    index_names = [f"jointset {i+1}" for i in range(len(df))]
    df.index = index_names
    #df = df.style.highlight_max()
    
    ##for index 

    # 데이터프레임을 출력합니다.
    return df



#이미지의 y축이 북쪽인 것을 기준으로 angle을 Strike로 변환
def angleToStrike(angle):
    return (angle+90)%180

#이미지의 y축이 북쪽인 것을 기준으로 angle을 Dip으로 변환
def angleToDip(angle):
    return (angle+90)%180

#Stereonet을 그리는 함수
def makeStereonet(data, title = None, option = 1):
    jointsetstrikes = []
    jointsetdips = []
    jointdips = []
    jointstrikes = []
    #colors = ['#7FFF00','#FFD700','#ADFF2F','#FF69B4','#778899','#FF4500','#8B008B','#008B8B']
    #['tab:green','tab:purple','tab:orange','tab:gray','tab:cyan','tab:brown','tab:pink', 'tab:olive']
    colorcnt = len(colors)
    for jointset in data:
        #이 부분은 strike를 구하게되면 수정할것
        jointdips.append(list(map(angleToDip,jointset["angles"])))
        jointstrikes.append(list(map(angleToStrike,jointset["angles"]))) 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='stereonet')
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    # Create x and y ticks every 10°
    y_ticks = np.arange(ymin, ymax, np.deg2rad(10))
    x_ticks = np.arange(ymin, ymax, np.deg2rad(10))

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    if title is not None:
        ax.set_title(title, y=1.25, fontsize=20)
    for i, (strikes, dips) in enumerate(zip(jointdips,jointstrikes)):
        color = colors[i%colorcnt]
        if option == 1:
            for j, (strike, dip) in enumerate(zip(strikes, dips)):
                ax.plane(strike, dip, c=color, label='Joint set%d(%d) %03d/%02d' % (i+1, j, strike, dip))
                ax.pole(strikes, dips, c=color,alpha=0.8)
        elif option == 2:
            meanstrike = np.mean(strikes)
            meandip = np.mean(dips)
            ax.plane(meanstrike, meandip, c=color, label='Joint set%d %03d/%02d' % (i+1, meanstrike, meandip))
            ax.pole(meanstrike, meandip, marker='*', markersize = 9, c=color,alpha=0.8,label='Joint set%d %03d/%02d' % (i+1, meanstrike, meandip))
            ax.pole(strikes, dips, c=color,alpha=0.8)
        elif option == 3:
            meanstrike = np.mean(strikes)
            meandip = np.mean(dips)
            ax.plane(meanstrike, meandip, c=color, label='Joint set%d %03d/%02d' % (i+1, meanstrike, meandip))
            ax.pole(meanstrike, meandip,c=color,alpha=0.8)
    ax.legend(fontsize=15)
    ax.set_azimuth_ticks([0, 90, 180, 270])
    ax.grid()

    return fig

def saveDataFrameAsImage(dataframe, output_filename):
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.table(cellText=dataframe.values,
              colLabels=dataframe.columns,
              rowLabels=dataframe.index,
              loc='center',
              cellLoc='center')
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.5)
    plt.close()
