import cv2
import time
import numpy as np

## 필터 값 모음
bilateralValues = [[90, 40, 100],
                   [50, 30, 80],
                   [70, 50, 100],
                   [90, 30, 90],
                   [80, 40, 80],
                   [70, 50, 80]]

adaptiveValues = [[91, 13],
                  [91, 17]]

img_url = "https://bclops.s3.ap-northeast-2.amazonaws.com/input.png"

## 이미지 경로 설정
NUMBER = "0000"
IMAGE = "evalutate/result/png/input_" + NUMBER + ".png"
# IMAGE = img_url 
IMAGE_AI = "evalutate/return/" + "output" + ".png"
IMAGE_OUTPUT = "Evalutate/return/" + "output" + "_AND.png"

## 이미지 로드
src = cv2.imread(IMAGE, cv2.IMREAD_ANYCOLOR)
origin = src.copy()
src = cv2.resize(origin, dsize=(1024, 512), interpolation=cv2.INTER_AREA)

## 타이머 작동
start = time.time()
# cv2.imshow("src", src)

bestPercent_AI, bestPercent_CV = 0, 0
bestResultImageAI = src.copy()

for t in range(len(bilateralValues)):
    # 1. grayscale
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 2. bilateral filter
    bValue1, bValue2, bValue3 = bilateralValues[t][0], bilateralValues[t][1], bilateralValues[t][2]
    dst_bilateral = cv2.bilateralFilter(src_gray, bValue1, bValue2, bValue3)

    # 3. adaptive threshold
    atValue1, atValue2 = adaptiveValues[0][0], adaptiveValues[0][1]
    dst_adaptiveThreshold = cv2.adaptiveThreshold(dst_bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                  atValue1, atValue2)

    # 4. closing - dilation
    open_kernel = np.ones((1, 1), np.uint8)
    close_kernel = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(dst_adaptiveThreshold, open_kernel, iterations=1)
    dilation = cv2.erode(dilation, close_kernel, iterations=1)

    # 5. median blur
    median = cv2.medianBlur(dilation, 3)

    # cv output
    filterResult = median
    cv_result = src.copy()
    #     cv2.imshow("sample", filterResult)
    for i in range(len(filterResult)):
        for j in range(len(filterResult[0])):
            if filterResult[i][j] == 0:
                cv_result[i][j][0], cv_result[i][j][1], cv_result[i][j][2] = 0, 0, 255
    #     cv2.imshow("sample2", cv_result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # AI image load & grayscale
    src_ai = cv2.imread(IMAGE_AI, cv2.IMREAD_ANYCOLOR)
    src_ai = cv2.resize(src_ai, dsize=(1024, 512), interpolation=cv2.INTER_AREA)

    # binary threshold (green - binary)
    #     src_ai = cv2.cvtColor(src_ai, cv2.COLOR_BGR2HSV)
    #     src_ai = cv2.inRange(src_ai, (50, 100, 0), (80, 255, 255))

    ret, src_ai_threshold = cv2.threshold(src_ai, 0, 255, cv2.THRESH_BINARY)

    #     cv2.imshow("sample_AI",src_ai_threshold)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # AND 연산
    and_result = src.copy()
    overlapped, AI_found, CV_found = 0, 0, 0

    for i in range(len(src_ai_threshold)):
        for j in range(len(src_ai_threshold[0])):
            if src_ai_threshold[i][j].all() == 0 and filterResult[i][j] == 0:
                overlapped += 1
                and_result[i][j][0], and_result[i][j][1], and_result[i][j][2] = 0, 0, 255
            if src_ai_threshold[i][j].all() == 0:
                AI_found += 1
            if filterResult[i][j] == 0:
                CV_found += 1
    #     cv2.imshow("and output",and_result)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    overlapped_percent_AI = overlapped / AI_found * 100
    overlapped_percent_CV = overlapped / CV_found * 100

    print("overlaps - AI", round(overlapped_percent_AI, 3), " CV", round(overlapped_percent_CV, 3))
    if bestPercent_AI < overlapped_percent_AI:
        bestPercent_AI = overlapped_percent_AI
        bestResultImageAI = and_result.copy()
        print("AI - updated")

stop = time.time()
print("best intersection - AI", round(bestPercent_AI, 3))
cv2.imshow("AI result", bestResultImageAI)
print("testing time :", round(stop - start, 3), "ms")

## 이미지 저장
# cv2.imwrite(IMAGE_OUTPUT, bestResultImageAI)

cv2.waitKey(0)
cv2.destroyAllWindows()
