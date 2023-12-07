
original_img = cv2.imread(original_image)
resultImg = drawAll(original_img,data)
cv2.imshow("resultimg", resultImg)


#cv2.imwrite("resultimg",resultImg)
for i in range(0,len(data)):
    jointset_result = drawJointset(original_img,data,"jointset%d" %i, setnum= i)
    cv2.imshow("resultimg%d"%i, jointset_result)
    cv2.imwrite("resultjointset%d.jpg"%i,jointset_result)
plt.close('all')
stereonet = makeStereonet(data)
plt.savefig('stereonetImg.jpg')
plt.show()
dataFrame = getDataFrame(data)
print(dataFrame)
saveDataFrameAsImage(dataFrame,"table.jpg")



stop = time.time()
print("testing time :", round(stop - start, 3), "ms")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("testing time :", round(stop - start, 3), "ms")
cv2.waitKey(0)
cv2.destroyAllWindows()