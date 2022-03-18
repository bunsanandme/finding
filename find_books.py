import cv2

image = cv2.imread("example2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

edged = cv2.Canny(gray, 10, 250)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)

(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
		total += 1

cv2.imshow("Output", image)
cv2.waitKey(0)
