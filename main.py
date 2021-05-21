import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr


def main():
    # Ornek olarak bir gorseli alip grayscale'a donusturuyoruz.
    img = cv2.imread('assets/image1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

    b_filter = cv2.bilateralFilter(gray, 11, 17, 17) # Noise dusuruyoruz.
    edged = cv2.Canny(b_filter, 30, 200) # Kenar tespiti.

    # gorsellerde countour(sekil) tespiti.
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        # kare sekillerinin tespiti icin ideal parametre verilir.
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)

    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))

    cropped_image = gray[x1:x2+1, y1:y2+1]

    # OCR teknolojisi kullanarak kirpilan gorseldeki plaka metnini kaziyoruz.

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    print(result)

    # OCR SONU

    # maskeleme ve kirpma islemi bitince islenen goruntuyu goster.

    # kazinan plaka numarasi.
    text = result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img,
                      text=text,
                      org=(approx[0][0][0], approx[1][0][1]+75),
                      fontFace=font,
                      fontScale=1,
                      color=(0, 255, 0),
                      thickness=2)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
