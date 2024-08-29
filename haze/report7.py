import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inverse_rgb_image_gray(inverse_image, image):
    b, g, r = cv2.split(inverse_image)
    main_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b1 = main_image - np.sqrt(b)
    g1 = main_image - np.sqrt(g)
    r1 = main_image - np.sqrt(r)

    pow_b = cv2.pow(b1, 2)
    pow_g = cv2.pow(g1, 2)
    pow_r = cv2.pow(r1, 2)

    sum_rgb = (pow_b + pow_g + pow_r)

    euq_rgb_gray = cv2.sqrt(sum_rgb)
    euq_rgb_gray_clp = np.clip(euq_rgb_gray, 0, 1)

    euq_rgb_gray = cv2.normalize(euq_rgb_gray, None, 0, 1, cv2.NORM_MINMAX)
    euclid_nor = np.clip(euq_rgb_gray, 0, 1)

    em_add = cv2.merge((b1, g1, r1))
    em_add = cv2.normalize(em_add, None, 0, 1, cv2.NORM_MINMAX)
    # cv2.imshow("merge",em_add)
    # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/em_add.png", em_add*255)
    # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/euq_rgb_gray_clp.png", euq_rgb_gray_clp*255)
    return euq_rgb_gray_clp, em_add, euclid_nor

def inverse_rgb_image_gray1(inverse_image, image):
    b, g, r = cv2.split(inverse_image)
    main_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b1 = main_image - np.sqrt(b)- 0.5*b
    g1 = main_image - np.sqrt(g)- 0.5*g
    r1 = main_image - np.sqrt(r)- 0.5*r
    #
    # b1 = main_image - np.sqrt(b)- b
    # g1 = main_image - np.sqrt(b)- g
    # r1 = main_image - np.sqrt(b)- r


    pow_b = cv2.pow(b1, 2)
    pow_g = cv2.pow(g1, 2)
    pow_r = cv2.pow(r1, 2)

    sum_rgb = (pow_b + pow_g + pow_r)

    euq_rgb_gray = cv2.sqrt(sum_rgb)
    euq_rgb_gray_clp = np.clip(euq_rgb_gray, 0, 1)

    euq_rgb_gray = cv2.normalize(euq_rgb_gray, None, 0, 1, cv2.NORM_MINMAX)
    euclid_nor = np.clip(euq_rgb_gray, 0, 1)

    em_add = cv2.merge((b1, g1, r1))
    em_add = cv2.normalize(em_add, None, 0, 1, cv2.NORM_MINMAX)
    # cv2.imshow("merge",em_add)
    # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/em_add.png", em_add*255)
    # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/euq_rgb_gray_clp.png", euq_rgb_gray_clp*255)
    return euq_rgb_gray_clp, em_add, euclid_nor

def image_inverse(image):
    b, g, r = cv2.split(image)
    b = 1 - b
    g = 1 - g
    r = 1 - r
    image_sum = cv2.merge((b, g, r))
    image_sum = cv2.normalize(image_sum, None, 0, 1, cv2.NORM_MINMAX)
    image_sum = np.clip(image_sum, 0, 1)
    # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/image_inv.png", image_sum*255)
    return image_sum

def image_sqrt(image):
    result_image = np.sqrt(image)
    # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/image_sqrt.png", result_image*255)
    return result_image

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def DarkChannel1(im, scales):
    b, g, r = cv2.split(im)
    min_channel = cv2.min(cv2.min(r, g), b)
    dark_sum = None
    for sz in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark_channel = cv2.erode(min_channel, kernel)
        if dark_sum is None:
            dark_sum = dark_channel.astype(np.float32)
        else:
            dark_sum += dark_channel
    dark_average = dark_sum / len(scales)
    return dark_average.astype(min_channel.dtype)


# def AtmLight(im, dark, top_percent=0.1):
#     """
#     개선된 대기광 추정 함수
#     :param im: 입력 이미지 (H, W, C)
#     :param dark: 다크 채널 이미지 (H, W)
#     :param top_percent: 가장 밝은 픽셀을 선택할 비율
#     :return: 추정된 대기광 값
#     """
#     [h, w] = im.shape[:2]
#     imsz = h * w
#     numpx = int(max(math.floor(imsz * top_percent), 1))  # 가장 밝은 픽셀의 개수
#     darkvec = dark.reshape(imsz)
#     imvec = im.reshape(imsz, 3)
#
#     # 다크 채널에서 가장 밝은 픽셀 선택
#     indices = darkvec.argsort()[::-1][:numpx]  # 가장 밝은 픽셀의 인덱스
#
#     # 선택된 픽셀에서 대기광의 평균 값 계산
#     atmsum = np.zeros([1, 3])
#     for ind in indices:
#         atmsum += imvec[ind]
#     A = atmsum / numpx
#     return A
def AtmLight(im, dark, dynamic_top_percent=True, contrast_enhanced_dark_channel=True):
    """
    개선된 대기광 추정 함수
    :param im: 입력 이미지 (H, W, C)
    :param dark: 다크 채널 이미지 (H, W)
    :param dynamic_top_percent: 동적 top_percent 사용 여부
    :param contrast_enhanced_dark_channel: 향상된 다크 채널 사용 여부
    :return: 추정된 대기광 값
    """
    [h, w] = im.shape[:2]
    imsz = h * w
    if dynamic_top_percent:

        average_brightness = np.mean(im)
        top_percent = min(max(0.01, 1 - average_brightness / 255), 0.4)
    else:
        top_percent = 0.1
    numpx = int(max(math.floor(imsz * top_percent), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    # if contrast_enhanced_dark_channel:
    #     # 로컬 및 명암 대비를 고려하여 다크 채널 향상
    #     enhanced_dark = cv2.Laplacian(dark, cv2.CV_32F)  # 예시 향상 로직
    #     darkvec = enhanced_dark.reshape(imsz)

    # 다크 채널에서 가장 밝은 픽셀 선택
    indices = darkvec.argsort()[::-1][:numpx]


    atmsum = np.zeros([1, 3])
    for ind in indices:
        atmsum += imvec[ind]
    A = atmsum / numpx
    return A

def TransmissionEstimate(im, A, scales):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * DarkChannel(im3, scales)
    return transmission



def main_tmap(image,sz):
    image_main = image
    dark = DarkChannel(image_main, sz)
    #cv2.imshow("dark",dark)
    A = AtmLight(image_main, dark)
    tmap_result = TransmissionEstimate(image_main, A, sz)
    #cv2.imshow("tmap_result",tmap_result)
    return A, tmap_result, dark



def inverse_sqrt_tamp(image,sz):
    inverse = image_inverse(image)
    sqrt_image = image_sqrt(inverse)
    dark = DarkChannel(sqrt_image, sz)
    #cv2.imshow("dark1",dark)
    A = AtmLight(sqrt_image, dark)
    tmap_result = TransmissionEstimate(sqrt_image, A, sz)
    #cv2.imshow("tmap_result1",tmap_result)
    return dark, tmap_result

def inverse_gray_tmap(image1,sz):
    inverse = image_inverse(image1)
    inverse_gray1, m_add, euclid_nor = inverse_rgb_image_gray(inverse, image1)
    # inverse_gray2, _, _ = inverse_rgb_image_gray1(inverse, image1)

    # cv2.imshow("merge",inverse_gray3)
    # cv2.waitKey(0)
    inverse_gray = inverse_gray1
    # inverse_gray = (inverse_gray1+ inverse_gray2)*0.5
    dark = DarkChannel(m_add, sz)
    #cv2.imshow("dark2",dark)
    A = AtmLight(m_add, dark)
    tmap_result = TransmissionEstimate(inverse, A, sz)
    #cv2.imshow("tmap_result2",tmap_result)
    return dark, tmap_result, inverse_gray, inverse_gray1



def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 16  # max 16
    eps = 0.0001;
    t = Guidedfilter(gray, et, r, eps);
    return t



def Recover(im, t, A, tx=0.1,  adjust_saturation=False):
    res = np.empty(im.shape, im.dtype);
    t = cv2.max(t, tx)

    for ind in range(3):  # Assuming im is in BGR format
        res[:, :, ind] = ((im[:, :, ind] - A[0, ind]) / t) + A[0, ind]

    final = res

    if adjust_saturation:
        # Convert to HSV color space to adjust the saturation
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Increase saturation by a factor, ensuring it doesn't exceed 255
        s = cv2.min(s * 1.5, 255).astype(hsv.dtype)

        final = cv2.merge([h, s, v])
        final = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)

    return final


def BF(gray,et):
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray) / 255;
    r = 16  # max 16
    eps = 0.0001;

    t = Guidedfilter(gray, et, r, eps);
    return t
# def BF(tmap0):
#     tmap0_uint8 = np.uint8(tmap0 * 255)
#
#     # Apply bilateral filter
#     # Let's assume d=9, sigmaColor=75, sigmaSpace=75 as starting values
#     tmap0_blur = cv2.bilateralFilter(tmap0*255, 16, 150, 150)
#
#     # If you are working with floating point images scaled between 0 and 1, convert back after filtering
#     tmap0_blur = tmap0_blur / 255
#
#     return tmap0_blur

def hist(img, le):
    img = np.uint8(img)
    channels = cv2.split(img)
    colors = ['b', 'g', 'r']
    for ch, color in zip(channels, colors):
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
        plt.plot(hist, color=color, label=le)
    plt.legend()


import numpy as np
from skimage.transform import resize


def multiscale_tmap(image_main, scales):
    multiscale_tmaps = []
    for scale in scales:

        scaled_image = resize(image_main, (int(image_main.shape[0] * scale), int(image_main.shape[1] * scale)))
        size=1

        A, tmap, dark_channel = main_tmap(scaled_image, size)
        dark1, tmap1, euq_tmap, euq_tmap2 = inverse_gray_tmap(scaled_image, size)
        dark2, tmap2 = inverse_sqrt_tamp(scaled_image, size)

              
        tmap = BF(scaled_image,tmap)
        tmap1 = BF(scaled_image,tmap1)
        tmap2 = BF(scaled_image,tmap2)

        tmap = np.clip(tmap, 0, 1)
        tmap1 = np.clip(tmap1, 0, 1)
        tmap2 = np.clip(tmap2, 0, 1)


        tmap_resized = resize(tmap, image_main.shape[:2])
        tmap1_resized = resize(tmap1, image_main.shape[:2])
        tmap2_resized = resize(tmap2, image_main.shape[:2])
        tmap3_resized = resize(euq_tmap, image_main.shape[:2])

        multiscale_tmaps.append((tmap_resized, tmap1_resized, tmap2_resized,tmap3_resized))


    final_tmap = np.mean([tmaps[0] for tmaps in multiscale_tmaps], axis=0)
    final_tmap1 = np.mean([tmaps[1] for tmaps in multiscale_tmaps], axis=0)
    final_tmap2 = np.mean([tmaps[2] for tmaps in multiscale_tmaps], axis=0)
    fianl_euq_tmap = np.mean([tmaps[3] for tmaps in multiscale_tmaps], axis=0)

    return final_tmap, final_tmap1, final_tmap2, fianl_euq_tmap




from tqdm import tqdm
if __name__ == "__main__":
    path = 'D:\\Hazedata\\ITTS\\JPEGImages'
    path1 = 'D:\\Hazedata\\ITTS\\Our'
    # path ='C:/Users/aasaa/Downloads/RTTS/JPEGImages'
    # path1 ='C:/Users/aasaa/Downloads/RTTS/b'
    # path1 ='D:/Hazedata/1/1'
    # path ='D:/Fog/Fog'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    list_img = os.listdir(path)


    for k123 in tqdm(range(0, len(list_img)), desc="Processing Images"):
        image = list_img[k123]
        i = list_img[k123].split('.png')
        i = i[0]
        ex = "png"

        image_main1 = cv2.imread("{}/{}.{}".format(path, i, ex))
        image_main = image_main1.astype(np.float32) / 255



        A, _, _ = main_tmap(image_main,1)
        tmap0,tmap1,tmap2,euq_tmap = multiscale_tmap(image_main, [1.0, 0.5, 0.25])
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/tmap0.png", tmap0 * 255)
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/tmap1.png", tmap1 * 255)
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/tmap2.png", tmap2 * 255)
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/euq_tmap.png", euq_tmap * 255)
        t_mul = (tmap1  *tmap2)
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/t_mul.png", t_mul * 255)

        #cv2.imshow("t_mul",t_mul)
        n_sq = t_mul ** 0.5
        # cv2.imwrite(/"C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/n_sq.png", n_sq * 255)
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/n_sq.png", n_sq * 255)

        #cv2.imshow("n_sq",n_sq)
        tmap_3 = (tmap0 + n_sq) * 0.5
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/tmap3.png", tmap_3 * 255)
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/tmapavg.png", tmap_3 * 255)

        #cv2.imshow("tmap_3",tmap_3)
        # max_image = np.minimum(euq_tmap, euq_tmap2)
        tmap_gamma0 = (euq_tmap + tmap_3) * 0.5
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/tmap3.png", tmap_gamma0 * 255)

        # # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/paper/gamma.png", tmap_gamma0 * 255)
        a = 0.1
        # #cv2.imshow("tmap_gamma0",tmap_gamma0)
        k21 = np.sqrt((1 - (a + np.min(tmap0))) / (np.max(tmap0) + (np.max(tmap0) * (1 - (np.min(tmap0) + a)))))  # 사용
        tmap_gamma1 = tmap_gamma0 ** k21
        # cv2.imwrite("C:/Users/aasaa/PycharmProjects/pythonProject/JY/original/pp/0.9.png", tmap_gamma1 * 255)

        new_tmap = TransmissionRefine(image_main, tmap_gamma1)
        #new_recover = Recover(image_main, tmap_gamma1, A)

        new_recover = Recover(image_main, new_tmap, A)
        cv2.imwrite("{}\{}.png".format(path1, i), new_recover*255)
