import cv2
import numpy as np
import requests
import shutil


def get_digits_from_captcha(image, show=False, scale=3, checkarea=True):
    # Convert the image to HSV and take the h values, which we'll use to distinguish the digits
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, _, _ = cv2.split(image_hsv)
    # Count the occurrences (pixels) with each hue, eg. bins[50] will store num. pixels with hue 50
    bins = np.bincount(h.flatten())
    # Get the top 5 must frequent hues, but not the most common as that's the white background
    peaks = bins.argsort()[-6:][::-1][1:]


    # image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # values, counts = np.unique(image_hsv.copy().reshape(-1, 3), axis=0, return_counts=True)
    # # ind = np.argpartition(-counts, kth=6)[1:6]
    # ind = counts.argsort()[-8:][::-1][1:]
    # peaks = values[ind]



    # Our list which will store the digit images
    digits = []

    overall = image.copy()
    digits_found = 0

    # Now go through each of the top hues and find the digit corresponding to it
    for peak_hue in peaks:
        if digits_found == 5:
            break
        # Get our mask which is pixels with the matching hue
        low = np.array(peak_hue)
        high = np.array(peak_hue)
        mask = cv2.inRange(h, low, high)

        # low = peak_hue
        # high = peak_hue
        # mask = cv2.inRange(image_hsv, low, high)


        # And use it to extract the corresponding part of the original colour image
        # blob = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow('Colour', get_scaled_image(blob, 2))
        # cv2.waitKey(0)

        # Get the contours which describes the outline of our mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Lists which will store the extremities of where we find the hue so we can get the bounding box
        min_xs = []
        min_ys = []
        max_xs = []
        max_ys = []
        # print('num', len(contours))
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 3 and checkarea:
                continue
            # print('Area', i, area)
            # For each contour, get its bounding box and store the extremities
            bbox = cv2.boundingRect(contour)
            min_xs.append(bbox[0])
            min_ys.append(bbox[1])
            max_xs.append(bbox[0] + bbox[2])
            max_ys.append(bbox[1] + bbox[3])

        if len(min_xs) == 0 or len(min_ys) == 0 or len(max_xs) == 0 or len(max_ys) == 0:
            continue
        else:
            digits_found += 1

        # Now calculate the extremities of the bounding box
        min_x = min(min_xs)
        max_x = max(max_xs)
        min_y = min(min_ys)
        max_y = max(max_ys)

        # Get a mask which will have all the contours
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, contours, -1, 255, -1)

        # Get the region of the image corresponding to this colour
        region = image.copy()[min_y:max_y, min_x:max_x]

        digits.append((region, min_x))

        cv2.rectangle(overall, (min_x, min_y), (max_x, max_y), (0, 0, 0), 1)

    digits.sort(key=lambda tup: tup[1])
    digits = [region for region, start in digits]

    if show:
        cv2.imshow('Digits marked', get_scaled_image(overall, scale))
        cv2.waitKey(5)

    return digits


def scale_and_grey_digit(digit):
    # Convert the image to grayscale, then to black and white
    gray_digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    _, black_white_image = cv2.threshold(gray_digit, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Set our images to be 20x20
    desired_size = 20

    # Get the initial size of the digit (height, width)
    old_size = black_white_image.shape[:2]

    # Get the scale for our updated image
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image, passing in (width, height)
    black_white_image = cv2.resize(black_white_image, (new_size[1], new_size[0]))

    # Calculate the width and height additions to make the image the correct size
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    t, b = delta_h // 2, delta_h - (delta_h // 2)
    l, r = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    resized_black_white_image = cv2.copyMakeBorder(black_white_image, t, b, l, r, cv2.BORDER_CONSTANT, value=color)

    return resized_black_white_image


def get_new_captcha():
    url = 'https://projecteuler.net/captcha/show_captcha.php'
    res = requests.get(url, stream=True)

    if res.status_code != 200:
        print('Failed to fetch image')
    else:
        with open('new_captcha.png', 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    new_captcha = cv2.imread('new_captcha.png')
    return new_captcha


def get_scaled_image(image, scale):
    old_size = image.shape[:2]
    new_size = tuple([int(x * scale) for x in old_size])

    # Resize the image, passing in (width, height)
    scaled_image = cv2.resize(image, (new_size[1], new_size[0]))
    return scaled_image
