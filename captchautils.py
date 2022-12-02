import cv2
import numpy as np
import requests
import shutil


def get_digits_from_captcha(image, show=False, scale=3, showDigits=True, debug=False):
    """
    Takes a captcha and returns the individual digits
    :param image: The captcha as requested from online
    :param show: Optional argument to show the Captcha with identified digits boxed
    :param scale: Optional argument for the scale of the shown image
    :param scale: Optional argument for if show=True, whether to show individual digits too
    :param debug: Optional argument for showing areas detected for each individual hue
    :return: List of the images of the 5 individual digits
    """
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

        hue_boxes = image.copy()
        # print('num', len(contours))
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print('Area for hue:', peak_hue, ':', i, ':', area)
            if area < 3:
                continue

            bbox = cv2.boundingRect(contour)

            cv2.rectangle(hue_boxes, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 0), 1)

            min_xs.append(bbox[0])
            min_ys.append(bbox[1])
            max_xs.append(bbox[0] + bbox[2])
            max_ys.append(bbox[1] + bbox[3])

        cv2.imshow('Hue boxes', get_scaled_image(hue_boxes, 3))
        cv2.waitKey(0)

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
        if not showDigits:
            cv2.imshow('Digits marked', get_scaled_image(overall, scale))
            cv2.waitKey(5)
        else:
            prepped = [scale_and_grey_digit(digit) for digit in digits]# + [np.zeros((20, 136 - 20*len(digits)))]
            prepped_image = np.hstack(prepped).astype(np.uint8)
            boxed_digits = overall.copy()
            overall_width = boxed_digits.shape[1]
            prepped_width = prepped_image.shape[1]
            if overall_width > prepped_width:
                prepped_height = prepped_image.shape[0]
                prepped_image = np.hstack([prepped_image, np.zeros((prepped_height, 136 - prepped_height*len(digits)))]).astype(np.uint8)
            else:
                prepped_height = prepped_image.shape[0]
                boxed_digits = np.hstack([boxed_digits, np.zeros((68, prepped_height*len(digits) - 136, 3))]).astype(np.uint8)
            prepped_image = cv2.cvtColor(prepped_image, cv2.COLOR_GRAY2BGR)
            original_and_digits = np.concatenate((boxed_digits, prepped_image), axis=0)
            cv2.imshow('Digits marked', get_scaled_image(original_and_digits, scale))
            cv2.waitKey(5)

    return digits


def scale_and_grey_digit(digit):
    """
    Takes in a colour digits and resizes it, pads it and makes it black and white
    :param digit: An image of the digit
    :return: The digit as a 20x20 black and white image
    """
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

    # Pad the digit so it is 20x20
    color = [0, 0, 0]
    resized_black_white_image = cv2.copyMakeBorder(black_white_image, t, b, l, r, cv2.BORDER_CONSTANT, value=color)

    return resized_black_white_image


def get_new_captcha():
    """
    Get a new captcha from the internet and return its image
    :return: A new captcha
    """
    url = 'https://projecteuler.net/captcha/show_captcha.php'
    res = requests.get(url, stream=True)

    if res.status_code != 200:
        print('Failed to fetch image, status code:', str(res.status_code))
    else:
        with open('new_captcha.png', 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    new_captcha = cv2.imread('new_captcha.png')
    return new_captcha


def get_scaled_image(image, scale):
    """
    Takes an image and returns a scaled version (for displaying)
    :param image: The image to be enlarged
    :param scale: The scale by which to resize the image
    :return: The resized image
    """
    old_size = image.shape[:2]
    new_size = tuple([int(x * scale) for x in old_size])

    # Resize the image, passing in (width, height)
    scaled_image = cv2.resize(image, (new_size[1], new_size[0]))
    return scaled_image


def trained_knn_model():
    """
    Creates and trains a KNN model from the training data
    :return: The trained model
    """
    # Open and read the training data
    training_file = open('training.txt')
    training_data = training_file.readlines()
    training_file.close()
    training_data = [line.strip() for line in training_data]

    # Save the labels as the last character of each line and the rest is the array for the data
    training_labels = np.array([line[-1] for line in training_data]).astype(np.float32)
    training_digits = np.array([np.fromstring(line[1:-2], dtype=int, sep=',').astype(np.float32) for line in training_data])

    # Create a KNN model and train using the loaded training digits and labels
    knn_model = cv2.ml.KNearest_create()
    knn_model.train(training_digits, cv2.ml.ROW_SAMPLE, training_labels)
    return knn_model


def predict_knn(knn_model, digits):
    """
    Passes the digits through the model and returns the prediction
    :param knn_model: The trained model to pass the digits through
    :param digits: The prepped digits to pass through the model
    :return: A string representing the prediction for the digits
    """
    # Pass the digits through the KNN model
    ret, result, neighbours, dist = knn_model.findNearest(digits, k=15)
    # Format the result as a string
    result = ''.join([str(int(digit)) for digit in list(result.flatten())])
    return result


def trained_svm_model():
    """
    Creates and trains a SVM model from the training data
    :return: The trained model
    """
    # Open and read the training data
    training_file = open('training.txt')
    training_data = training_file.readlines()
    training_file.close()
    training_data = [line.strip() for line in training_data]

    # Save the labels as the last character of each line and the rest is the array for the data
    training_labels = np.array([line[-1] for line in training_data]).astype(np.float32)
    training_digits = np.array([np.fromstring(line[1:-2], dtype=int, sep=',').astype(np.float32) for line in training_data])

    # Create a SVM model and train using the loaded training digits and labels
    svm_model = cv2.ml.SVM()
    svm_model.trainAuto(training_digits, training_labels)
    return svm_model


def predict_svm(svm_model, digits):
    """
    Passes the digits through the model and returns the prediction
    :param svm_model: The trained model to pass the digits through
    :param digits: The prepped digits to pass through the model
    :return: A string representing the prediction for the digits
    """
    # Pass the digits through the SVM model
    ret, result, neighbours, dist = svm_model.predict(digits, k=15)
    # Format the result as a string
    result = ''.join([str(int(digit)) for digit in list(result.flatten())])
    return result


def show_training_digits():
    """
    A testing function to show the training digits which we have
    :return: None
    """
    training_file = open('training.txt')
    training_data = training_file.readlines()
    training_file.close()
    training_data = [line.strip() for line in training_data]

    # Save the labels as the last character of each line and the rest is the array for the data
    training_labels = np.array([line[-1] for line in training_data]).astype(np.float32)
    training_digits = np.array([np.fromstring(line[1:-2], dtype=int, sep=',').astype(np.float32) for line in training_data])
    print(training_digits.shape)
    display_digits = training_digits.reshape(training_digits.shape[0], 20, 20)
    for ind, digit in enumerate(display_digits):
        title = 'Digit: ' + str(int(training_labels[ind]))
        cv2.imshow(title, digit)
        cv2.waitKey(0)


