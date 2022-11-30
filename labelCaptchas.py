import captchautils
import cv2
import numpy as np

file_path = 'Captchas/'
start_index = 146


def save_digits_to_file(images, values):
    labelled_digit_file = open('training.txt', 'a')
    for i in range(5):
        digit_image = np.array(images[i].flatten())
        digit_value = values[i]

        digit_image_list = digit_image.tolist()
        info = str(digit_image_list) + digit_value + '\n'
        labelled_digit_file.write(info)

    labelled_digit_file.close()


index = start_index
while True:
    # Load the next captcha
    file_name = 'captcha%d.png' % index
    captcha = cv2.imread(file_path + file_name)
    if captcha is None:
        print("Failed to load image")
        exit(-1)

    digits = captchautils.get_digits_from_captcha(captcha)
    prepped_digits = [captchautils.scale_and_grey_digit(digit) for digit in digits]

    cv2.imshow('Captcha', captchautils.get_scaled_image(captcha, 2))
    cv2.waitKey(1)
    answer = input("Value:")
    if len(answer) != 5:
        print('Next index to do:', index)
        break
    save_digits_to_file(prepped_digits, answer)
    index += 1

