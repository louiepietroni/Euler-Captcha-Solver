import captchautils
import cv2
import numpy as np

file_path = 'Captchas/'
start_index = 146


def save_digits_to_file(images, values):
    with open('training.txt', 'a') as labelled_digit_file:
        for i in range(5):
            digit_image = np.array(images[i].flatten())
            digit_value = values[i]

            digit_image_list = digit_image.tolist()
            info = str(digit_image_list) + digit_value + '\n'
            labelled_digit_file.write(info)


def remove_digits_from_file():
    with open('training.txt', 'r+') as file:
        lines = file.readlines()
        # move file pointer to the beginning of a file
        file.seek(0)
        # truncate the file
        file.truncate()

        # start writing lines except the first line
        # lines[1:] from line 2 to last line
        file.writelines(lines[:-5])


def get_digits_in_file():
    with open('training.txt', 'r') as file:
        lines = file.readlines()
        print('Training digits:', len(lines))


# index = start_index
# while True:
#     # Load the next captcha
#     file_name = 'captcha%d.png' % index
#     captcha = cv2.imread(file_path + file_name)
#     if captcha is None:
#         print("Failed to load image")
#         exit(-1)
#
#     digits = captchautils.get_digits_from_captcha(captcha)
#     prepped_digits = [captchautils.scale_and_grey_digit(digit) for digit in digits]
#
#     cv2.imshow('Captcha', captchautils.get_scaled_image(captcha, 2))
#     cv2.waitKey(1)
#     answer = input("Value:")
#     if len(answer) != 5:
#         print('Next index to do:', index)
#         break
#     save_digits_to_file(prepped_digits, answer)
#     index += 1

while True:
    captcha = captchautils.get_new_captcha()

    digits = captchautils.get_digits_from_captcha(captcha, show=True)
    prepped_digits = [captchautils.scale_and_grey_digit(digit) for digit in digits]

    # cv2.imshow('Captcha', captchautils.get_scaled_image(captcha, 2))
    # cv2.waitKey(1)
    answer = input("Enter Captcha value:")
    if answer == 'skip':
        print('Skipping this captcha')
        continue
    elif answer == 'undo':
        remove_digits_from_file()
        print('Removing last Captcha digits from file')
        continue
    elif answer == 'len':
        get_digits_in_file()
        continue
    elif len(answer) == 5:
        save_digits_to_file(prepped_digits, answer)
    else:
        break



