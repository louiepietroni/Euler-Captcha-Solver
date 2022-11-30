import captchautils
import numpy as np


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


while True:
    captcha = captchautils.get_new_captcha()

    digits = captchautils.get_digits_from_captcha(captcha, show=True)
    prepped_digits = [captchautils.scale_and_grey_digit(digit) for digit in digits]

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
