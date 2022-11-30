import captchautils
import numpy as np


def save_digits_to_file(images, values):
    """
    Takes in 5 digit images and their corresponding values and appends them to the training file
    :param images: List of 5 numpy arrays, each representing a digit
    :param values: 5 character string labelling the digits
    :return: None
    """
    with open('training.txt', 'a') as labelled_digit_file:
        for i in range(5):
            digit_image = np.array(images[i].flatten())
            digit_value = values[i]

            digit_image_list = digit_image.tolist()
            info = str(digit_image_list) + digit_value + '\n'
            labelled_digit_file.write(info)


def remove_digits_from_file():
    """
    Removes the 5 last lines from the training file (representing the most recent captcha)
    :return: None
    """
    with open('training.txt', 'r+') as file:
        lines = file.readlines()
        # move file pointer to the beginning of a file
        file.seek(0)
        # truncate the file
        file.truncate()

        file.writelines(lines[:-5])


def get_digits_in_file():
    """
    Print the number of training digits we have recorded so far
    :return: None
    """
    with open('training.txt', 'r') as file:
        lines = file.readlines()
        print('Training digits:', len(lines))


while True:
    # Get a new captcha
    captcha = captchautils.get_new_captcha()
    # Split it into its digits (which shows it also) and get the prepped digits too
    digits = captchautils.get_digits_from_captcha(captcha, show=True)
    prepped_digits = [captchautils.scale_and_grey_digit(digit) for digit in digits]

    # Get the user to label the captcha, (eg. 74395) although there are a few special commands too
    answer = input("Enter Captcha value:")
    if answer == 'skip':
        # Skip will ignore this captcha and move on (eg. digits recognised incorrectly)
        print('Skipping this captcha')
        continue
    elif answer == 'undo':
        # Undo will remove the previous captchas digits from the file and ignore this captcha
        # Eg you realise you entered the previous captcha incorrectly
        remove_digits_from_file()
        print('Removing last Captcha digits from file')
        continue
    elif answer == 'len':
        # Gets the number of digits currently in the training data
        get_digits_in_file()
        continue
    elif len(answer) == 5:
        # If the answer is 5 digits, take as a label and save
        save_digits_to_file(prepped_digits, answer)
    else:
        # Otherwise just end the training data collection
        break
