import captchautils
import numpy as np
import cv2


training_file = open('training.txt')
training_data = training_file.readlines()
training_file.close()
training_data = [line.strip() for line in training_data]

training_labels = np.array([line[-1] for line in training_data]).astype(np.float32)
training_digits = np.array([np.fromstring(line[1:-2], dtype=int, sep=',').astype(np.float32) for line in training_data])

# print(training_labels.shape)
# print(training_digits.shape)

knn = cv2.ml.KNearest_create()
knn.train(training_digits, cv2.ml.ROW_SAMPLE, training_labels)


while True:
    new_captcha = captchautils.get_new_captcha()
    # cv2.imshow('Test', new_captcha)
    # cv2.waitKey(5)

    digits = captchautils.get_digits_from_captcha(new_captcha, show=True)
    prepped_digits = np.array([captchautils.scale_and_grey_digit(digit) for digit in digits])

    # cv2.imshow('Test', prepped_digits[0])
    # cv2.waitKey(5)

    # digits_for_knn = [p_digit.reshape(-1, 400).astype(np.float32) for p_digit in prepped_digits]
    digits_for_knn = prepped_digits.reshape(-1, 400).astype(np.float32)
    # print(digits_for_knn.shape)

    ret,result,neighbours, dist = knn.findNearest(digits_for_knn, k=15)
    print('Prediction: ', end='')
    for res in result:
        print(int(res[0]), end='')
    print()

    cont = input("Continue?")
    if len(cont) != 0:
        break
