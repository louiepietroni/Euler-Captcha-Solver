import captchautils
import numpy as np
import cv2

knn_model = captchautils.trained_knn_model()
# svm_model = captchautils.trained_svm_model()

# Loop to get new captchas and solve them
while True:
    # Get a new captcha
    new_captcha = captchautils.get_new_captcha()

    # Get and prep the individual digits of the captcha and show the captcha too
    digits = captchautils.get_digits_from_captcha(new_captcha, show=True)
    prepped_digits = np.array([captchautils.scale_and_grey_digit(digit) for digit in digits])
    size = prepped_digits[0].shape[0]

    # Prepare the digits for the captcha to be fed through the model
    digits_for_prediction = prepped_digits.reshape(-1, size ** 2).astype(np.float32)

    # Pass the digits through the KNN model
    prediction = captchautils.predict_knn(knn_model, digits_for_prediction)
    # Print out the prediction for the captcha
    print('KNN prediction:', prediction)

    # # Pass the digits through the SVM model
    # prediction = captchautils.predict_svm(svm_model, digits_for_prediction)
    # # Print out the prediction for the captcha
    # print('SVM prediction:', prediction)

    # If enter pressed, continue, otherwise stop
    cont = input("Continue?")
    if len(cont) != 0:
        break
