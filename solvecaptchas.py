import captchautils
import numpy as np
import cv2

trained_model = captchautils.trained_knn_model()

# Loop to get new captchas and solve them
while True:
    # Get a new captcha
    new_captcha = captchautils.get_new_captcha()

    # Get and prep the individual digits of the captcha and show the captcha too
    digits = captchautils.get_digits_from_captcha(new_captcha, show=True)
    prepped_digits = np.array([captchautils.scale_and_grey_digit(digit) for digit in digits])

    # Prepare the digits for the captcha to be fed through the KNN model
    digits_for_knn = prepped_digits.reshape(-1, 400).astype(np.float32)

    # Pass the digits through the KNN model
    prediction = captchautils.predict_digits(trained_model, digits_for_knn)
    # Print out the prediction for the captcha
    print('Prediction:', prediction)

    # If enter pressed, continue, otherwise stop
    cont = input("Continue?")
    if len(cont) != 0:
        break
