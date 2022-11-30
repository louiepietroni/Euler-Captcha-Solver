import captchautils
import numpy as np
import cv2

# Open and read the training file
training_file = open('training.txt')
training_data = training_file.readlines()
training_file.close()
training_data = [line.strip() for line in training_data]

# Save the labels as the last character of each line and the rest is the array for the data
training_labels = np.array([line[-1] for line in training_data]).astype(np.float32)
training_digits = np.array([np.fromstring(line[1:-2], dtype=int, sep=',').astype(np.float32) for line in training_data])

# Create a KNN model and traing using the loaded training digits and labels
knn = cv2.ml.KNearest_create()
knn.train(training_digits, cv2.ml.ROW_SAMPLE, training_labels)

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
    ret,result,neighbours, dist = knn.findNearest(digits_for_knn, k=15)
    # Print out the prediction for the captcha
    print('Prediction: ', end='')
    for res in result:
        print(int(res[0]), end='')
    print()

    # If enter pressed, continue, otherwise stop
    cont = input("Continue?")
    if len(cont) != 0:
        break
