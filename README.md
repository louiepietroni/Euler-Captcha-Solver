# Euler-Captcha-Solver
A program to solve Project Euler captchas using openCV for character detection and KNN for digit recognition

getcaptchas:
  Downloads captchas from project euler and saves them with increasing file names. change startindex for the first file name to use in this run

labelcaptchas:
  From start index, goes through the captchas which have been downloaded and shows them, you can then type the correct answer which'll be saved

solvecaptchas:
  Downloads a captcha and predicts its value
  
catpchautils:
  The code with all the captcha image recognition code
