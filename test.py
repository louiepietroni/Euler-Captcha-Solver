import cv2
import numpy as np
import pytesseract

SZ = 20


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 0.01:
    # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img






# Minimum percentage of pixels of same hue to consider dominant colour
MIN_PIXEL_CNT_PCT = 0

route = 'Captchas/'
file_name = 'captcha11.png'

image = cv2.imread(route + file_name)
if image is None:
    print("Failed to load image")
    exit(-1)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# We're only interested in the hue
h,_,_ = cv2.split(image_hsv)
# Let's count the number of occurrences of each hue
# Eg bins[50] will be the number of hue=50s we have in the image
bins = np.bincount(h.flatten())
# And then find the dominant hues
# peaks = np.where(bins > (h.size * MIN_PIXEL_CNT_PCT))[0]

top = bins.argsort()[-6:][::-1][1:]
peaks = top

# START
im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
values, counts = np.unique(im_hsv.copy().reshape(-1, 3), axis=0, return_counts=True)
# values, counts = np.unique(a, return_counts=True)
ind = np.argpartition(-counts, kth=6)[1:6]
print(values[ind])  # prints the 5 most frequent elements
print(top)

peaks = values[ind]

images = []

overall = image.copy()

# Now let's find the shape matching each dominant hue
for i, peak in enumerate(peaks):
    # First we create a mask selecting all the pixels of this hue

    # mask = cv2.inRange(h, peak, peak)
    # low = np.array(peak)
    # high = np.array(peak)
    # mask = cv2.inRange(h, low, high)

    low = peak
    high = peak
    mask = cv2.inRange(im_hsv, low, high)



    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)

    # And use it to extract the corresponding part of the original colour image
    blob = cv2.bitwise_and(image, image, mask=mask)

    # _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for j, contour in enumerate(contours):

    min_xs = []
    min_ys = []
    max_xs = []
    max_ys = []

    for j, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        min_xs.append(bbox[0])
        min_ys.append(bbox[1])
        max_xs.append(bbox[0] + bbox[2])
        max_ys.append(bbox[1] + bbox[3])

    min_x = min(min_xs)
    max_x = max(max_xs)
    min_y = min(min_ys)
    max_y = max(max_ys)

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, 255, -1)


    # region = blob.copy()[min_y:max_y, min_x:max_x]
    region = image.copy()[min_y:max_y, min_x:max_x]
    region_masked = region

    # region_mask = contour_mask[min_y:max_y, min_x:max_x]
    # region_masked = cv2.bitwise_and(region, region, mask=region_mask)
    # file_name_section = "colourblobs-%d-hue_%03d-region_%d-section.png" % (i, peak, j)
    # file_name_section = "Outputs/%ssection%d.png" % (file_name, peak)
    # cv2.imwrite(file_name_section, region_masked)
    # print(" * wrote '%s'" % file_name_section)

    # Extract the pixels belonging to this contour
    # result = cv2.bitwise_and(blob, blob, mask=contour_mask)
    result = image.copy()

    # cv2.imshow('Pre', result)
    # cv2.waitKey(0)

    # And draw a bounding box
    top_left, bottom_right = (min_x, min_y), (max_x, max_y)
    cv2.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 0, 0), 1)
    cv2.rectangle(overall, (min_x, min_y), (max_x, max_y), (0, 0, 0), 1)
    # file_name_bbox = "Outputs/%sbbox%d.png" % (file_name, peak)
    # cv2.imwrite(file_name_bbox, result)


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # with_border = cv2.copyMakeBorder(region, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # cv2.imshow('Pre', thresh1)
    # cv2.waitKey(0)

    # HERE
    desired_size = 20

    im = thresh1.copy()
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    # cv2.imshow("image", new_im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    images.append((new_im, min_x))
    # END

    # text = pytesseract.image_to_string(thresh1)
    # print('text here', text)

file_name_bbox = "Outputs/solved%s" % file_name
cv2.imwrite(file_name_bbox, overall)

cv2.imshow('Digits identified', overall)
cv2.waitKey(0)


# START OF DIGIT SECTION
img = cv2.imread('digits.png')

# cv.imshow('this', img)
# cv.waitKey(0)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array: its size will be (50,100,20,20)
x = np.array(cells)
# Now we prepare the training data and test data
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train it on the training data, then test it with the test data with k=1
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
ret,result,neighbours,dist = knn.findNearest(test,k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
# print( accuracy )


first = x[30][20].reshape(-1,400).astype(np.float32)
# cv.imshow('this', first)
# cv.waitKey(0)
ret,result,neighbours,dist = knn.findNearest(first,k=5)
# print('Here', result)



images.sort(key=lambda tup: tup[1])  # sorts in place

print('Prediction: ', end='')
for image, l_start in images:
    first = image.reshape(-1,400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(first, k=5)
    print(int(result[0][0]), end='')
print()


