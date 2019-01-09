# import the necessary packages
import cv2
import numpy as np
from flask import Flask, request, Response, send_file
import requests
import json

# Initialize the Flask application
app = Flask(__name__)

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def get_opencv_major_version(lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib

    # return the major version number
    return int(lib.__version__.split(".")[0])


def is_cv2(or_better=False):
    # grab the OpenCV major version number
    major = get_opencv_major_version()

    # check to see if we are using *at least* OpenCV 2
    if or_better:
        return major >= 2

    # otherwise we want to check for *strictly* OpenCV 2
    return major == 2


 
class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    pts = np.array(pts, dtype = "float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def warp_speed(image):
    resized = resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
     
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    #thresh = cv2.threshold(gray, 60, 50, cv2.THRESH_BINARY)[1]

    canny = cv2.Canny(gray,0, 100);
     
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if is_cv2() else cnts[1]
    sd = ShapeDetector()

    # loop over the contours
    largestVal = 0
    largestIndex = 0
    counter =0
    for c in cnts:
        counter+=1
        print(len(cnts))
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        #cX = int((M["m10"] / M["m00"]) * ratio)
        #cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
    
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        #       0.5, (255, 255, 255), 2)
        area = cv2.contourArea(c);
    
        if (area > largestVal):
            largestIndex = counter
            
        print("Sucess")

        # show the output image
        
    epsilon = 0.1*cv2.arcLength(cnts[largestIndex-1],True)
    approx = cv2.approxPolyDP(cnts[largestIndex-1],epsilon,True)
    print ("approx", approx)

    croppedImg = image.copy()
    #print([point[0] for point in approx])
    #exit()

    test = [point[0] for point in approx]
    test = [(x, y) for (x,y) in test]
    order = sorted(test, key=lambda point: point[0] + point[1] * 300)
    print(test)

    #order[0], order[1] = order[1], order[0]
    print(order)

    #x + y * 1000

    #exit()

    warped = four_point_transform(image, test)

    return warped, canny

    ##maxx = 200
    ##
    ##rows,cols,ch = croppedImg.shape
    ##pts1 = np.float32(order)
    ##pts2 = np.float32([[0,0],[maxx,0],[0,maxx],[maxx,maxx]])
    ## 
    ##M = cv2.getPerspectiveTransform(pts1,pts2)
    ##
    ##croppedImg = cv2.warpPerspective(image,M,(maxx,maxx))

    #cv2.imshow("Cropped", warped)
    #cv2.imshow("Image", image)
    #cv2.imshow("Canny", canny)
    #cv2.waitKey(0)

# route http posts to this method
@app.route('/', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    warped = warp_speed(img)

    retval, buffer = cv2.imencode('.png', warped)
    png_as_text = base64.b64encode(buffer)
    response = make_response(png_as_text)
    response.headers['Content-Type'] = 'image/png'
    return response

    # do some fancy processing here....

    # build a response dict to send back to client
    #response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
    #            }
    # encode response using jsonpickle
    #response_pickled = jsonpickle.encode(response)

    #return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
#app.run(host="127.0.0.1", port=5000)
image = cv2.imread("yo.jpg")
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
warped, canny = warp_speed(image)
cv2.imshow("Image", image)
cv2.imshow("Canny", canny)
cv2.imshow("Cropped", warped)
cv2.waitKey(0)
