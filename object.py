import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
import time

def Shootmail():
    message = Mail(
        from_email='ashishupadhyay93@gmail.com',
        to_emails='ashishupadhyay93@gmail.com',
        subject='Intruder Alert',
        html_content='<strong>There could be some one in your house </strong>')
    with open('attachment.jpg', 'rb') as f:
        data = f.read()
        f.close()
    encoded_file = base64.b64encode(data).decode()

    attachedFile = Attachment(
        FileContent(encoded_file),
        FileName('attachment.jpg'),
        FileType('image'),
        Disposition('attachment')
    )
    message.attachment = attachedFile

    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e)
    print("mail Sent")
    time.sleep(5)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def ImageDetection(frozen,config,labels,image):
    model = cv2.dnn_DetectionModel(frozen,config)
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    file_name = labels
    with open(file_name,'rt') as fpt:
        classLable = fpt.read().rstrip('\n').split('\n')
    img = cv2.imread(image)
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.6)
    FlatConfi = confidence.flatten()
    index = 0
    
    for  ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,(255,0,0),2)
        TextLable = " {0} {1:.2f}".format(classLable[ClassInd-1],FlatConfi[index])
        cv2.putText(img,TextLable,(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2)
        index+=1

    
    # print(confidence[1])
    # print(FlatConfi[1])
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.show()

        
def VideoDetection(frozen,config,labels):
    model = cv2.dnn_DetectionModel(frozen,config)
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    file_name = labels
    with open(file_name,'rt') as fpt:
        classLable = fpt.read().rstrip('\n').split('\n')

    camera = cv2.VideoCapture(0)
    font_scale = 1
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    while (True):
        
        (grabbed, img) = camera.read()
        img = cv2.flip(img, 1)

        ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.75)
        if (len(ClassIndex)!=0):
            for  ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
                FlatConfi = confidence.flatten()
                index = 0
                if(ClassInd<=80):
                    cv2.rectangle(img,boxes,(255,0,0),2)
                    TextLable = " {0} {1:.2f}".format(classLable[ClassInd-1],FlatConfi[index])
                    # print(classLable[ClassInd-1])
                    if(classLable[ClassInd-1]=='person'):
                        # print("Entered True")
                        cv2.imwrite("attachment.jpg", img)
                        Shootmail()
                    
                    cv2.putText(img,TextLable,(boxes[0]+10,boxes[1]+20),font,fontScale=font_scale,color=(0,0,255),thickness=1)
                    index+=1
        # cv2.imshow("video feed",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img = image_resize(img, height = 800)
        cv2.imshow("video Feed",img)
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
    # free up memory
    camera.release()
    cv2.destroyAllWindows()


def VideoDetectionMulti(frozen,config,labels):
    model = cv2.dnn_DetectionModel(frozen,config)
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    file_name = labels
    with open(file_name,'rt') as fpt:
        classLable = fpt.read().rstrip('\n').split('\n')

    camera = cv2.VideoCapture(0)
    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN
    cap = cv2.VideoCapture(0)
    n_rows = 3
    n_images_per_row = 3

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, ch = frame.shape

        roi_height = height // n_rows
        roi_width = width // n_images_per_row

        images = []

        for x in range(0, n_rows):
            for y in range(0,n_images_per_row):
                tmp_image=frame[x*roi_height:(x+1)*roi_height, y*roi_width:(y+1)*roi_width]
                images.append(tmp_image)

        # Display the resulting sub-frame
        for x in range(0, n_rows):
            for y in range(0, n_images_per_row):
                # cv2.imshow(str(x*n_images_per_row+y+1), images[x*n_images_per_row+y])
                # img =  cv2.flip(images[x*n_images_per_row+y],1)
                img =  images[x*n_images_per_row+y]

                
                ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.6)
                if (len(ClassIndex)!=0):
                    for  ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
                        if(ClassInd<=80):
                            cv2.rectangle(img,boxes,(255,0,0),2)
                            cv2.putText(img,classLable[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=2)
                images[x*n_images_per_row+y] = image_resize(images[x*n_images_per_row+y], height = 150)
                cv2.imshow(str(x*n_images_per_row+y+1), images[x*n_images_per_row+y])
                cv2.moveWindow(str(x*n_images_per_row+y+1), 100+(y*roi_width), 50+(x*roi_height))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # image_path = r'C:\Users\itsas\OneDrive\Desktop\Test_Code\Object_Detection\attachment.jpg'
    # directory = r'C:\Users\itsas\OneDrive\Desktop\Test_Code\Object_Detection'
    config = 'Config.pbtxt'
    frozen = 'frozen_inference_graph.pb'
    lable_file_name = 'Lables.txt'
    image_file='test-image3.jpg'
    # os.chdir(directory)
    # ImageDetection(frozen,config,lable_file_name,image_file)
    VideoDetection(frozen,config,lable_file_name)
    
    
