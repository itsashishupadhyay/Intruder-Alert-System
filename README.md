# Project Aim
We do leave our house complete un-attended for better half of our day, and In case of Theft, it would be good to have Time and Probably the Face capture of the Intruder.
This simple Script Uses Open CV and Pre-trained Tenserflow model to detect Object in the frame, If that is a "HUMAN" trigger a maling Script, to warn with attached image capture.
End goal would be to Run this script over a Raspberry Pi, but i dont have a camera module, if you do have please verify that for me
redirect your questions at ashishupadhyay93@gmail.com

I choose [MobileNet-SSD v3](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)

## Pre-Requirement
1. Install the required Package
`$ pip install -r requirements.txt`

2. Go to [MobileNet-SSD v3](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API) and download the Weights and Config files, They required if you want to update the model

3. ``Already done`` Make a Lable.txt containing all the labes, the above model has 80 lables

4. Setup an Email API, i used `sendgrid` if you decide to use the same, follow their documentation and update the parameters

## Working

### Image Detection
Paste any Picture you want to analize in the Folder and update the `image_file` in main
Uncomment `ImageDetection`

### Live Video Detection and Send Email If Person is Detected

On `Line 131` Comment out `Shootmail()` on first try and see the detection and percentage confidence, adjust the confidence `        ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.75)
`
### LiveVideoMulti (Still Experimental)

Wanted to divide Bigger frame into 3*3 smaller farmes and wanted to test each seperately

## Screenshots

Email:
![alt text](https://raw.githubusercontent.com/itsashishupadhyay/Intruder-Alert-System/main/img1.png "Email")


