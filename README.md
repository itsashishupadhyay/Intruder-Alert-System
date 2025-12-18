# 🚨 Guardian Eye: Smart AI-Powered Intruder Detection System

> **Turn your webcam into an intelligent security guard that never sleeps!** A Python-powered real-time object detection system that watches your home and sends instant email alerts when unwanted guests arrive.

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.1.1-green.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNet--SSD-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is Guardian Eye?

**Guardian Eye** is a smart, lightweight Python surveillance system that uses **AI-powered computer vision** to detect intruders in real-time. Built with **OpenCV** and **TensorFlow's MobileNet-SSD v3**, it transforms any webcam into an intelligent security camera that recognizes people and sends instant email alerts with photo evidence.

Perfect for:
- 🏠 **Home Security** - Monitor your home while you're away
- 🔍 **Object Detection** - Identify 80+ different objects in real-time
- 🤖 **AI Learning Projects** - Learn computer vision and deep learning
- 🍓 **Raspberry Pi Projects** - Deploy on embedded systems for IoT security
- 📧 **Smart Notifications** - Get instant alerts when motion is detected

---

## ✨ Features

- ⚡ **Real-Time Detection** - Instant person/object recognition using live webcam feed
- 🧠 **AI-Powered** - TensorFlow MobileNet-SSD v3 pre-trained model (80+ object classes)
- 📸 **Smart Alerts** - Automatic email notifications with attached snapshots
- 🎯 **Customizable Confidence** - Adjust detection sensitivity (default: 75%)
- 🖼️ **Dual Mode** - Works with both static images and live video streams
- 🔧 **Lightweight** - Optimized for low-resource environments like Raspberry Pi
- 🌐 **Open Source** - Free to use, modify, and distribute

---

## 🛠️ Tech Stack & Libraries

| Technology | Purpose | Version |
|------------|---------|---------|
| **OpenCV** | Real-time computer vision and image processing | 4.1.1 |
| **TensorFlow** | Deep learning model (MobileNet-SSD v3) | Via OpenCV DNN |
| **NumPy** | Numerical computing and array operations | 1.16.2 |
| **Matplotlib** | Image visualization and plotting | 3.0.3 |
| **SendGrid** | Email API for instant alert notifications | 6.6.0 |

### 🧠 AI Model: MobileNet-SSD v3

- **Architecture**: Single Shot Detector (SSD) with MobileNet backbone
- **Input Size**: 320x320 pixels
- **Classes**: 80 objects from COCO dataset (person, car, dog, etc.)
- **Performance**: Optimized for real-time detection on CPU
- **Accuracy**: Configurable confidence threshold (60-90%)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.x
- Webcam (USB or built-in)
- SendGrid API Key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/itsashishupadhyay/Intruder-Alert-System.git
   cd Intruder-Alert-System
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download AI Model Files**

   Download the **MobileNet-SSD v3** model files from [OpenCV TensorFlow Models](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API):
   - `frozen_inference_graph.pb` (model weights)
   - `Config.pbtxt` (model configuration)

   Place these files in the project root directory.

4. **Configure Email Alerts**

   Sign up for a free [SendGrid account](https://sendgrid.com/) and get your API key:
   ```bash
   export SENDGRID_API_KEY='your_sendgrid_api_key_here'
   ```

   Update email addresses in `object.py` (lines 11-13):
   ```python
   from_email='your_email@gmail.com',
   to_emails='recipient_email@gmail.com',
   ```

---

## 📖 How to Use

### Mode 1: Live Video Surveillance (Intruder Detection)

Monitor your space in real-time and receive email alerts when a person is detected:

```bash
python object.py
```

**How it works:**
1. Opens your webcam feed
2. Analyzes each frame using AI object detection
3. Detects people with 75% confidence threshold
4. Captures snapshot when person detected
5. Sends instant email alert with photo
6. Press `q` to quit

**Customize detection sensitivity** (line 120):
```python
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.75)
# Lower = more sensitive (0.6-0.9 recommended)
```

### Mode 2: Static Image Analysis

Analyze any image for objects:

1. Place your image in the project folder
2. Edit `object.py` (line 215):
   ```python
   image_file = 'your_image.jpg'
   ```
3. Uncomment line 217:
   ```python
   ImageDetection(frozen, config, lable_file_name, image_file)
   ```
4. Comment line 218:
   ```python
   # VideoDetection(frozen, config, lable_file_name)
   ```
5. Run: `python object.py`

### Mode 3: Multi-Grid Detection (Experimental)

Divide the video feed into a 3x3 grid for area-specific monitoring:

1. Replace line 218 with:
   ```python
   VideoDetectionMulti(frozen, config, lable_file_name)
   ```
2. Run: `python object.py`

---

## 📁 Project Structure

```
Intruder-Alert-System/
├── object.py                      # Main application code
├── requirements.txt               # Python dependencies
├── Lables.txt                     # 80 object class labels (COCO dataset)
├── frozen_inference_graph.pb      # TensorFlow model weights (download)
├── Config.pbtxt                   # Model configuration (download)
├── attachment.jpg                 # Captured intruder photo (generated)
└── README.md                      # This file
```

---

## 🎓 How It Works

### Detection Pipeline

```
Webcam Feed → Frame Capture → Preprocessing → AI Model → Object Detection → Person Found?
                                                               ↓ Yes
                                          Save Image → Send Email Alert → Continue Monitoring
```

### Code Breakdown

1. **Video Capture** (`cv2.VideoCapture`)
   - Captures frames from webcam at real-time
   - Flips image horizontally for mirror effect

2. **Model Preprocessing**
   - Resizes frames to 320x320 pixels
   - Normalizes pixel values (0-255 → -1 to 1)
   - Adjusts RGB color channels

3. **Object Detection** (`model.detect`)
   - Runs MobileNet-SSD inference
   - Returns: class ID, confidence score, bounding box
   - Filters results by confidence threshold

4. **Person Detection Logic**
   - Checks if detected object is "person"
   - Saves frame as `attachment.jpg`
   - Triggers email notification

5. **Email Alert** (`Shootmail`)
   - Encodes image in Base64
   - Sends via SendGrid API with attachment
   - Includes timestamp and alert message

---

## 🍓 Raspberry Pi Deployment

This project is designed to run on **Raspberry Pi** for edge-based home security!

### Recommended Setup:
- **Hardware**: Raspberry Pi 3B+ or 4
- **Camera**: Raspberry Pi Camera Module or USB Webcam
- **OS**: Raspberry Pi OS (Bullseye)

### Pi-Specific Setup:
```bash
# Enable camera
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Install OpenCV with Pi optimizations
pip install opencv-contrib-python

# Run on boot (add to crontab)
@reboot /usr/bin/python3 /home/pi/Intruder-Alert-System/object.py
```

---

## ⚙️ Configuration & Customization

### Adjust Detection Confidence
```python
# Line 120, 84, or 191 depending on mode
confThreshold=0.75  # Values: 0.0-1.0 (higher = stricter)
```

### Detect Multiple Objects
Modify line 129 to detect other objects:
```python
if classLable[ClassInd-1] in ['person', 'car', 'dog']:
    # Trigger alert for person, car, OR dog
```

### Email Cooldown
Change delay between alerts (line 38):
```python
time.sleep(5)  # Seconds between emails
```

### Change Detection Box Color
Modify line 126:
```python
cv2.rectangle(img, boxes, (255,0,0), 2)  # RGB: (R, G, B)
```

---

## 📧 Email Alert Example

When an intruder is detected, you'll receive:

**Subject:** Intruder Alert
**Body:** There could be someone in your house
**Attachment:** `attachment.jpg` (snapshot with bounding box)

![Email Alert Example](https://raw.githubusercontent.com/itsashishupadhyay/Intruder-Alert-System/main/img1.png)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Camera not found** | Check camera permissions, try `cv2.VideoCapture(1)` |
| **Low FPS** | Reduce resolution or increase confidence threshold |
| **Too many false alerts** | Increase `confThreshold` to 0.8 or 0.9 |
| **Email not sending** | Verify `SENDGRID_API_KEY` environment variable |
| **Model files missing** | Download from [OpenCV Wiki](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API) |

---

## 🎯 Use Cases & Applications

- **Home Security Systems** - Monitor entryways while you're away
- **Pet Detection** - Track when your pets enter restricted areas
- **Package Delivery Alerts** - Know when packages arrive
- **Elder Care Monitoring** - Check on elderly family members
- **Wildlife Observation** - Detect animals in your backyard
- **Smart Doorbell** - DIY Ring/Nest alternative
- **Office Security** - After-hours monitoring
- **IoT Learning Projects** - Educational AI/ML projects

---

## 🚀 Future Enhancements

- [ ] Multi-camera support
- [ ] Cloud storage integration (AWS S3, Google Drive)
- [ ] Face recognition for known/unknown persons
- [ ] Motion-triggered recording
- [ ] Mobile app notifications (push notifications)
- [ ] Night vision support (IR camera)
- [ ] Web dashboard for remote viewing
- [ ] Telegram/Discord bot integration

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas needing help:**
- Raspberry Pi Camera Module testing
- Face recognition integration
- Performance optimization
- Documentation improvements

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Ashish Upadhyay**

- Email: ashishupadhyay93@gmail.com
- GitHub: [@itsashishupadhyay](https://github.com/itsashishupadhyay)

---

## 🌟 Acknowledgments

- **OpenCV Team** - Computer vision library
- **TensorFlow Team** - MobileNet-SSD model
- **SendGrid** - Email API service
- **COCO Dataset** - Object detection training data

---

## 📊 Keywords for Discovery

`python object detection`, `opencv intruder detection`, `tensorflow person detection`, `raspberry pi security camera`, `ai home surveillance`, `real-time object detection python`, `smart home security system`, `mobilenet-ssd python`, `computer vision projects`, `iot security camera`, `python webcam monitoring`, `email alert system python`, `deep learning security`, `python ai projects`, `home automation python`

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐️ on GitHub!

**Questions?** Open an issue or email ashishupadhyay93@gmail.com

---

<div align="center">

**Built with ❤️ using Python, OpenCV & TensorFlow**

*Protecting homes one frame at a time* 🏠🔒

</div>
