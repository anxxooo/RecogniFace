# RecogniFace

RecogniFace is a facial recognition application built using Python, Streamlit, and FaceNet. It allows users to upload images for facial recognition or capture images using a webcam (if running locally).

## Features
- Facial recognition using FaceNet
- Image upload for recognition
- Streamlit-based user interface

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- Virtual environment (optional but recommended)

### Clone the Repository
```sh
git clone https://github.com/your-username/recogniface.git
cd recogniface
```

### Create and Activate Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Running the Application

### Local Deployment (With Webcam Support)
```sh
streamlit run RecogniFace.py
```

### Cloud Deployment (Without Webcam)
Most cloud environments do not support direct webcam access. Instead, use image uploads.
You can access the deployed version of RecogniFace here:  
[RecogniFace Live App](https://recogniface.streamlit.app/)


## Usage
- **Run Locally** to use the webcam feature.
- **Upload an Image** when using a cloud deployment.
- The app processes the image and identifies known faces using FaceNet.

## Troubleshooting
### Camera Access Issues
If running locally and getting `can't open camera by index`, ensure:
- Your webcam is not being used by another application.
- You have the correct permissions to access the webcam.

For cloud deployment, modify the app to accept image uploads instead of using a webcam.

### TensorFlow Errors
If you encounter CUDA or TensorFlow errors:
- Ensure TensorFlow is installed correctly.
- If using a CPU-only machine, install the CPU version of TensorFlow:
  ```sh
  pip install tensorflow-cpu
  ```
  
## Acknowledgments
- FaceNet for facial recognition
- Streamlit for UI
- OpenCV for image processing

