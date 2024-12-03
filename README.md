# AI-Powered Image Colorization Using Deep Learning

## Overview
This project demonstrates the application of deep learning for colorizing black-and-white images. Using a pre-trained model based on the Caffe framework, the application transforms grayscale images into realistic, colored ones. A Streamlit-powered interface ensures accessibility, allowing users to upload, process, and download their colorized images effortlessly.

## Features
- **User-Friendly Interface**: Upload images in formats like JPG, JPEG, or PNG.
- **Pre-trained Model**: Utilizes a robust colorization model trained on large datasets.
- **Real-Time Results**: Processes images and provides output promptly.
- **Downloadable Output**: Save the colorized images locally.

## Prerequisites
To run this project, ensure the following:

### System Requirements
- Python 3.7+
- Internet connection for installing dependencies

### Python Dependencies
Install the required libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Setup and Execution
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Add Model Files**:
   Place the following files in the `model/` directory:
   - `colorization_deploy_v2.prototxt`
   - `pts_in_hull.npy`
   - `colorization_release_v2.caffemodel`

3. **Run the Application**:
   Execute the Streamlit app:
   ```bash
   streamlit run colorization.py
   ```

4. **Access the Interface**:
   Open your browser and navigate to the URL displayed in the terminal, typically `http://localhost:8501/`.

## Directory Structure
```plaintext
project-folder/
├── colorization.py        # Main Streamlit application
├── requirements.txt       # List of dependencies
├── model/                 # Directory for model files
│   ├── colorization_deploy_v2.prototxt
│   ├── pts_in_hull.npy
│   └── colorization_release_v2.caffemodel
└── README.md              # Documentation
```

## Example Workflow
1. Upload a black-and-white image through the interface.
2. Click the "Colorize Image" button.
3. View the processed, colorized image.
4. Download the image using the provided link.

## Notes
- Ensure the `model/` directory contains all necessary files before running the app.
- Use a virtual environment to avoid conflicts with existing Python packages.

## References
- [Colorful Image Colorization - ECCV 2016](https://arxiv.org/abs/1603.08511)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

## License
This project is open-source and licensed under the MIT License.
