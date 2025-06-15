
---

# Tennis Analysis Project

This project provides a comprehensive, deep-learning-based analysis of tennis matches from video footage. It detects and tracks players and the ball, reconstructs the court, and calculates advanced statistics like ball speed, player speed, distance covered, and shot counts.

![tennis_demo](https://github.com/user-attachments/assets/69101b00-c3d8-41f4-9da2-9478eae0936a)


## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Custom Models](#training-custom-models)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- **Object Detection**:
  - Detects players on the court.
  - Utilizes a custom-trained **YOLO model** for robust detection of fast-moving tennis balls, enhancing accuracy over generic models.

- **Player & Ball Tracking**:
  - Tracks the movement of each player and the ball across the entire video sequence.

- **Court Detection & Analysis**:
  - Employs a custom-trained **Convolutional Neural Network (CNN)** with PyTorch to estimate the court's key points.
  - Determines player positions relative to the court lines.
  - Measures real-world distances on the court.

- **Advanced Analytics & Statistics**:
  - **Ball Status**: Determines if the ball landed "in" or "out" of the court.
  - **Shot Counter**: Counts the number of times each player hits the ball.
  - **Ball Speed**: Estimates the speed of the ball after a shot.
  - **Player Metrics**: Calculates the total distance covered and the current speed of each player.

## Technology Stack
- **Programming Language**: Python
- **Deep Learning Frameworks**: PyTorch, Ultralytics YOLO
- **Computer Vision**: OpenCV
- **Numerical Computation**: NumPy
- **Core Libraries**:
  - `roboflow` (for dataset management/download)
  - `supervision` (for tracking and annotation utilities)

## Project Structure
The project is organized into modular components for clarity and scalability.

```
TENNIS_ANALYSIS-MAIN/
├── analysis/              # Scripts for statistical analysis (ball speed, distance, etc.)
├── constants/             # Project-wide constants (file paths, colors, configs)
├── court_line_detector/   # Contains the CNN model and logic for court keypoint estimation
├── input_videos/          # Directory for source videos to be analyzed
├── mini_court/            # Logic for drawing the 2D mini-court representation
├── models/                # Stores trained model weights (.pt, .onnx, etc.)
├── output_videos/         # Default directory for processed videos with annotations
├── runs/detect/           # Default output from YOLO inference runs
├── tracker_stubs/         # Stub files or interfaces for tracker integration
├── trackers/              # Implementation of tracking algorithms (e.g., ByteTrack)
├── training/              # Scripts and notebooks for training the custom models
├── utils/                 # Utility functions (video processing, drawing, file handling)
├── main.py                # Main script to run the full analysis pipeline
├── README.md              # This file
└── yolo_inference.py      # Standalone script for running YOLO object detection
```

## Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/TENNIS_ANALYSIS-MAIN.git
    cd TENNIS_ANALYSIS-MAIN
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    *(Note: You should create a `requirements.txt` file by running `pip freeze > requirements.txt`)*
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download Pre-trained Models:**
    Place the pre-trained model weights for the **ball detector (YOLO)** and the **court line detector (CNN)** into the `/models` directory.

5.  **Add Input Video:**
    Place the tennis video you want to analyze into the `/input_videos` directory.

## Usage

To run the full analysis pipeline on a video, execute the `main.py` script.

```sh
python main.py --input_video "input_videos/your_video_name.mp4"
```
The script will process the video and save the final output with all the analytics and annotations to the `/output_videos` directory.

## Training Custom Models

This project includes two custom-trained models. You can find the training scripts and logic in the `/training` directory.

### 1. YOLO Ball Detector
To achieve high accuracy in detecting the small, fast-moving tennis ball, a custom YOLO model was trained.

- **Dataset**: A labeled dataset of tennis ball images is required.
- **Training**: Use the scripts in `/training/yolo_training` to start a training run.
- **Output**: The best-performing model weight (`best.pt`) should be saved to the `/models` directory for inference.

### 2. Court Keypoint Estimator
A CNN was trained using PyTorch to predict the pixel locations of the court's key points from a given frame.

- **Dataset**: A dataset of tennis court images with annotated key points is needed.
- **Training**: The PyTorch training script is located in the `/training/court_detector_training` folder.
- **Output**: The final trained model should be saved to the `/models` directory.

## Future Improvements
- [ ] **Real-time Analysis**: Adapt the pipeline to work with a live video stream.
- [ ] **UI Dashboard**: Create a web-based dashboard (using Streamlit or Flask) to display stats interactively.
- [ ] **Pose Estimation**: Integrate human pose estimation to analyze player form (e.g., serve, forehand).
- [ ] **Multi-Camera Support**: Add functionality to fuse data from multiple camera angles for more accurate 3D tracking.

---
