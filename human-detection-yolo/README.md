# Human Detection YOLO

This project implements a human detection system using the YOLO (You Only Look Once) model. The goal is to train a YOLO model to detect humans in images and perform real-time detection.

## Project Structure

```
human-detection-yolo
├── data
│   ├── images          # Directory containing images for training
│   └── annotations     # Directory containing YOLO format annotation files
├── src
│   ├── train.py        # Script to train the YOLO model
│   ├── detect.py       # Script to perform human detection
│   └── utils.py        # Utility functions for data processing and visualization
├── models
│   └── yolo_config.yaml # Configuration settings for the YOLO model
├── requirements.txt     # Python dependencies for the project
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd human-detection-yolo
   ```

2. **Install Dependencies**
   Create a virtual environment and install the required packages:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   - Place your training images in the `data/images` directory.
   - Place the corresponding YOLO format annotation files in the `data/annotations` directory.

## Training the Model

To train the YOLO model, run the following command:
```
python src/train.py
```

## Running Detection

After training, you can perform human detection using the trained model:
```
python src/detect.py --image <path-to-image>
```

## Configuration

Modify the `models/yolo_config.yaml` file to adjust model parameters such as input size and number of classes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.