import cv2
import numpy as np
import tensorflow as tf

class HumanDetector:
    def __init__(self, model_path, input_size=(416, 416)):
        self.model = tf.saved_model.load(model_path)
        self.input_size = input_size

    def preprocess_image(self, image):
        image = cv2.resize(image, self.input_size)
        image = image / 255.0
        return np.expand_dims(image, axis=0)

    def detect(self, image):
        input_tensor = self.preprocess_image(image)
        detections = self.model(input_tensor)
        return detections

    def draw_detections(self, image, detections, confidence_threshold=0.5):
        for detection in detections:
            score = detection['score']
            if score >= confidence_threshold:
                bbox = detection['bbox']
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), 
                              (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(image, f"Confidence: {score:.2f}", 
                            (int(bbox[0]), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image

def main(image_path, model_path):
    detector = HumanDetector(model_path)
    image = cv2.imread(image_path)
    detections = detector.detect(image)
    output_image = detector.draw_detections(image, detections)
    cv2.imshow("Detections", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python detect.py <image_path> <model_path>")
    else:
        main(sys.argv[1], sys.argv[2])