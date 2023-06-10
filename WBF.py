import os
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

# Define the folder paths for images and labels
image_folder = "C:/Users/rezua/Desktop/balaced train/all class/images/"
label_folder = "C:/Users/rezua/Desktop/balaced train/all class/new labels/"

# Define the class mapping (class index to class ID)
class_mapping = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "11",
    12: "12",
    13: "13"
    # Add more class mappings as needed
}

# Iterate over each label file
for label_file in os.listdir(label_folder):
    label_path = os.path.join(label_folder, label_file)

    # Read the labels from the file
    with open(label_path, "r") as file:
        labels = file.readlines()

    # Parse the labels
    bounding_boxes = []
    class_ids = []
    for label in labels:
        class_id, x, y, width, height = map(float, label.strip().split())
        # Normalize the bounding box coordinates to [0, 1] range
        x_min = x / 512
        y_min = y / 512
        x_max = (x + width) / 512
        y_max = (y + height) / 512
        bounding_boxes.append([x_min, y_min, x_max, y_max])
        class_ids.append(int(class_id))

    # Apply WBF to consolidate bounding boxes
    boxes_list = [bounding_boxes]
    scores_list = [[1.0] * len(bounding_boxes)]  # Assign equal weights to all bounding boxes
    labels_list = [class_ids]
    wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.2)

    # Save the consolidated bounding boxes back to the label file
    with open(label_path, "w") as file:
        for box, score, label in zip(wbf_boxes, wbf_scores, wbf_labels):
            x_min, y_min, x_max, y_max = box
            # Convert the normalized coordinates back to the original image size if needed
            x_min *= 512
            y_min *= 512
            x_max *= 512
            y_max *= 512
            width = x_max - x_min
            height = y_max - y_min
            class_id = class_mapping[label]
            file.write(f"{class_id} {x_min} {y_min} {width} {height}\n")


