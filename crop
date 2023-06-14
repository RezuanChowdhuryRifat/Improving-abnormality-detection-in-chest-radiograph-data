import os
import cv2

# Define paths to input and output directories
image_dir = 'C:/Users/rezua/Desktop/balaced train/all class/images/'
label_dir = 'C:/Users/rezua/Desktop/balaced train/all class/new labels/'
output_dir = 'C:/Users/rezua/Desktop/balaced train/all class/cropped/'

# Define a dictionary mapping class IDs to class names
class_dict = {
    0: 'Aortic enlargement',
    1: 'Atelectasis',
    2: 'Calcification',
    3: 'Cardiomegaly',
    4: 'Consolidation',
    5: 'ILD',
    6: 'Infiltration',
    7: 'Lung Opacity',
    8: 'Nodule-Mass',
    9: 'Other lesion',
    10: 'Pleural effusion',
    11: 'Pleural thickening',
    12: 'Pneumothorax',
    13: 'Pulmonary fibrosis'
}

# Create output directories for each class
for class_id, class_name in class_dict.items():
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
# Loop over all image files in the input directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg'):
        # Read the image file
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # Read the corresponding label file
        label_path = os.path.join(label_dir, filename[:-4] + '.txt')
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Create a dictionary to hold bounding boxes for each class in the image
        class_boxes = {}

        # Loop over all bounding boxes in the label file
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert YOLO format to x,y coordinates of top-left corner and width,height
            x = int((x_center - width / 2) * image.shape[1])
            y = int((y_center - height / 2) * image.shape[0])
            w = int(width * image.shape[1])
            h = int(height * image.shape[0])

            # Add the bounding box to the dictionary for the appropriate class
            if class_id not in class_boxes:
                class_boxes[class_id] = []
            class_boxes[class_id].append((x, y, w, h))

        # Loop over all classes in the image
        for class_id, boxes in class_boxes.items():
            # Create a separate crop for each bounding box in the class
            for i, (x, y, w, h) in enumerate(boxes):
                # Crop the image using the bounding box
                crop = image[y:y + h, x:x + w]

                # Save the cropped image to the appropriate output directory
                class_name = class_dict[class_id]
                output_path = os.path.join(output_dir, class_name, filename[:-4] + f'_{class_id}_{i}.jpg')
                cv2.imwrite(output_path, crop)
