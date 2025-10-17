from ultralytics import YOLO

# Load a model
model = YOLO("/home/juxi/ultralytics/ultralytics/yolo11n-obb.pt")

# Run batched inference on a list of images
results = model("/home/juxi/ultralytics/ultralytics/assets/car.jpg")  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="/home/juxi/ultralytics/ultralytics/output/car_output.jpg")  # save to disk