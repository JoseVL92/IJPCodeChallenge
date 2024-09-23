import os
import cv2
from ultralytics import YOLO


DEFAULT_YOLO_MODEL_PATH = os.path.abspath("./yolov8n.pt")
CAT_LABEL_ID = 15

# Load the YOLO model once to reuse across multiple calls
MODEL = YOLO(DEFAULT_YOLO_MODEL_PATH)


def validate_file_exists(file_path, description):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{description} '{file_path}' not found")


def detect_cats(img_path, model, thresh=0.3, show=False):
    """Detects cats in an image using a YOLO pretrained model"""
    validate_file_exists(img_path, "Image file")

    try:
        result = model.predict(img_path, conf=thresh, show=show)[0]
    except Exception as e:
        raise RuntimeError(f"YOLO model failed to make predictions: {e}")

    predictions = [
        {
            "x": bbox.xywhn[0][0].item(),
            "y": bbox.xywhn[0][1].item(),
            "w": bbox.xywhn[0][2].item(),
            "h": bbox.xywhn[0][3].item()
        }
        for bbox in result.boxes if bbox.cls.item() == CAT_LABEL_ID
    ]

    return predictions


def resize_and_place_dog(cat_box, dog_img, img):
    """Resizes and places the dog image on the cat bounding box in the original image"""
    img_height, img_width = img.shape[:2]
    box_center_x, box_center_y = int(cat_box['x'] * img_width), int(cat_box['y'] * img_height)
    box_width, box_height = int(cat_box['w'] * img_width), int(cat_box['h'] * img_height)

    # Calculate bounding box coordinates
    x = max(0, box_center_x - box_width // 2)
    y = max(0, box_center_y - box_height // 2)
    x_end = min(x + box_width, img_width)
    y_end = min(y + box_height, img_height)
    box_width, box_height = x_end - x, y_end - y

    resized_dog_img = cv2.resize(dog_img, (box_width, box_height))

    # Handle dog image with alpha channel (transparency)
    if resized_dog_img.shape[2] == 4:
        alpha_dog = resized_dog_img[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_dog
        for c in range(0, 3):  # Loop over RGB channels
            img[y:y + box_height, x:x + box_width, c] = (
                alpha_dog * resized_dog_img[:, :, c] +
                alpha_background * img[y:y + box_height, x:x + box_width, c]
            )
    else:
        img[y:y + box_height, x:x + box_width] = resized_dog_img


def replace_cats_with_dogs(img_path, dog_img_path, output_path, thresh=0.3, show=False, result_show_time=5000):
    """Detects cats in the image and replaces them with a dog image"""
    # Validate input files
    validate_file_exists(img_path, "Image file")
    validate_file_exists(dog_img_path, "Dog image")

    # Load images
    img = cv2.imread(img_path)
    dog_img = cv2.imread(dog_img_path, cv2.IMREAD_UNCHANGED)

    if dog_img is None:
        raise ValueError(f"Dog image '{dog_img_path}' is not a valid image")

    # Step 1: Detect cats in the image
    cat_predictions = detect_cats(img_path, MODEL, thresh=thresh, show=False)

    # Step 2: Replace detected cat regions with the dog image
    if cat_predictions:
        for cat in cat_predictions:
            resize_and_place_dog(cat, dog_img, img)

        # Step 3: Save the result
        cv2.imwrite(output_path, img)

        if show:
            cv2.imshow("Replaced Cats with Dogs", img)
            cv2.waitKey(result_show_time)  # Close window after result_show_time miliseconds
            cv2.destroyAllWindows()
    else:
        print("No cats detected")


if __name__ == "__main__":
    replace_cats_with_dogs('dog_cats.jpeg', 'dog_image.jpeg', 'output_image.jpg', thresh=0.3, show=True)
