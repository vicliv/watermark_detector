import cv2
import numpy as np

def detect_watermark(image_path, template_path, threshold=0.8):
    """
    Detect a watermark by template matching within a 100x50 crop 
    at the bottom-right corner of the image.
    
    :param image_path: Path to the input image.
    :param template_path: Path to the watermark template image.
    :param threshold: Match threshold for deciding if the watermark is present.
    :return: Boolean indicating whether the watermark was detected.
    """

    # Read the main image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    # Read the template image
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Could not read template from {template_path}")

    # Get image dimensions
    h_img, w_img = img.shape[:2]

    # Define the region of interest (ROI) in the bottom-right corner.
    # This will crop a 100x50 region (width=100, height=50) from the bottom-right.
    crop_width = 100
    crop_height = 50
    x_start = w_img - crop_width
    y_start = h_img - crop_height
    
    # Handle edge cases: if the image is smaller than the crop size
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    # Crop the ROI
    roi = img[y_start:h_img, x_start:w_img]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # Check if the best match exceeds our threshold
    if max_val >= threshold:
        print(f"Watermark detected! Match confidence: {max_val:.2f}")
        return True
    else:
        print(f"No watermark detected. Highest match confidence: {max_val:.2f}")
        return False


if __name__ == "__main__":
    # Example usage:
    test_image_path = "generated_image.jpg"
    watermark_template_path = "grok.jpg"

    # This threshold might need tuning depending on your template and image conditions.
    watermark_found = detect_watermark(
        test_image_path,
        watermark_template_path,
        threshold=0.1
    )
    print("Watermark Found:", watermark_found)
