import cv2

def load_image(image_path):
    """Load an image from the specified path."""
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    """Preprocess the image by converting it to grayscale and applying Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    return blurred

def detect_fire(image):
    """Detect fire regions in the preprocessed image using contour analysis."""
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fire_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 and area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 3:
                fire_regions.append((x, y, x + w, y + h))
    
    return fire_regions

def draw_rectangles(image, regions):
    """Draw rectangles around the fire regions on the image."""
    for (x1, y1, x2, y2) in regions:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image

def display_image(image):
    """Display the image with the detected fire regions."""
    cv2.imshow("Fire Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to the image
image_path = "Fire_inside_an_abandoned_convent_in_Massueville,_Quebec,_Canada.jpg"

# Load and preprocess the image
image = load_image(image_path)
preprocessed_image = preprocess_image(image)

# Detect fire regions in the image
fire_regions = detect_fire(preprocessed_image)

# Draw rectangles around the fire regions on the image
image_with_regions = draw_rectangles(image.copy(), fire_regions)

# Display the image with the detected fire regions
display_image(image_with_regions)
