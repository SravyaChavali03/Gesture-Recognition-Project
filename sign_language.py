# Import necessary libraries
import cv2
import numpy as np
import tensorflow as tf

# ========== 1️⃣ Load the Trained Model ==========
model = tf.keras.models.load_model("sign_language_model.h5")  # Load your trained model

# Debug: Check the model's output shape
print("Model output shape:", model.output_shape)

# Define the categories (26 letters of the ASL alphabet)
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# ========== 2️⃣ Capture or Provide an Image ==========
def capture_image():
    """
    Capture an image from the webcam.
    Press 'q' to take a picture.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture image.")
            break

        # Display the webcam feed
        cv2.imshow("Webcam", frame)

        # Press 'q' to capture the image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            img = frame
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    return img

# ========== 3️⃣ Preprocess the Image ==========
def preprocess_image(img):
    """
    Resize and normalize the image for the model.
    """
    # Resize the image to 64x64 pixels (or the size used during training)
    img = cv2.resize(img, (64, 64))

    # Normalize the pixel values to the range [0, 1]
    img = img / 255.0

    # Add a batch dimension (the model expects a batch of images)
    img = np.expand_dims(img, axis=0)
    return img

# ========== 4️⃣ Predict the Class ==========
def predict_image(img):
    """
    Predict the ASL letter in the image using the trained model.
    """
    # Preprocess the image
    img_processed = preprocess_image(img)

    # Make a prediction
    prediction = model.predict(img_processed)

    # Debug: Print the prediction shape and array
    print("Prediction shape:", prediction.shape)
    print("Prediction array:", prediction)

    # Ensure the prediction array is in the correct shape
    prediction = np.squeeze(prediction)  # Remove extra dimensions

    # Get the predicted class and confidence
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Debug: Print the predicted index and categories length
    print(f"Predicted Index: {predicted_index}, Categories Length: {len(categories)}")

    # Handle potential index errors
    try:
        predicted_class = categories[predicted_index]
    except IndexError:
        print(f"Error: Predicted index {predicted_index} is out of range for categories.")
        predicted_class = "Unknown"

    return predicted_class, confidence

# ========== 5️⃣ Main Function ==========
if __name__ == "__main__":
    # Ask the user to choose an option
    print("Choose an option:")
    print("1. Capture image from webcam")
    print("2. Load image from file")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        # Capture image from webcam
        print("Press 'q' to capture the image.")
        img = capture_image()
    elif choice == "2":
        # Load image from file
        image_path = input("Enter the path to the image: ")
        img = cv2.imread(image_path)
    else:
        print("Invalid choice. Exiting.")
        exit()

    # Check if the image was successfully loaded or captured
    if img is not None:
        # Predict the ASL letter in the image
        predicted_class, confidence = predict_image(img)
        print(f"Predicted class: {predicted_class} (Confidence: {confidence:.2f}%)")

        # Display the image with the prediction
        cv2.putText(img, f"{predicted_class} ({confidence:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to load or capture the image.")