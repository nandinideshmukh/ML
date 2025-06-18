## 🚀 Project Highlights

- **Fraud Detection System**  -> **task2.ipynb** <br>
  Developed an intelligent fraud detection pipeline leveraging **Logistic Regression**, **Random Forest**, and a **Convolutional Neural Network (CNN)** to identify anomalies in financial transactions with high precision.

- **Customer Churn Prediction**  -> **task3.ipynb**<br>
  Built a predictive model using ensemble-based **boosting algorithms** including **Gradient Boosting**, **AdaBoost**, **XGBoost**, and **Random Forest** to proactively detect potential customer churn .

- **Spam Message Classifier**  -> **spam_ham.ipynb**<br>
  Engineered a robust spam detection model utilizing **TF-IDF vectorization**, **Naive Bayes**, and **Support Vector Machine (SVM)** to filter unsolicited communications with high accuracy.

## 🧠 CNN-Based Image Classification

For models involving Convolutional Neural Networks (CNNs), grayscale image data was preprocessed using the following pipeline:

1. The image is loaded in **grayscale** mode using OpenCV.
2. It is resized to a consistent shape of **124×124** pixels.
3. Pixel values are **normalized** to the range `[0, 1]` by dividing by 255.
4. The image is expanded with a batch dimension to make it compatible with the CNN input layer.
5. Early stopping used as a call back to reduce computation cost and stop the training of model when accuracy does not seem to improve.
   
```python
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")

    image = cv2.resize(image, (124, 124))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return image
