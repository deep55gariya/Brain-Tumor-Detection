import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import imutils
from PIL import Image
from plotly import graph_objs as go

# Streamlit Configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Load data function
def load_data(dir_path, img_size=(100, 100)):
    x, y, labels = [], [], {}
    for i, path in enumerate(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(os.path.join(dir_path, path)):
                if not file.startswith('.'):
                    img = cv2.imread(os.path.join(dir_path, path, file))
                    img = cv2.resize(img, img_size)
                    x.append(img)
                    y.append(i)
    return np.array(x), np.array(y), labels

# Plot confusion matrix function
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot(plt)

# Plot samples function
def plot_samples(x, y, labels_dict, n=50):
    for index in range(len(labels_dict)):
        imgs = x[np.argwhere(y == index)][:n]
        j, i = 10, int(n / 10)  # Ensure j is assigned before being used
        plt.figure(figsize=(15, 6))
        for c, img in enumerate(imgs, start=1):
            plt.subplot(i, j, c)
            plt.imshow(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))
            plt.xticks([]); plt.yticks([])
        plt.suptitle(f'Tumor: {labels_dict[index]}')
        st.pyplot(plt)

# Crop images function
def crop_imgs(set_name, add_pixels_value=0, target_size=(224, 224)):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            extLeft, extRight = tuple(c[c[:, :, 0].argmin()][0]), tuple(c[c[:, :, 0].argmax()][0])
            extTop, extBot = tuple(c[c[:, :, 1].argmin()][0]), tuple(c[c[:, :, 1].argmax()][0])
            new_img = img[extTop[1] - add_pixels_value:extBot[1] + add_pixels_value, 
                          extLeft[0] - add_pixels_value:extRight[0] + add_pixels_value].copy()
            new_img = cv2.resize(new_img, target_size, interpolation=cv2.INTER_CUBIC)
            if new_img.shape == (target_size[1], target_size[0], 3):
                set_new.append(new_img)
    return np.array(set_new)

# Save new images function
def save_new_images(x_set, y_set, folder_name):
    for i, (img, imclass) in enumerate(zip(x_set, y_set)):
        class_folder = 'YES' if imclass == 1 else 'NO'
        cv2.imwrite(os.path.join(folder_name, class_folder, f'{i}.jpg'), img)

# Preprocess images function
def preprocess_imgs(set_name, img_size):
    set_new = [preprocess_input(cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)) for img in set_name]
    return np.array(set_new)

# Streamlit UI
st.title("Brain Tumor Detection")

# Data Directory
IMG_PATH = "C:/Users/user/Desktop/Project/datasets"
TRAIN_DIR, VAL_DIR, TEST_DIR = 'train/', 'VAL/', 'TEST/'
IMG_SIZE = (224, 224)

# Load Data
if st.button("Load Data"):
    x_train, y_train, labels = load_data(os.path.join(IMG_PATH, TRAIN_DIR), IMG_SIZE)
    x_test, y_test, _ = load_data(os.path.join(IMG_PATH, TEST_DIR), IMG_SIZE)
    x_val, y_val, _ = load_data(os.path.join(IMG_PATH, VAL_DIR), IMG_SIZE)
    
    st.session_state.update({'x_train': x_train, 'y_train': y_train, 'labels': labels, 
                             'x_test': x_test, 'y_test': y_test, 'x_val': x_val, 'y_val': y_val})
    
    st.write(f'{len(x_train)} training images loaded.')
    st.write(f'{len(x_test)} testing images loaded.')
    st.write(f'{len(x_val)} validation images loaded.')

    # Plot Class Distribution
    y_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    trace0 = go.Bar(x=['Train Set'], y=[y_counts[0]], name='No', marker=dict(color='#33cc33'))
    trace1 = go.Bar(x=['Train Set'], y=[y_counts[1]], name='Yes', marker=dict(color='#ff3300'))
    layout = go.Layout(title='Count of classes in the train set', xaxis={'title': 'Set'}, yaxis={'title': 'Count'})
    fig = go.Figure(data=[trace0, trace1], layout=layout)
    st.plotly_chart(fig)

    # Show Sample Images
    plot_samples(x_train, y_train, labels, 30)

# Data Cropping
if st.button("Crop Data"):
    if 'x_train' in st.session_state:
        x_train_crop = crop_imgs(st.session_state.x_train)
        x_val_crop = crop_imgs(st.session_state.x_val)
        x_test_crop = crop_imgs(st.session_state.x_test)

        # Show Cropped Samples
        plot_samples(x_train_crop, st.session_state.y_train, st.session_state.labels, 30)

        # Save Cropped Images
        for folder in ['TRAIN_CROP', 'VAL_CROP', 'TEST_CROP']:
            os.makedirs(os.path.join(folder, 'YES'), exist_ok=True)
            os.makedirs(os.path.join(folder, 'NO'), exist_ok=True)

        save_new_images(x_train_crop, st.session_state.y_train, 'TRAIN_CROP/')
        save_new_images(x_val_crop, st.session_state.y_val, 'VAL_CROP/')
        save_new_images(x_test_crop, st.session_state.y_test, 'TEST_CROP/')
    else:
        st.warning("Please load the data first.")

# Data Augmentation
if st.button("Augment Data"):
    if 'x_train' in st.session_state:
        demo_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.05,
            height_shift_range=0.05,
            rescale=1./255,
            shear_range=0.05,
            brightness_range=[0.1, 1.5],
            horizontal_flip=True,
            vertical_flip=True
        )

        os.makedirs('preview', exist_ok=True)
        x = st.session_state.x_train[0].reshape((1,) + st.session_state.x_train[0].shape)
        
        for i, batch in enumerate(demo_datagen.flow(x, batch_size=1)):
            plt.figure(figsize=(10, 10))
            plt.imshow(batch[0].astype('uint8'))
            plt.axis('off')
            plt.savefig(f'preview/augmented_{i}.jpg')
            if i >= 20:
                break
        
        st.write("Data augmentation preview saved in 'preview' folder.")
    else:
        st.warning("Please load the data first.")

# Model Building
if st.button("Build and Train Model"):
    if 'x_train' in st.session_state:
        # Preprocess Images
        x_train = preprocess_imgs(st.session_state.x_train, IMG_SIZE)
        x_val = preprocess_imgs(st.session_state.x_val, IMG_SIZE)
        y_train, y_val = st.session_state.y_train, st.session_state.y_val

        # Model Definition
        base_model = MobileNet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, pooling='avg', weights='imagenet')
        model = Sequential([
            base_model,
            Flatten(),
            BatchNormalization(),
            Dropout(0.7),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Model Training
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            x_train, y_train, epochs=30, validation_data=(x_val, y_val), 
            callbacks=[early_stop]
        )

        st.session_state.model = model
        st.write("Model training complete.")

        # Plot Training History
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(history.history['loss'], label='Train Loss')
        axs[0].plot(history.history['val_loss'], label='Validation Loss')
        axs[0].set_title('Loss')
        axs[0].legend()

        axs[1].plot(history.history['accuracy'], label='Train Accuracy')
        axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axs[1].set_title('Accuracy')
        axs[1].legend()

        st.pyplot(fig)
    else:
        st.warning("Please load the data first.")

# Model Evaluation
if st.button("Evaluate Model"):
    if 'model' in st.session_state and 'x_test' in st.session_state:
        x_test = preprocess_imgs(st.session_state.x_test, IMG_SIZE)
        y_test = st.session_state.y_test
        y_pred_prob = st.session_state.model.predict(x_test)
        y_pred = np.where(y_pred_prob > 0.5, 1, 0)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, classes=['No Tumor', 'Tumor'], title='Confusion Matrix')
    else:
        st.warning("Please train the model first.")

# Predict Tumor
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    image = np.array(image.resize(IMG_SIZE))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Ensure the model is trained
    if 'model' in st.session_state:
        prediction = st.session_state.model.predict(image)
        st.write(f'Prediction: {"Tumor Detected" if prediction > 0.5 else "No Tumor Detected"}')
    else:
        st.warning("Please train the model first.")

# Clear Session
if st.button("Clear Session"):
    st.session_state.clear()
    st.write("Session cleared.")
