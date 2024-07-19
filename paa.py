import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# Global variables
SAVE = False
SEED = 111

# Setting seed for consistent results
tf.keras.utils.set_random_seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Data Visualization updates
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams.update({'font.size': 14})

# Data Classifications
CLASS_TYPES = ['pituitary', 'notumor', 'meningioma', 'glioma']
N_TYPES = len(CLASS_TYPES)

# Function for importing data
def get_data_labels(directory, do_shuffle=True, random_state=0):
    data_path = []
    data_labels = []
    
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for image in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image)
            data_path.append(image_path)
            data_labels.append(label)
            
    if do_shuffle:
        data_path, data_labels = shuffle(data_path, data_labels, random_state=random_state)
            
    return data_path, data_labels

# Streamlit user inputs
USER_PATH = st.text_input("Enter the path to the brain tumor MRI dataset:", "C:/Users/user/Desktop/brain tumor/dataset")
train_dir = os.path.join(USER_PATH, 'Training')
test_dir = os.path.join(USER_PATH, 'Testing')

# Getting data using above function
train_paths, train_labels = get_data_labels(train_dir)
test_paths, test_labels = get_data_labels(test_dir)

st.write('## Training Data')
st.write(f'Number of Paths: {len(train_paths)}')
st.write(f'Number of Labels: {len(train_labels)}')

st.write('## Testing Data')
st.write(f'Number of Paths: {len(test_paths)}')
st.write(f'Number of Labels: {len(test_labels)}')

# Plotting the data
class_counts_train = [len([x for x in train_labels if x == label]) for label in CLASS_TYPES]
class_counts_test = [len([x for x in test_labels if x == label]) for label in CLASS_TYPES]

fig, ax = plt.subplots(ncols=3, figsize=(20, 14))

# Training data pie chart
ax[0].set_title('Training Data')
ax[0].pie(class_counts_train, labels=[label.title() for label in CLASS_TYPES], colors=['#FAC500', '#0BFA00', '#0066FA', '#FA0000'],
          autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum(class_counts_train) / 100), explode=tuple(0.01 for i in range(N_TYPES)),
          textprops={'fontsize': 20})

# Train-test split pie chart
ax[1].set_title('Train Test Split')
ax[1].pie([len(train_labels), len(test_labels)], labels=['Train', 'Test'], colors=['darkcyan', 'orange'],
          autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum([len(train_labels), len(test_labels)]) / 100), explode=(0.1, 0),
          startangle=85, textprops={'fontsize': 20})

# Testing data pie chart
ax[2].set_title('Testing Data')
ax[2].pie(class_counts_test, labels=[label.title() for label in CLASS_TYPES], colors=['#FAC500', '#0BFA00', '#0066FA', '#FA0000'],
          autopct=lambda p: '{:.2f}%\n{:,.0f}'.format(p, p * sum(class_counts_test) / 100), explode=tuple(0.01 for i in range(N_TYPES)),
          textprops={'fontsize': 20})

st.pyplot(fig)

# Display a sample image
im = load_img(train_paths[3], target_size=(150, 150))
im = img_to_array(im)
im = np.expand_dims(im, axis=0)
im /= np.max(im)
im = array_to_img(im[0])
st.image(im, caption='Sample Image')

# Function to display a list of images based on the given index
def show_images(paths, label_paths, index_list=range(10), im_size=250, figsize=(12, 8), save=False):
    num_images = len(index_list)
    num_rows = (num_images + 3) // 4
    fig, ax = plt.subplots(nrows=num_rows, ncols=4, figsize=figsize)
    ax = ax.flatten()
    for i, index in enumerate(index_list):
        if i >= num_images:
            break
        image = load_img(paths[index], target_size=(im_size, im_size))
        ax[i].imshow(image)
        ax[i].set_title(f'{index}: {label_paths[index]}')
        ax[i].axis('off')
    plt.tight_layout()
    if save:
        plt.savefig('show_image.pdf')
    else:
        st.pyplot(fig)

# Show sample images
show_images(train_paths, train_labels, im_size=350, figsize=(13, 10), index_list=[0, 94, 235, 17, 61, 324, 55, 45, 374, 65, 391, 488])

# Image size
image_size = (150, 150)

# Training batch size
batch_size = 32

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   brightness_range=(0.85, 1.15),
                                   width_shift_range=0.002,
                                   height_shift_range=0.002,
                                   shear_range=12.5,
                                   zoom_range=0,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   fill_mode="nearest")

# Applying the generator to training data with constant seed
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode="categorical",
                                                    seed=SEED)

# No augmentation of the test data, just rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Applying the generator to testing data with constant seed
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode="categorical",
                                                  shuffle=False,
                                                  seed=SEED)

# Accessing class indices for training data generator
class_indices_train = train_generator.class_indices
class_indices_train_list = list(train_generator.class_indices.keys())

st.write("## Categorical types for the training data:")
st.write(class_indices_train)

def show_ImageDataGenerator(train_generator, num_samples=5, figsize=(12, 12), save=False):
    """
    Function to visualize how the ImageDataGenerator augments the data
    """
    
    # Generate augmented samples
    augmented_samples = next(iter(train_generator))

    # Extract images from the batch
    images = augmented_samples[0][:num_samples]

    # Display the augmented images
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis('off')
        
    plt.tight_layout()
        
    if save:
        plt.savefig('show_ImageDataGenerator.pdf')
        
    st.pyplot(fig)

show_ImageDataGenerator(train_generator, num_samples=5, figsize=(12.5, 8), save=SAVE)

# Image shape: height, width, RBG
image_shape = (image_size[0], image_size[1], 3)

# Training epochs
epochs = 40

# Steps per epoch
steps_per_epoch = train_generator.samples // batch_size

# Validation steps
validation_steps = test_generator.samples // batch_size

st.write(f'Image shape: {image_shape}')
st.write(f'Epochs: {epochs}')
st.write(f'Batch size: {batch_size}')
st.write(f'Steps Per Epoch: {steps_per_epoch}')
st.write(f'Validation steps: {validation_steps}')

# Define the model architecture
model = models.Sequential([
    
    # Convolutional layer 1
    Conv2D(32, (4, 4), activation="relu", input_shape=image_shape),
    MaxPooling2D(pool_size=(3, 3)),

    # Convolutional layer 2
    Conv2D(64, (4, 4), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),

    # Convolutional layer 3
    Conv2D(128, (4, 4), activation="relu"),
    MaxPooling2D(pool_size=(3, 3)),

    # Convolutional layer 4
    Conv2D(128, (4, 4), activation="relu"),
    Flatten(),

    # Full connect layers
    Dense(512, activation="relu"),
    Dropout(0.5, seed=SEED),
    Dense(N_TYPES, activation="softmax")
])

model.summary()

optimizer = Adam(learning_rate=0.001, beta_1=0.869, beta_2=0.995)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_generator,
    validation_steps=validation_steps
)

# –––––––––––––––––––––––––––––––––––––– #
# Output Images and Labels Visualization #
# –––––––––––––––––––––––––––––––––––––– #
def plot_sample_predictions(model, test_generator, categories, num_samples=9, figsize=(12, 8)):
    """
    Nice display of prediction samples to see CNN predictions
    for classification.
    """
    # Make predictions on the test dataset
    predictions = model.predict(test_generator)
    predicted_categories = np.argmax(predictions, axis=1)
    true_categories = test_generator.classes

    # Randomly sample test images
    test_images = np.array(test_generator.filepaths)
    sample_indices = np.random.choice(len(test_images), size=num_samples, replace=False)
    sample_images = test_images[sample_indices]
    sample_predictions = [categories[predicted_categories[i]] for i in sample_indices]
    sample_true_labels = [categories[true_categories[i]] for i in sample_indices]

    # Plot sample images with their predicted and true labels
    plt.figure(figsize=figsize)
    
    # Loop over samples
    for i, image_path in enumerate(sample_images):
        # Form subplot and plot
        plt.subplot(3, 3, i + 1)
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis("off")
        
        # Set axis label color depending on correct prediction or not
        prediction_color = 'green' if sample_predictions[i] == sample_true_labels[i] else 'red'
        plt.title(f"Predicted: {sample_predictions[i]}\nTrue: {sample_true_labels[i]}", color=prediction_color)
        
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Plot sample predictions
plot_sample_predictions(model, test_generator, CLASS_TYPES, num_samples=9, figsize=(12, 8))

# –––––––––––––––––––––––––––––––––––––– #
#            Confusion matrix            #
# –––––––––––––––––––––––––––––––––––––– #
def CM(CNN_model, test_generator, categories):
    """
    Function to return the confusion matrix of a given CNN model.
    """
    from sklearn.metrics import confusion_matrix
    # Predictions on test dataset
    predictions = CNN_model.predict(test_generator)
    predicted_categories = np.argmax(predictions, axis=1)
    true_categories = test_generator.classes

    # Create a confusion matrix
    confusion_matrix_array = confusion_matrix(true_categories, predicted_categories)
    
    return confusion_matrix_array

# Compute confusion matrix
cm = CM(model, test_generator, CLASS_TYPES)
st.write("Confusion Matrix:")
st.write(cm)

# –––––––––––––––––––––––––––––––––––––– #
#             Metric Analysis            #
# –––––––––––––––––––––––––––––––––––––– #
def calculate_metrics(confusion_matrix, categories):
    """
    Function to calculate important metrics for multi-classification problems.
    """
    # Calculating 4 different metrics
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    # Printing the results based on each category
    for i, category in enumerate(categories):
        print(f"Class: {category.title()}")
        print(f"Precision: {precision[i]:.3f}")
        print(f"Recall: {recall[i]:.3f}")
        print(f"F1-Score: {f1_score[i]:.3f}\n")
        
    # Showing the total accuracy of the model
    print(f"\nAccuracy: {accuracy:.3f}")

# Calculate and display metrics
calculate_metrics(cm, CLASS_TYPES)
