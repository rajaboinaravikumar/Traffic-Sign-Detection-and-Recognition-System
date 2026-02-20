import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ---------------- GLOBAL VARIABLES ----------------
train_features = []
train_labels = []
train_paths = []
train_brightness = []

test_features = []
test_paths = []

model = None
label_encoder = LabelEncoder()
preprocess_applied = False

cnn_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

# ---------------- UTILITY FUNCTIONS ----------------
def log(msg):
    output_box.insert(tk.END, msg + "\n")
    output_box.see(tk.END)
    root.update_idletasks()

def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def resize_image(img, size=(224,224)):
    return img.resize(size)

# ---------------- FEATURE EXTRACTION ----------------
def extract_labeled_features(base_folder):
    features, labels, brightness, paths = [], [], [], []

    for label in os.listdir(base_folder):
        class_path = os.path.join(base_folder, label)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_path, file)
                img = safe_open_image(img_path)
                if img is None:
                    continue

                if preprocess_applied:
                    img = ImageOps.exif_transpose(img)

                img = resize_image(img)
                arr = keras_image.img_to_array(img)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)

                feat = cnn_model.predict(arr, verbose=0).flatten()
                features.append(feat)

                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                brightness.append(np.mean(gray))

                labels.append(label)
                paths.append(img_path)

    return np.array(features), np.array(labels), np.array(brightness), np.array(paths)

def extract_features(folder):
    feats, paths = [], []
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, file)
            img = safe_open_image(img_path)
            if img is None:
                continue

            if preprocess_applied:
                img = ImageOps.exif_transpose(img)

            img = resize_image(img)
            arr = keras_image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)

            feat = cnn_model.predict(arr, verbose=0).flatten()
            feats.append(feat)
            paths.append(img_path)

    return np.array(feats), np.array(paths)

# ---------------- BUTTON FUNCTIONS ----------------
def load_train():
    global train_features, train_labels, train_brightness, train_paths
    folder = filedialog.askdirectory()
    if folder:
        train_features, train_labels, train_brightness, train_paths = extract_labeled_features(folder)
        log("✔ TRAIN DATA LOADED SUCCESSFULLY.")
        log(f"• Total Training Images : {len(train_labels)}")

        classes, counts = np.unique(train_labels, return_counts=True)
        for c, n in zip(classes, counts):
            log(f"• {c} : {n}")

def load_test():
    global test_features, test_paths
    folder = filedialog.askdirectory()
    if folder:
        test_features, test_paths = extract_features(folder)
        log("✔ TEST DATA LOADED SUCCESSFULLY.")
        log(f"• Total Test Images : {len(test_features)}")

def Preprocess():
    global preprocess_applied
    preprocess_applied = True
    log("✔ PREPROCESSING COMPLETED SUCCESSFULLY")

def train_model():
    global model
    if len(train_features) == 0:
        log("❌ Load training data first")
        return

    y = label_encoder.fit_transform(train_labels)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_features, y)

    acc = np.mean(model.predict(train_features) == y)
    log("✔ MODEL TRAINED SUCCESSFULLY")
    log("• Algorithm : Random Forest")
    log(f"• Training Accuracy : {round(acc*100,2)} %")

# ---------------- EDA ----------------
def eda_class_count():
    if len(train_labels) == 0:
        log("❌ Load training data first")
        return
    labels, count = np.unique(train_labels, return_counts=True)
    plt.figure(figsize=(8,4))
    sns.barplot(x=labels, y=count)
    plt.xticks(rotation=45)
    plt.title("Traffic Sign Class Distribution")
    plt.show()

def eda_brightness():
    if len(train_brightness) == 0:
        log("❌ Load training data first")
        return
    plt.figure(figsize=(7,4))
    sns.histplot(train_brightness, bins=25, kde=True)
    plt.title("Image Brightness Distribution")
    plt.show()

def eda_pca():
    if len(train_features) == 0:
        log("❌ Load training data first")
        return
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(train_features)
    plt.figure(figsize=(6,5))
    plt.scatter(reduced[:,0], reduced[:,1], c=label_encoder.transform(train_labels))
    plt.title("PCA Feature Distribution")
    plt.show()

# ---------------- PREDICTION (UPDATED AS REQUESTED) ----------------
def predict_image():
    if model is None:
        log("❌ Train model first")
        return

    img_path = filedialog.askopenfilename()
    if not img_path:
        return

    img = safe_open_image(img_path)
    if preprocess_applied:
        img = ImageOps.exif_transpose(img)

    img_resized = resize_image(img)
    arr = keras_image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    feat = cnn_model.predict(arr, verbose=0).flatten().reshape(1, -1)
    pred = model.predict(feat)
    label = label_encoder.inverse_transform(pred)[0]

    # ---- LOG RESULT IN TEXT BOX ----
    log("✔ PREDICTION COMPLETED")
    log(f"• Predicted Traffic Sign : {label}")

    # ---- POPUP WINDOW ----
    popup = tk.Toplevel(root)
    popup.title("Prediction Result")
    popup.geometry("300x350")
    popup.resizable(True, True)

    display_img = Image.open(img_path).resize((220,220))
    popup_img = ImageTk.PhotoImage(display_img)

    img_label = tk.Label(popup, image=popup_img)
    img_label.image = popup_img
    img_label.pack(pady=10)

    text_label = tk.Label(
        popup,
        text=f"PREDICTED : {label.upper()}",
        font=("Arial", 14, "bold")
    )
    text_label.pack(pady=10)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Traffic Sign Recognition System")
root.geometry("1200x850")

tk.Label(
    root,
    text="Traffic Sign Detection and Recognition System",
    font=("Arial",18,"bold")
).pack(pady=6)

main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_frame = tk.Frame(main_frame, width=260)
left_frame.pack(side="left", fill="y")
left_frame.pack_propagate(False)

right_frame = tk.Frame(main_frame)
right_frame.pack(side="left", fill="both", expand=True)

btn_opts = {"width":25, "height":2, "bg":"lightgreen"}

tk.Button(left_frame, text="Load Train Folder", command=load_train, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="Load Test Folder", command=load_test, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="Preprocess", command=Preprocess, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="Train Model", command=train_model, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="EDA Class Count", command=eda_class_count, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="EDA Brightness", command=eda_brightness, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="EDA PCA", command=eda_pca, **btn_opts).pack(pady=2)
tk.Button(left_frame, text="Predict Image", command=predict_image, **btn_opts).pack(pady=2)

output_box = tk.Text(right_frame, height=18, width=92)
output_box.pack(fill="both", expand=True)

root.mainloop()
