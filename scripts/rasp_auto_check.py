# OS choice: strongly recommended to use Raspberry Pi OS Lite (32-bit). It has no GUI,
# leaving all the precious memory for the AI model.
# Install core dependencies: after connecting via SSH, run the following to install base tools:
# sudo apt-get update
# sudo apt-get install -y python3-pip python3-pil python3-numpy libatlas-base-dev
# pip3 install onnxruntime

# sudo nano /etc/dphys-swapfile
# # Change CONF_SWAPSIZE from 100 to 1024
# sudo /etc/init.d/dphys-swapfile restart

import os
import time
import datetime
import smtplib
import numpy as np
from PIL import Image
from subprocess import call
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import onnxruntime as ort

# ================= Configuration =================
MODEL_PATH = "/home/pi/leaf_checker/model.onnx"
LABEL_PATH = "/home/pi/leaf_checker/labels.txt"
LOG_PATH = "/home/pi/leaf_checker/growth_log.txt"
IMAGE_PATH = "/home/pi/leaf_checker/latest_leaf.jpg"

# Email configuration
SENDER_EMAIL = "your-account@qq.com"
AUTH_CODE = "your-16-digit-auth-code"  # obtained from your email provider's settings
RECEIVER_EMAIL = "recipient@example.com"
# =========================================

def send_email(result_text, image_path):
    """Send an email with the recognition result and photo attached"""
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = f"[Plant Report] {result_text}"

    body = f"Raspberry Pi monitoring report:\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nStatus: {result_text}"
    msg.attach(MIMEText(body, "plain"))

    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                img_data = f.read()
            image = MIMEImage(img_data, name="leaf.jpg")
            msg.attach(image)

        server = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=30)
        server.login(SENDER_EMAIL, AUTH_CODE)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def capture_image():
    """Take a photo using the new libcamera stack"""
    # Limit resolution to save memory and speed up sending
    call(["libcamera-still", "-o", IMAGE_PATH, "--immediate", "--width", "800", "--height", "600"])

def predict():
    """Run MobileNet V3 inference"""
    if not os.path.exists(MODEL_PATH):
        return "Model file missing", 0.0

    # Load labels
    with open(LABEL_PATH, "r", encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    # Initialize the inference engine
    session = ort.InferenceSession(MODEL_PATH)

    # Preprocess the image
    img = Image.open(IMAGE_PATH).convert('RGB').resize((224, 224))
    img_data = np.array(img).transpose(2, 0, 1).astype(np.float32)

    # Normalize (must match PyTorch training)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_data = (img_data / 255.0 - mean) / std
    img_data = np.expand_dims(img_data, axis=0)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})

    # Parse results
    probs = outputs[0][0]
    idx = np.argmax(probs)
    return labels[idx], probs[idx]

def main():
    print(f"--- Task started: {datetime.datetime.now()} ---")
    try:
        capture_image()
        label, conf = predict()
        result_str = f"{label} (confidence: {conf:.2f})"

        # Log the result
        with open(LOG_PATH, "a") as f:
            f.write(f"{datetime.datetime.now()}: {result_str}\n")

        # Send the email
        send_email(result_str, IMAGE_PATH)

    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        print(error_msg)
        with open(LOG_PATH, "a") as f:
            f.write(f"{datetime.datetime.now()}: ERROR {error_msg}\n")

if __name__ == "__main__":
    main()

# Steps to set up the scheduled (cron) job:

# In SSH, run crontab -e.

# Add the following two lines at the bottom of the file (runs every day at 8am and 8pm):
# 0 8 * * * /usr/bin/python3 /home/pi/leaf_checker/auto_check.py >> /home/pi/leaf_checker/cron_log.txt 2>&1
# 0 20 * * * /usr/bin/python3 /home/pi/leaf_checker/auto_check.py >> /home/pi/leaf_checker/cron_log.txt 2>&1
