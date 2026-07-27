# Code snippet for the part that runs on the Raspberry Pi

# 1. Load weights (only needs to happen once when the program starts)
cam_weights = np.load("/home/pi/leaf_checker/cam_weights.npy")

def predict_and_explain():
    # ... image preprocessing code (same as before) ...

    # 2. Run inference (ONNX Runtime)
    # Note: session.run now returns two results
    outputs = session.run(['class_logits', 'feature_maps'], {'input_image': img_data})

    logits = outputs[0][0]       # classification scores
    feature_maps = outputs[1][0] # feature maps (576, 7, 7)

    idx = np.argmax(logits)      # predicted class ID

    # 3. Compute CAM (generate the heatmap)
    # Take out the weight vector for this class
    w_k = cam_weights[idx]       # Shape: (576,)

    # Matrix multiplication: weights * feature maps
    # Use numpy broadcasting or einsum for efficient computation
    # Formula: Heatmap = Sum(Weight_i * FeatureMap_i)
    cam = np.einsum('c,chw->hw', w_k, feature_maps)

    # ... subsequent ReLU, normalization, overlay on image (see earlier answer) ...
