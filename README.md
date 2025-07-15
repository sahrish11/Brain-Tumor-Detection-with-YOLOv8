# Brain-Tumor-Detection-with-YOLOv8

## Overview  
This Jupyter Notebook implements object detection using the YOLOv8n model on a custom dataset as part of the EE4211 project. The analysis explores:  
- Training a lightweight YOLO model on annotated image data  
- Evaluating performance metrics and detection accuracy  
- Testing detection results on unseen images  

## Key Questions Explored  
1. How effective is the YOLOv8n model for real-time object detection on a small custom dataset?  
2. What are the strengths and limitations of using a lightweight model like YOLOv8n?  
3. How can training parameters and data augmentation improve detection accuracy?  

## Setup  

### Libraries Used  
- `ultralytics`: YOLOv8 training and inference  
- `opencv-python`: Image loading and annotation  
- `matplotlib`: Visualizing training metrics and predictions  
- `os`, `shutil`: File and directory management  

### Data Collection  
- **Dataset**: Custom-labeled object detection dataset (in YOLO format)  
- **Annotation Format**: YOLO `.txt` with bounding boxes and class labels  
- **Directory Structure**:  
  - `data.yaml`: Configuration file for training  
  - `images/train`, `images/val`: Training and validation images  
  - `labels/train`, `labels/val`: Corresponding labels  

## Steps  
1. **Environment Setup**: Installed `ultralytics` and verified setup  
2. **Data Preparation**: Organized dataset and updated `data.yaml`  
3. **Model Training**: Trained YOLOv8n using default or tuned hyperparameters  
4. **Performance Evaluation**: Analyzed training loss, precision, recall, and mAP  
5. **Inference & Visualization**: Ran inference on test images and visualized results  
6. **Model Export**: Saved trained model for future deployment  

## Key Findings  

### Model Performance  
- YOLOv8n achieved fast training and inference with limited compute usage  
- Detection accuracy was reasonable for lightweight architecture on a small dataset  

### Visualization  
- Bounding boxes aligned well with objects of interest  
- Some false positives/negatives observed in edge cases  

### Metrics Summary  
- **Precision**: ~0.94 
- **Recall**: ~0.84  
- **mAP50**: ~0.92  

## Conclusion  
The notebook demonstrates how to effectively train and evaluate a YOLOv8n model on a small, custom object detection task. Key insights include:  
- YOLOv8n is well-suited for resource-constrained applications  
- Training quality is heavily influenced by annotation accuracy and data diversity  
- Performance can be improved with data augmentation or hyperparameter tuning  

## How to Use This Notebook  
1. Install required libraries (`!pip install ultralytics opencv-python`)  
2. Prepare dataset and `data.yaml` file in the correct format  
3. Run training and evaluation cells in sequence  
4. Test detection with `model.predict()` on new images  

## Future Work  
- Use YOLOv8m/l/x variants for improved accuracy  
- Implement data augmentation and mosaic training  
- Deploy trained model to mobile or edge devices using ONNX or TensorRT  

**Note**: Ensure GPU acceleration (e.g., Colab or local CUDA) is enabled for faster training. YOLOv8n is optimized for speed, not necessarily highest accuracy.
