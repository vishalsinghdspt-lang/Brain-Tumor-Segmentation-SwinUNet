# Brain Tumor Segmentation using Swin-UNet

## 1. Abstract

Brain tumor segmentation is an important task in medical image analysis. Manual tumor delineation from MRI scans is time-consuming and depends on expert radiologists. This project presents an automated brain tumor segmentation system using the Swin-UNet deep learning architecture and the BraTS 2021 dataset.

The system processes four MRI modalities including T1, T1ce, T2 and FLAIR. Image preprocessing techniques such as normalization, resizing and augmentation are applied before training. The model is trained using Dice Loss combined with Cross Entropy Loss and optimized using AdamW.

A Flask-based web application is developed to allow users to upload MRI scans, generate tumor segmentation predictions, view WT, TC and ET visualizations, compare predictions with ground truth masks, download results and maintain prediction history. The system is evaluated using Dice Score, IoU, Training Loss and Validation Loss.

The proposed system demonstrates the effectiveness of transformer-based medical image segmentation and provides a user-friendly platform for brain tumor analysis.

## 2. Introduction

Brain tumors are among the most dangerous neurological disorders affecting human health. Magnetic Resonance Imaging (MRI) is widely used to detect and analyze brain tumors because it provides detailed information about brain tissues.

Traditional manual segmentation of tumor regions is labor-intensive and prone to variability among radiologists. Deep learning methods have significantly improved the efficiency and accuracy of medical image segmentation.

Swin-UNet combines the strength of the U-Net encoder-decoder architecture with Swin Transformer blocks. This allows the model to capture both local and global image features, improving segmentation performance.

This project develops a complete Swin-UNet based brain tumor segmentation system along with an interactive Flask web application for visualization and analysis.

## 3. Problem Statement

Brain tumor identification and segmentation from MRI scans is a challenging task in medical image analysis. Manual segmentation performed by radiologists is time-consuming, labor-intensive and subject to human variability.
 
Traditional image processing techniques often fail to accurately capture complex tumor boundaries and different tumor subregions. Therefore, there is a need for an automated and accurate brain tumor segmentation system capable of identifying tumor regions from multi-modal MRI scans.
 
This project addresses the problem by implementing a Swin-UNet based deep learning model that automatically segments brain tumors using the BraTS 2021 dataset and provides prediction visualization through a Flask web application.

## 4. Project Objectives

The objectives of this project are:

- To develop an automatic brain tumor segmentation system using Swin-UNet.
- To utilize the BraTS 2021 dataset for model training and evaluation.
- To preprocess MRI scans using normalization, resizing and augmentation techniques.
- To evaluate model performance using Dice Score and IoU metrics.
- To compare Swin-UNet performance with standard U-Net.
- To create a Flask-based web application for MRI upload and prediction.
- To visualize Whole Tumor (WT), Tumor Core (TC) and Enhancing Tumor (ET).
- To allow users to download and store prediction results.

## 5. Literature Review

Brain tumor segmentation has been an active research area in medical image analysis. Traditional image segmentation methods relied on thresholding, region growing and edge detection techniques. These methods often struggled with variations in tumor shape, size and intensity.

The introduction of Deep Convolutional Neural Networks (CNNs) significantly improved segmentation performance. U-Net became one of the most widely used architectures due to its encoder-decoder structure and skip connections.

Recent advancements in Vision Transformers have further improved segmentation accuracy. Swin Transformer introduces hierarchical feature extraction and shifted window attention mechanisms that capture both local and global image features efficiently.

Swin-UNet combines the strengths of U-Net and Swin Transformers to provide improved segmentation performance for medical imaging tasks. This project utilizes Swin-UNet for automatic brain tumor segmentation using the BraTS 2021 dataset.

## 6. Existing System

Traditional brain tumor diagnosis relies on radiologists manually examining MRI scans and identifying tumor boundaries. Manual segmentation requires significant time and expertise and may lead to inter-observer variability.

Conventional image processing methods such as thresholding, clustering and edge detection provide limited performance when applied to complex tumor structures.

Deep learning methods such as CNN-based U-Net improved segmentation performance; however, they may struggle to capture long-range dependencies and global contextual information in large medical images.

These limitations motivated the development of transformer-based segmentation architectures such as Swin-UNet.

## 7. Proposed System

The proposed system uses Swin-UNet, a transformer-based medical image segmentation architecture, to automatically segment brain tumors from MRI scans.

The system accepts four MRI modalities including T1, T1ce, T2 and FLAIR as input. MRI images are preprocessed using normalization, resizing and augmentation techniques before being passed to the model.

The trained model predicts tumor segmentation masks and generates visualizations for Whole Tumor (WT), Tumor Core (TC) and Enhancing Tumor (ET).

A Flask-based web application provides an interactive interface for users to upload MRI scans, generate predictions, visualize results, compare predictions with ground truth masks and download output images.

## 8. Dataset Description

This project utilizes the Brain Tumor Segmentation (BraTS) 2021 dataset.

The dataset contains multi-modal MRI scans collected from brain tumor patients. Each patient record includes:

- FLAIR MRI
- T1 MRI
- T1ce MRI
- T2 MRI
- Ground Truth Segmentation Mask

A total of 300 patient cases were selected for training and evaluation purposes.

The dataset was divided at the patient level into:

- 60% Training Data (180 Patients)
- 40% Validation Data (120 Patients)

The dataset was loaded using the Nibabel library for NIfTI file processing.

## 9. Image Preprocessing

Image preprocessing plays an important role in improving model performance and training stability.

The following preprocessing techniques were applied:

### Normalization

MRI intensity values were normalized to zero mean and unit variance.

### Resizing

Images and corresponding segmentation masks were resized to a standard spatial resolution.

### Data Augmentation

The following augmentation techniques were applied:

- Horizontal Flipping
- Rotation Augmentation

### Multi-Modal Input

Four MRI modalities were stacked into a single 4-channel input tensor for model training.

## 10. Swin-UNet Architecture

The proposed segmentation model is based on the Swin-UNet architecture implemented using PyTorch and MONAI.

Swin-UNet combines:

- U-Net Encoder-Decoder Structure
- Swin Transformer Blocks
- Skip Connections

The model processes multi-modal MRI input and produces multi-class segmentation outputs.

Model Configuration:

- Input Channels: 4
- Output Classes: 4
- Feature Size: 48

The architecture captures both local and global image information, improving segmentation performance compared to conventional CNN-based approaches.

## 11. Training Methodology

The Swin-UNet model was trained using the BraTS 2021 dataset.

Training Procedure:

1. Load MRI dataset
2. Apply preprocessing
3. Generate training batches
4. Perform forward propagation
5. Compute loss
6. Perform backpropagation
7. Update model weights
8. Validate model performance
9. Save checkpoints

Optimizer:
AdamW

Loss Function:
Combination of Dice Loss and Cross Entropy Loss

The training process monitored both training loss and validation loss throughout the training phase.

## 12. Evaluation Methodology

Model performance was evaluated using standard medical image segmentation metrics.

### Dice Similarity Coefficient (DSC)

Dice Score measures the overlap between predicted and ground truth segmentation masks.

### Intersection over Union (IoU)

IoU measures the ratio between intersection and union of predicted and actual tumor regions.

### Training Loss

Training loss was monitored during model optimization.

### Validation Loss

Validation loss was used to evaluate model generalization capability.

## 13. Flask Web Application

A Flask-based web application was developed to provide an interactive interface for end users.

The application includes:

Figure 1: Home Page
Location: report_images/home_page.png

Figure 2: Dashboard Page
Location: report_images/dashboard_page.png

Figure 3: Upload MRI Page
Location: report_images/upload_page.png

Figure 4: Result Page
Location: report_images/result_page.png

Figure 5: History Page
Location: report_images/history_page.png

Figure 6: About Page
Location: report_images/about_page.png

### Home Page

Provides project overview and navigation.

### Dashboard

Displays performance statistics, visualization metrics and project information.

### Upload Page

Allows users to upload MRI scans.

### Result Page

Displays:

- Original MRI
- Ground Truth
- Predicted Mask
- WT Visualization
- TC Visualization
- ET Visualization

### History Page

Stores and displays previous prediction records using SQLite.

### About Page

Provides project details and implementation information.

## 14. Results and Discussion

The trained Swin-UNet model successfully segmented brain tumor regions from MRI scans.

Generated outputs included:

- Predicted Segmentation Masks
- Whole Tumor (WT)
- Tumor Core (TC)
- Enhancing Tumor (ET)

Performance Metrics:

- Dice Score: 0.415
- IoU Score: 0.218
- Training Loss: 12.46
- Validation Loss: 10.81

Training and validation curves were generated to analyze learning behavior.

The results demonstrate the effectiveness of Swin-UNet for brain tumor segmentation tasks.

## 15. Testing

The system was tested under multiple scenarios.

### Test Case 1

Upload valid MRI files.

Expected Result:
Successful prediction generation.

Status:
Passed

### Test Case 2

Upload incomplete MRI modalities.

Expected Result:
Validation error displayed.

Status:
Passed

### Test Case 3

Download prediction results.

Expected Result:
Output image downloaded successfully.

Status:
Passed

### Test Case 4

View prediction history.

Expected Result:
Stored records displayed successfully.

Status:
Passed

## 16. Conclusion

This project successfully implemented a complete brain tumor segmentation system using Swin-UNet and the BraTS 2021 dataset.

The system integrates MRI preprocessing, deep learning-based segmentation, evaluation metrics and a Flask web application into a single platform.

The developed application provides an efficient and user-friendly solution for visualizing and analyzing brain tumor segmentation results.

The project demonstrates the practical application of transformer-based deep learning techniques in medical image analysis.

## 17. Future Scope

Several improvements can be incorporated in future versions:

- Hyperparameter Optimization
- Conditional Random Fields (CRF)
- Morphological Post-Processing
- Additional Dataset Support
- Multi-GPU Training
- Cloud Deployment
- Real-Time Medical Imaging Integration
- Enhanced Visualization Features

## 18. References

1. BraTS 2021 Benchmark Dataset.
2. Swin-UNet: Unet-like Pure Transformer for Medical Image Segmentation.
3. MONAI Documentation.
4. PyTorch Documentation.
5. Flask Documentation.
6. Nibabel Documentation.
7. IEEE Publications on Brain Tumor Segmentation.
8. Medical Image Analysis Research Papers.
