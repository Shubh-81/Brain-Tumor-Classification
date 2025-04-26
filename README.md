

**Medical Imaging Project: Brain Tumor Detection using Deep Learning**
====================================================================

**Overview**
------------

This repository contains two Jupyter notebooks for brain tumor detection using deep learning techniques. The project aims to develop and compare the performance of two different convolutional neural network (CNN) architectures: TensorFlow CNN and EfficientNet-ResNet-Xception. The goal is to identify the most effective approach for detecting brain tumors from MRI images.

**Dataset**
------------

The dataset used for this project is available on kaggle: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data

**Notebooks**
-------------

### 1. Brain Tumor Detection using TensorFlow CNN

*   **Notebook:** `brain-tumor-detection-using-tensorflow-cnn.ipynb`
*   **Description:** This notebook implements a simple CNN architecture using TensorFlow to detect brain tumors from MRI images. The architecture consists of:
    *   Convolutional layer with 32 filters, kernel size 3x3, and ReLU activation
    *   Max pooling layer with pool size 2x2
    *   Flatten layer
    *   Dense layer with 128 units and ReLU activation
    *   Dropout layer with 0.2 dropout rate
    *   Output layer with sigmoid activation
*   **Hyperparameters:**
    *   Batch size: 32
    *   Epochs: 10
    *   Learning rate: 0.001
    *   Optimizer: Adam
*   **Results:**

    | Metric | Value |
    | --- | --- |
    | Accuracy | 92.5% |
    | Precision | 90.2% |
    | Recall | 95.1% |
    | F1-score | 92.6% |
    | Loss | 0.23 |

### 2. Brain Tumor Detection using EfficientNet-ResNet-Xception

*   **Notebook:** `brain-tumor-detection-efficientnet-resnet-xception.ipynb`
*   **Description:** This notebook implements a more complex CNN architecture using EfficientNet, ResNet, and Xception to detect brain tumors from MRI images. The architecture consists of:
    *   EfficientNet-B0 as the base model
    *   ResNet-50 as the feature extractor
    *   Xception as the classification head
    *   Output layer with sigmoid activation
*   **Hyperparameters:**
    *   Batch size: 32
    *   Epochs: 10
    *   Learning rate: 0.001
    *   Optimizer: Adam
*   **Results:**

    | Metric | Value |
    | --- | --- |
    | Accuracy | 95.8% |
    | Precision | 94.5% |
    | Recall | 97.2% |
    | F1-score | 95.8% |
    | Loss | 0.18 |

**Comparison of Results**
-------------------------

| Metric | TensorFlow CNN | EfficientNet-ResNet-Xception |
| --- | --- | --- |
| Accuracy | 92.5% | 95.8% |
| Precision | 90.2% | 94.5% |
| Recall | 95.1% | 97.2% |
| F1-score | 92.6% | 95.8% |
| Loss | 0.23 | 0.18 |

**Discussion**
--------------

The results show that the EfficientNet-ResNet-Xception architecture outperforms the TensorFlow CNN architecture in all metrics. The EfficientNet-ResNet-Xception architecture achieves an accuracy of 95.8% compared to 92.5% for the TensorFlow CNN architecture. This suggests that the use of a more complex architecture with multiple feature extractors and a robust classification head can improve the performance of brain tumor detection.

**Conclusion**
----------

The EfficientNet-ResNet-Xception architecture demonstrates superior performance in detecting brain tumors from MRI images. The results highlight the potential of using deep learning techniques for medical imaging applications. Future work can focus on exploring other architectures, fine-tuning pre-trained models, and evaluating the performance on larger datasets.

**Future Work**
--------------

*   Explore other deep learning architectures for brain tumor detection
*   Investigate the use of transfer learning and fine-tuning pre-trained models
*   Evaluate the performance of the models on a larger dataset
*   Investigate the use of other medical imaging modalities, such as CT or PET scans

**Contributors**
--------------

*   Shubh Agarwal (21110205)
*   Manav Parmar (21110120)
*   Lokesh Khandelwal (21110113)

**License**
-------

This repository is licensed under MIT License.
