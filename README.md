# data-centric-image-classification
Applying various data engineering techniques into image classification task for KAIST DS801 term project

1. What we’ve done so far
      * Outlier detection: 
          * Based on the distances between image embedding and its class' average
          * Based on the distance towards the cluster centroid from 10-means clustering of the embeddings
      * Handling noisy labels:
          * Combining sample selection and relabeling using pre-trained EfficientNet
          * Fitted accumulated losses on 2 GMM and remove images that have 95% confidence coming from the larger loss (noise) distribution
          * Perform relabeling only when confidence > 90%
          * Final selection using confidence from ResNet-34
      * Tackling data imbalances
          * Using focal loss when handling noisy labels (pre-training, accumulating loss for sample selection, and when training for relabeling)
          * Augmentation only on images with high confidence from ResNet-34 using AutoAugment (ImageNet) and Random Cutmix + Mixup
