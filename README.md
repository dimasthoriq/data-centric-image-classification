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
      * Tackling data imbalances
          * Using focal loss when handling noisy labels (pre-training, accumulating loss for sample selection, and when training for relabeling)

2. Timeline
      * ~24/05/27: Setting up image augmentation method and saving the final training images into one folder
      * ~24/05/29: Training the baseline reference model on our training images and then evaluate on our validation images
      * ~24/06/03: Modify based on the evaluation
