# Muntaha — Learning Notes & Resources

## Phase 1 — CNN from Scratch
I am documenting whatever I am learning. 
https://medium.com/@journeycoding248

### Foundations: My First Neural Network (Social Network Ads — Purchase Prediction)

### What I built
A feedforward (dense/multi-layer) neural network in TensorFlow/Keras to predict 
whether a user will purchase a product (binary classification: 0 = No, 1 = Yes) 
based on two features — Age and Estimated Salary — using the Social Network Ads 
dataset. This was my first hands-on neural network built end-to-end, before 
moving into CNNs for the actual leaf disease classifier.

### Dataset
Social_Network_Ads.csv — 400 rows, features: Age, EstimatedSalary, 
target: Purchased (binary)

### What I did
- Loaded and explored the data with pandas
- Split into train/test sets using `train_test_split` from scikit-learn
- Built a simple dense neural network using TensorFlow/Keras
- Trained and evaluated the model on the held-out test set

### Resources used
- Zero to Mastery: Machine Learning & AI course (TensorFlow/deep learning section)

### What I got stuck on / had to debug
- Kaggle dataset path resolution (dataset was mounted under 
  `/kaggle/input/datasets/muntahapolaris/dataset/`, took a moment to locate 
  correctly via `os.walk`)
- Hit a `KeyboardInterrupt` during training at one point — had to manage 
  session/runtime carefully

### What I understood better after this
- The actual mechanics of splitting data into train/test with sklearn
- How raw tabular features get fed into a dense neural network
- First real hands-on feel for defining, compiling, and training a Keras model 
  before adding the complexity of convolutional layers
- This is my "Phase 0" — the foundational rep before Phase 1 (CNN on PlantVillage)

### Today I worked with handling imbalanced dataset
- Today I used the Churn prediction dataset to practice dealing with imbalanced dataset.
- I worked with four techniques random undersampling, random oversampling, SMOTE, and ensemble bagging with undersampling


### CNN
I am starting working with CNN. I shall soon prepare my medium post covering all the theory that I covered.
For the code practice I am building project using dataset: https://www.kaggle.com/c/cifar-10
I shall make separate repository for this or same repo with separate readme.md
