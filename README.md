# Deepcats: a WGAN-GP model trained to generate images of cats

<img width="1088" height="1088" alt="image" src="https://github.com/user-attachments/assets/85cbd917-04c2-4dde-bbaf-d4ccf69e4e4d" />



## How to train
Step 1: download dataset and extract to working directory: https://www.kaggle.com/datasets/crawford/cat-dataset
NOTE that the working dir must NOT contain any images or subfolders that contain images

Step 2: ```pip install torch torchvision functorch adabelief-pytorch```

Step 3: create folders ```fakecats``` and ```models``` in working directory

Step 4: transfer-train WGAN-GP: ```python deepcats-gan.py```

Step 5: evaluate: ```python deepcats-eval.py```
