# Deepcats: a WGAN-GP model trained to generate images of cats

![image](https://github.com/user-attachments/assets/9dffe1b3-7557-4083-8b3b-b46dbd34789b)



## How to train
Step 1: download dataset and extract to working directory: https://www.kaggle.com/datasets/crawford/cat-dataset
NOTE that the working dir must NOT contain any images or subfolders that contain images

Step 2: ```pip install torch torchvision functorch flash-attn adabelief-pytorch```

Step 3: create folders ```fakecats``` and ```models``` in working directory

Step 4: transfer-train WGAN-GP: ```python deepcats-gan.py```

Step 5: evaluate: ```python deepcats-eval.py```
