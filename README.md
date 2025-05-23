# Deepcats: a WGAN-GP model trained to generate images of cats

![image](https://github.com/user-attachments/assets/1cd83086-7571-4563-aa73-3a6ad476e9d0)



## How to train
Step 1: download dataset and extract to working directory: https://www.kaggle.com/datasets/crawford/cat-dataset
NOTE that the working dir must NOT contain any images or subfolders that contain images

Step 2: ```pip install torch torchvision functorch flash-attn adabelief-pytorch```

Step 3: create folders ```fakecats``` and ```models``` in working directory

Step 4: transfer-train WGAN-GP: ```python deepcats-gan.py```

Step 5: evaluate: ```python deepcats-eval.py```
