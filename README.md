# Deepcats: a WGAN-GP model trained to generate images of cats

![image](https://github.com/user-attachments/assets/c10d2507-88b2-4356-bcae-f5b2a1c761ba)

![image](https://github.com/user-attachments/assets/073da792-be1a-4673-adbf-98c785c2d865)


## How to train
Step 1: download dataset and extract to working directory: https://www.kaggle.com/datasets/crawford/cat-dataset
NOTE that the working dir must NOT contain any images or subfolders that contain images

Step 2: ```pip install torch torchvision functorch flash-attn adabelief-pytorch```

Step 3: create folders ```fakecats``` and ```models``` in working directory

Step 4: pre-train inverse GAN : ```python deepcats-inverse-gan.py```

Step 5: delete everything in ```fakecats``` folder

Step 6: delete everything in ```models``` folder except ```encoder_100000``` and ```decoder_100000```

Step 7: transfer-train WGAN-GP: ```python deepcats-gan.py```

Step 8: evaluate: ```python deepcats-eval.py```

## Deepcats-specific quirks
Deepcats is trained in TWO phases: the Inverse GAN autoencoder and the WGAN-GP.

The Inverse GAN autoencoder is a special autoencoder that uses a GAN discriminator to penalize the KL-divergence between the encoder's latent space and a standard normal distribution.
This encourages the encoder's latent representation to be as close to a standard normal distribution as possible.
Because the encoder's latent representation is extremely close to a standard normal distribution, it should generate novel images when fed random noise drawn from a standard normal distribution.
The decoder (generator) is first trained on the autoencoder objective and then transferred trained to the WGAN-GP objective.
When learning the WGAN-GP objective, the decoder receives a divergence penalty on how much it has forgotten the autoencoder objective, which prevents training instability and catastrophic forgetting.
