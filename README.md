# Landscape-GAN
Generate fake landscapes using a Generative Adversarial Network

![Alt Text](https://media.giphy.com/media/f6P4SEJUPdRuKtHE9p/giphy.gif)

The above gif contains images generated during the training of a Generative Adversial Network using landscapes as training data. The training set contained ~3500 images obtained from a [Kaggle](https://www.kaggle.com/arnaud58/landscape-pictures) dataset. Data converted to a numpy file for quicker processing, as image processing took around 10-20 minutes each time. All training was done using Google Cloud computing on two GPUs, which is indicated in GAN.py. After training was finished (~9 hours), I was left with a generator.h5 file that could be used to generate these landscapes.

While I mainly worked on this project out of interest, I learned a lot about these networks and am pretty happy with how my final results turned out. In the future, I hope to retrain at a higher resolution and for a longer time to see how much improvement is possible.

I included code for the network, as well as my sampled results and .npy file in case anyone reading this wanted to test it out for themselves. The loadNewData parameter passed into the main training method should be adjusted according to what the user plans to train on. Plus, GPU training can be adjusted through the os.environ assignment near the top, and by the numGPUs parameter in the main training method.

Enjoy!

I referenced code from [here](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb) and [here](https://skymind.ai/wiki/generative-adversarial-network-gan) to provide a starting point for my own training.
