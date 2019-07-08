import pickle
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    with open('train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    """
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[-1]): 
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(np.reshape(img, [28,28]), cmap='Greys_r')
    plt.show()
    

    