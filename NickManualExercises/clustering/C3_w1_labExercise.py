import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from kmeans_manual import kmeans


# Global variables
current_file = Path(__file__)
project_root = current_file.parent.parent.parent
# image_path = project_root / 'C3_UnsupervisedLearning' / 'W1' / 'Lab' / 'bird_small.png' # original image
# tgt_compression = 0.998
# max_iters = 16
image_path = 'reindeer.jpg' # more fun image
tgt_compression = 0.9997
K_override = None
max_iters = 1


def count_unique_colors(img):
    """Count unique colors in an image after converting to uint8."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        pixels = (img * 255).astype(np.uint8)
    else:
        pixels = img.astype(np.uint8)
    pixels = pixels.reshape(-1, pixels.shape[-1])
    return len(np.unique(pixels, axis=0))

def display_images(original, compressed, num_colors, compression_pct):
    """Display original and compressed images side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

    # Handle float images (0-1) and int images (0-255)
    if original.dtype in [np.float32, np.float64]:
        ax1.imshow(original)
        ax2.imshow(compressed)
    else:
        ax1.imshow(original/255)
        ax2.imshow(compressed/255)

    ax1.set_title('Original')
    ax2.set_title(f'Compressed ({num_colors} colors)\n{compression_pct:.4f}% compression')
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.tight_layout()
    plt.show()

def main():
    # Read image
    original_img = plt.imread(str(image_path))

    # Count colors and reshape image for processing
    num_unique_colors = count_unique_colors(original_img)
    X_img = original_img.reshape(-1, 3)

    # Calculate number of clusters
    if K_override is None:
        K = max(1, int(np.round(num_unique_colors * (1 - tgt_compression))))
        print(f"Target k: {K} from target compression: {tgt_compression}")
    else:
        K = K_override
        print(f"Hardcoded k: {K}")


    # Perform k-means clustering
    cluster_labels, centroids = kmeans(
        X_img,
        k=K,
        max_iters=max_iters,
        verbose=True,
        reduceClusters=False
    )

    # Reconstruct compressed image
    X_recovered = centroids[cluster_labels]
    X_recovered = X_recovered.reshape(original_img.shape)

    # Calculate compression statistics
    colors_compressed = count_unique_colors(X_recovered)
    compression_pct = (1 - colors_compressed / num_unique_colors) * 100

    print(f"Compressed from {num_unique_colors:,} to {colors_compressed:,} colors")
    print(f"Achieved {compression_pct:.4f}% compression")

    # Display results
    display_images(original_img, X_recovered, colors_compressed, compression_pct)

if __name__ == "__main__":
    main()
