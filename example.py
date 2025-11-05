#!/usr/bin/env python3
"""
Example script showing how to use the spectral unmixing functions.
"""

from func import Unmix

# Example 1: Basic usage with default parameters
# Uncomment and modify paths to use:
# abundance_maps = Unmix(
#     raw_path='path/to/your/image.raw',
#     hdr_path='path/to/your/image.hdr',
#     q=3  # Number of endmembers (e.g., rock, paint, background)
# )

# Example 2: Using PPI for endmember extraction
# abundance_maps = Unmix(
#     raw_path='path/to/your/image.raw',
#     hdr_path='path/to/your/image.hdr',
#     EM_method='PPI',
#     q=3,
#     abun_method='FCLSU'
# )

# Example 3: Using NMF for abundance estimation
# abundance_maps = Unmix(
#     raw_path='path/to/your/image.raw',
#     hdr_path='path/to/your/image.hdr',
#     EM_method='NFINDR',
#     q=3,
#     abun_method='NMF'
# )

# Example 4: With cropping and normalization disabled
# abundance_maps = Unmix(
#     raw_path='path/to/your/image.raw',
#     hdr_path='path/to/your/image.hdr',
#     q=3,
#     crop_region=(100, 100, 500, 500),  # (x, y, width, height)
#     normalize=False
# )

print("Example script - modify paths and uncomment examples to use")
print("For command-line usage, see: python main.py --help")

