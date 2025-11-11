import os
import numpy as np
import matplotlib.pyplot as plt
import re
from osgeo import gdal
from sklearn.decomposition import NMF
import pysptools.eea as eea
from sklearn.cluster import KMeans
from scipy.optimize import nnls

def extract_envi_info_from_file(file_path):
    """
    Extract metadata from ENVI header file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Header file not found: {file_path}")
    
    sample_pattern = re.compile(r'samples\s*=\s*(\d+)', re.IGNORECASE)
    band_pattern = re.compile(r'bands\s*=\s*(\d+)', re.IGNORECASE)
    line_pattern = re.compile(r'lines\s*=\s*(\d+)', re.IGNORECASE)

    with open(file_path, 'r') as file:
        header_text = file.read()

    samples_match = sample_pattern.search(header_text)
    bands_match = band_pattern.search(header_text)
    lines_match = line_pattern.search(header_text)

    samples = int(samples_match.group(1)) if samples_match else None
    bands = int(bands_match.group(1)) if bands_match else None
    lines = int(lines_match.group(1)) if lines_match else None
    
    if samples is None or bands is None or lines is None:
        raise ValueError("Could not extract all required dimensions from header file")
    
    wavelengths = re.findall(r'\d+\.\d+', header_text)
    wavelengths = [float(wavelength) for wavelength in wavelengths]
    
    if not wavelengths:
        print("Warning: No wavelengths found in header file")

    return samples, bands, lines, wavelengths

def load_raw_hyperspectral_data(raw_path, hdr_path, wavelengths, bands, lines, samples, crop_region):
    """
    Load hyperspectral data from ENVI format file
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    
    raw_data = np.empty((bands, lines, samples), dtype=np.uint16)

    raw_dataset = gdal.Open(raw_path)
    if raw_dataset is None:
        raise IOError(f"Could not open raw file: {raw_path}")

    for i in range(bands):
        band_data = raw_dataset.GetRasterBand(i+1).ReadAsArray()
        raw_data[i, :, :] = band_data

    raw_data = np.transpose(raw_data, (1, 2, 0))
    print("Shape of raw data:", raw_data.shape)
    
    if wavelengths:
        print("Start wavelength =", wavelengths[0])
        print("End wavelength =", wavelengths[-1])
        if len(wavelengths) > 1:
            print("Step wavelength =", wavelengths[1]-wavelengths[0])
    
    num_bands = len(wavelengths) if wavelengths else bands

    # Cropping
    if crop_region:
        x, y, width, height = crop_region
        if x < 0 or y < 0 or x + width > samples or y + height > lines:
            raise ValueError(f"Crop region {crop_region} exceeds image dimensions ({samples}, {lines})")
        cropped_data = raw_data[y:y+height, x:x+width, :]
        return cropped_data
    else:
        # Reshape
        raw_data = raw_data.reshape(lines, samples, num_bands)
        return raw_data

def plot_abundance_maps(abundance_maps, save_dir=None, base_filename='abundance_map', colormap='viridis'):
    """
    Plot abundance maps for each endmember
    """
    if len(abundance_maps.shape) != 3:
        raise ValueError(f"Expected 3D array, got shape {abundance_maps.shape}")
    
    num_endmembers = abundance_maps.shape[2]
    fig, axes = plt.subplots(1, num_endmembers, figsize=(5 * num_endmembers, 5))
    
    if num_endmembers == 1:
        axes = [axes]

    for i in range(num_endmembers):
        im = axes[i].imshow(abundance_maps[:, :, i], cmap=colormap, vmin=0, vmax=1)
        axes[i].set_title(f'Abundance Map - Endmember {i + 1}')
        axes[i].set_xlabel('Column')
        axes[i].set_ylabel('Row')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save individual maps
        for i in range(num_endmembers):
            fig_single = plt.figure(figsize=(8, 8))
            plt.imshow(abundance_maps[:, :, i], cmap=colormap, vmin=0, vmax=1)
            plt.axis('off')
            save_path = os.path.join(save_dir, f'{base_filename}_endmember_{i + 1}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close(fig_single)
        # Save combined figure
        save_path = os.path.join(save_dir, f'{base_filename}_combined.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)
        print(f'Abundance maps saved to {save_dir}')
    else:
        plt.tight_layout(pad=1)
        plt.show()

def Unmix(raw_path, hdr_path, EM_method='NFINDR', q=2, abun_method='FCLSU', crop_region=None, normalize=True):
    """
    Perform spectral unmixing on hyperspectral images
    """
    if q < 2:
        raise ValueError("Number of endmembers (q) must be at least 2")
    
    # 1: Extract metadata from header
    print("Extracting metadata from header file...")
    samples, bands, lines, wavelengths = extract_envi_info_from_file(hdr_path)
    print(f"Image dimensions: {samples} x {lines}, {bands} bands")
    
    # 2: Load data
    print("Loading hyperspectral data...")
    raw_data = load_raw_hyperspectral_data(raw_path, hdr_path, wavelengths, bands, lines, samples, crop_region)
    print(f"Data loaded with shape: {raw_data.shape}")
    
    # 3: Endmember extraction
    print(f"Extracting {q} endmembers using {EM_method}...")
    if EM_method == 'NFINDR':
        EM_spectra = eea.NFINDR().extract(M=raw_data, q=q, transform=None, maxit=100, normalize=normalize, mask=None)
    elif EM_method == 'PPI':
        EM_spectra = eea.PPI().extract(M=raw_data, q=q, numSkewers=1000, normalize=normalize, mask=None)
    elif EM_method == 'KMeans':
        kmeans = KMeans(n_clusters=q, random_state=0, n_init=10)
        num_rows, num_cols, num_bands = raw_data.shape
        kmeans.fit(raw_data.reshape(-1, num_bands))
        EM_kmeans = kmeans.cluster_centers_
        # Normalize endmember spectra
        if normalize:
            EM_spectra = (EM_kmeans - np.min(EM_kmeans, axis=1, keepdims=True)) / \
                        (np.max(EM_kmeans, axis=1, keepdims=True) - np.min(EM_kmeans, axis=1, keepdims=True) + 1e-10)
        else:
            EM_spectra = EM_kmeans
    else:
        raise ValueError(f"Unknown endmember extraction method: {EM_method}")
    
    print(f"Endmembers extracted. Shape: {EM_spectra.shape}")

    # 4: Abundance estimation
    print(f"Estimating abundances using {abun_method}...")
    num_rows, num_columns, num_bands = raw_data.shape
    reshaped_data = raw_data.reshape(-1, num_bands)
    
    # Normalize data for better unmixing (if using FCLSU)
    if abun_method == 'FCLSU' and normalize:
        data_mean = np.mean(reshaped_data, axis=0)
        data_std = np.std(reshaped_data, axis=0)
        reshaped_data_norm = (reshaped_data - data_mean) / (data_std + 1e-10)
        EM_spectra_norm = (EM_spectra - data_mean) / (data_std + 1e-10)
    else:
        reshaped_data_norm = reshaped_data
        EM_spectra_norm = EM_spectra
    
    if abun_method == 'FCLSU':
        # Fully Constrained Least Squares Unmixing
        # Solve: min ||Ax - y||^2 subject to x >= 0 and sum(x) = 1
        A = EM_spectra_norm.T
        num_pixels = reshaped_data_norm.shape[0]
        num_endmembers = EM_spectra_norm.shape[0]
        abundance_maps = np.zeros((num_pixels, num_endmembers))
        
        # Solve for each pixel using NNLS with sum-to-one constraint
        print("Solving FCLSU for each pixel...")
        for i in range(num_pixels):
            if (i + 1) % 10000 == 0:
                print(f"  Processing pixel {i+1}/{num_pixels}")
            # Use NNLS (Non-Negative Least Squares)
            x, _ = nnls(A, reshaped_data_norm[i])
            # Normalize to sum-to-one constraint
            if np.sum(x) > 0:
                x = x / np.sum(x)
            abundance_maps[i] = x
        
        abundance_maps = abundance_maps.reshape(num_rows, num_columns, num_endmembers)

    elif abun_method == 'NMF':
        # Non-negative Matrix Factorization (unsupervised method)
        print("Running NMF decomposition...")
        nmf = NMF(n_components=q, max_iter=600, init='nndsvd', random_state=0)
        W = nmf.fit_transform(reshaped_data)  # W = abundance maps
        H = nmf.components_  # H = endmember spectra
        abundance_maps = W.reshape(num_rows, num_columns, -1)
    else:
        raise ValueError(f"Unknown abundance method: {abun_method}")

    # Normalize abundance maps to [0, 1] for visualization
    abundance_maps = np.clip(abundance_maps, 0, 1)
    min_val = np.min(abundance_maps)
    max_val = np.max(abundance_maps)
    if max_val > min_val:
        abundance_maps = (abundance_maps - min_val) / (max_val - min_val)

    print("Abundance estimation completed.")
    print(f"Abundance maps shape: {abundance_maps.shape}")
    print(f"Abundance range: [{np.min(abundance_maps):.3f}, {np.max(abundance_maps):.3f}]")

    plot_abundance_maps(abundance_maps)
    
    return abundance_maps