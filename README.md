# Spectral Unmixing for Rock Art Hyperspectral Images

This project provides tools for performing spectral unmixing on hyperspectral images of rock art. Spectral unmixing is a technique used to decompose mixed pixels into their constituent endmembers (pure materials) and their corresponding abundance fractions.

## Features

- **Endmember Extraction Methods:**
  - NFINDR (N-FINDR) - Finds endmembers by maximizing simplex volume
  - PPI (Pixel Purity Index) - Identifies pure pixels using skewers
  - KMeans - Clustering-based endmember extraction

- **Abundance Estimation Methods:**
  - FCLSU (Fully Constrained Least Squares Unmixing) - Linear unmixing with sum-to-one and non-negativity constraints
  - NMF (Non-negative Matrix Factorization) - Unsupervised decomposition

- **Data Support:**
  - ENVI format hyperspectral images (.raw/.hdr)
  - Customizable cropping regions
  - Automatic wavelength extraction from headers

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** GDAL installation might require additional system dependencies. On Ubuntu/Debian:
```bash
sudo apt-get install gdal-bin libgdal-dev
pip install GDAL
```

## Usage

### Command Line Interface

Basic usage:
```bash
python main.py --raw path/to/image.raw --hdr path/to/image.hdr --q 3
```

With custom endmember extraction:
```bash
python main.py --raw image.raw --hdr image.hdr --q 3 --em-method PPI
```

With NMF abundance estimation:
```bash
python main.py --raw image.raw --hdr image.hdr --q 3 --abun-method NMF
```

With cropping and output directory:
```bash
python main.py --raw image.raw --hdr image.hdr --q 3 --crop 0 0 500 500 --output results/
```

### Python API

```python
from func import Unmix

# Perform spectral unmixing
abundance_maps = Unmix(
    raw_path='path/to/image.raw',
    hdr_path='path/to/image.hdr',
    EM_method='NFINDR',  # or 'PPI', 'KMeans'
    q=3,  # number of endmembers
    abun_method='FCLSU',  # or 'NMF'
    crop_region=(0, 0, 500, 500),  # optional: (x, y, width, height)
    normalize=True
)
```

## Command Line Arguments

- `--raw`: Path to the raw hyperspectral image file (.raw) **[required]**
- `--hdr`: Path to the ENVI header file (.hdr) **[required]**
- `--q`: Number of endmembers to extract (default: 2)
- `--em-method`: Endmember extraction method - NFINDR, PPI, or KMeans (default: NFINDR)
- `--abun-method`: Abundance estimation method - FCLSU or NMF (default: FCLSU)
- `--crop`: Crop region as x y width height (optional)
- `--output`: Output directory for saving abundance maps (optional)
- `--normalize` / `--no-normalize`: Enable/disable normalization (default: enabled)

## Project Structure

```
rock art/
├── func.py              # Core functions for spectral unmixing
├── main.py              # Command-line interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How It Works

1. **Data Loading**: Reads ENVI format hyperspectral images and extracts metadata (dimensions, wavelengths) from header files.

2. **Endmember Extraction**: Identifies pure spectral signatures (endmembers) representing different materials in the scene:
   - **NFINDR**: Finds endmembers by maximizing the volume of the simplex formed by the endmembers
   - **PPI**: Uses skewers to identify pixels that are pure in at least one direction
   - **KMeans**: Clusters pixels and uses cluster centers as endmembers

3. **Abundance Estimation**: Calculates the fractional abundance of each endmember in each pixel:
   - **FCLSU**: Solves a constrained least squares problem where abundances sum to 1 and are non-negative
   - **NMF**: Factorizes the data matrix into non-negative abundance and endmember matrices

4. **Visualization**: Generates abundance maps showing the spatial distribution of each endmember.

## Output

The script generates abundance maps for each endmember, showing:
- Spatial distribution of each material
- Fractional abundance values (0-1, normalized)
- Visual representation as grayscale images

## Notes

- For rock art analysis, typically 2-5 endmembers are used (representing rock, paint, weathering, etc.)
- NFINDR is generally recommended for endmember extraction when the number of endmembers is known
- FCLSU is faster and more interpretable, while NMF can capture non-linear mixing effects
- The code assumes hyperspectral data is in standard ENVI format

## Troubleshooting

- **GDAL errors**: Ensure GDAL is properly installed with system dependencies
- **Memory issues**: Use cropping to process smaller regions of large images
- **Poor results**: Try different endmember extraction methods or adjust the number of endmembers (q)

## References

Keshava, N. (2003). A Survey of Spectral Unmixing Algorithms. *Lincoln Laboratory Journal*, 14, 55-78.

Djelil, Y. F. (2024). *Spectral Unmixing of Rock Art Hyperspectral Images* [Master's thesis, University of Eastern Finland]. UEF eRepository. https://erepo.uef.fi/items/945f5642-6494-4af9-a0c2-3d9f16266222

Therien, C. (2013-2018). PySptools: Endmembers Extraction Algorithms (EEA). https://pysptools.sourceforge.io/eea.html

