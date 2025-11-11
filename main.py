#!/usr/bin/env python3
import argparse
import os
import sys
from func import Unmix

def main():
    parser = argparse.ArgumentParser(
        description='Perform spectral unmixing on rock art hyperspectral images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with NFINDR and FCLSU
  python main.py --raw data/image.raw --hdr data/image.hdr --q 3

  # Use PPI for endmember extraction
  python main.py --raw data/image.raw --hdr data/image.hdr --q 3 --em-method PPI

  # Use NMF for abundance estimation
  python main.py --raw data/image.raw --hdr data/image.hdr --q 3 --abun-method NMF

  # Crop region and save results
  python main.py --raw data/image.raw --hdr data/image.hdr --q 3 --crop 0 0 500 500 --output results/
        """
    )
    
    parser.add_argument('--raw', type=str, required=True,
                       help='Path to the raw hyperspectral image file (.raw)')
    parser.add_argument('--hdr', type=str, required=True,
                       help='Path to the ENVI header file (.hdr)')
    parser.add_argument('--q', type=int, default=2,
                       help='Number of endmembers to extract (default: 2)')
    parser.add_argument('--em-method', type=str, default='NFINDR',
                       choices=['NFINDR', 'PPI', 'KMeans'],
                       help='Endmember extraction method (default: NFINDR)')
    parser.add_argument('--abun-method', type=str, default='FCLSU',
                       choices=['FCLSU', 'NMF'],
                       help='Abundance estimation method (default: FCLSU)')
    parser.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                       help='Crop region: x, y, width, height')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for saving abundance maps')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize data during endmember extraction (default: True)')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Disable normalization')
    
    args = parser.parse_args()
    if not os.path.exists(args.raw):
        print(f"Error: Raw file not found: {args.raw}")
        sys.exit(1)
    
    if not os.path.exists(args.hdr):
        print(f"Error: Header file not found: {args.hdr}")
        sys.exit(1)
    
    crop_region = None
    if args.crop:
        if len(args.crop) != 4:
            print("Error: Crop region must have 4 values: x, y, width, height")
            sys.exit(1)
        crop_region = tuple(args.crop)
    
    print("=" * 60)
    print("Spectral Unmixing for Rock Art Hyperspectral Images")
    print("=" * 60)
    print(f"Raw file: {args.raw}")
    print(f"Header file: {args.hdr}")
    print(f"Number of endmembers: {args.q}")
    print(f"Endmember extraction method: {args.em_method}")
    print(f"Abundance estimation method: {args.abun_method}")
    if crop_region:
        print(f"Crop region: {crop_region}")
    print(f"Normalize: {args.normalize}")
    print("=" * 60)
    
    try:
        abundance_maps = Unmix(
            raw_path=args.raw,
            hdr_path=args.hdr,
            EM_method=args.em_method,
            q=args.q,
            abun_method=args.abun_method,
            crop_region=crop_region,
            normalize=args.normalize,
            save_dir=args.output
        )
        
        if args.output:
            print(f"\nResults saved to: {args.output}")
        
        print("\nSpectral unmixing completed successfully!")
        
    except Exception as e:
        print(f"\nError during spectral unmixing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

