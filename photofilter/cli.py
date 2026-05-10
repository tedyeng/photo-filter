import argparse
import sys
from pathlib import Path
from .core import process_folder
from .utils import load_config

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='photofilter',
        description='Select the best‑composed and sharpest photo from a burst.'
    )
    parser.add_argument('-i', '--input', type=Path, required=True,
                        help='Path to folder containing burst images (JPG + RAW).')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Destination folder for the selected best photo.')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Minimum composite score to accept a photo.')
    parser.add_argument('--env-file', type=Path,
                        help='Path to .env config file for weighting, GPU, etc.')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU acceleration if CUDA is available.')
    return parser

def main(argv: list = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.env_file)
    use_gpu = args.gpu and config.get('gpu_enabled', True)
    # Call core processing (implementation will be added later)
    process_folder(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
        config=config,
        use_gpu=use_gpu,
    )
    return 0

if __name__ == '__main__':
    sys.exit(main())
