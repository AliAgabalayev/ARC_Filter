import sys
from pathlib import Path
import cv2
import numpy as np
import tifffile as tiff
from concurrent.futures import ProcessPoolExecutor

try:
    from arc_filter import ARC_filter
except ImportError:
    def ARC_filter(img): return img 


def get_bit_depth(img: np.ndarray) -> int:
    """Bit depth detection using NumPy dtypes"""
    match img.dtype:
        case np.uint16:
            return 16
        case np.int32 | np.uint32:
            return 32
        case _:
            return 8


def load_image(file_path: Path) -> tuple[np.ndarray | None, bool]:
    is_tiff = file_path.suffix.lower() in {'.tiff', '.tif'}
    try:
        if is_tiff:
            return tiff.imread(file_path), True
        return cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED), False
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None, False


def process_color_space(img: np.ndarray, max_val: float, is_tiff: bool) -> tuple[np.ndarray, np.ndarray]:
    """Normalization and RGB to HSV conversion"""
    img_3ch = img[..., :3]
    if is_tiff:
        img_rgb = img_3ch.astype(np.float32) / max_val
    else:
        img_rgb = img_3ch[..., ::-1].astype(np.float32) / max_val
    
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return img_hsv[:, :, 2].astype(np.float64), img_hsv


def save_image(out_path: Path, img_processed: np.ndarray, is_color: bool, 
               bit_depth: int, is_tiff: bool, img_hsv: np.ndarray | None = None, max_val: float = 255.0) -> None:
    if is_color and img_hsv is not None:
        img_hsv[:, :, 2] = img_processed.astype(np.float32)
        img_rgb_out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        if bit_depth > 8 and is_tiff:
            img_out = np.round(img_rgb_out * max_val).astype(np.uint16)
            tiff.imwrite(out_path, img_out)
        else:
            img_out = np.round(img_rgb_out * max_val).astype(np.uint8)
            if is_tiff:
                tiff.imwrite(out_path, img_out)
            else:
                cv2.imwrite(str(out_path), img_out[..., ::-1])
    else:
        if bit_depth > 8 and is_tiff:
            img_out = np.round(img_processed * max_val).astype(np.uint16)
            tiff.imwrite(out_path, img_out)
        else:
            img_out = np.round(img_processed * max_val).astype(np.uint8)
            tiff.imwrite(out_path, img_out) if is_tiff else cv2.imwrite(str(out_path), img_out)


def process_single_file(file_path: Path) -> str:
    """Processes a single image file. Returns a status message string."""
    img_types = ['Gray', 'RGB']
    out_filename = file_path.parent / f"{file_path.stem}_out{file_path.suffix}"
    
    try:
        # 1. Image ingestion
        img, is_tiff = load_image(file_path)
        if img is None:
            return f"Skipped (Error loading): {file_path.name}"
            
        # 2. Structural dimension and bit depth evaluation
        bit_depth = get_bit_depth(img)
        max_val = float((2 ** bit_depth) - 1)
        is_color = (img.ndim == 3 and img.shape[2] in (3, 4))
        
        # 3. Brightness channel extraction and normalization
        if is_color:
            img_normalized, img_hsv = process_color_space(img, max_val, is_tiff)
        else:
            if img.ndim == 3:
                if img.shape[2] == 4:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if not is_tiff else cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if not is_tiff else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img
            img_normalized = img_gray.astype(np.float64) / max_val
            img_hsv = None
            
        # 4. Processing via the ARC filter
        img_processed = np.clip(ARC_filter(img_normalized), 0.0, 1.0)
        
        # 5. File reconstruction and saving
        save_image(out_filename, img_processed, is_color, bit_depth, is_tiff, img_hsv, max_val)
        
        return f"Processed successfully: {file_path.name} ({bit_depth}-bit {img_types[int(is_color)]} -> {out_filename.name})"
        
    except Exception as e:
        return f"Error processing {file_path.name}: {str(e)}"


def main():
    supported_extensions = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    # Setup subfolder path relative to the script location
    subfolder = Path('.') / 'Images'
    if not subfolder.is_dir():
        print('Error: Subfolder "Images" is not found')
        return

    # Gather all candidate files in the specified "Images" directory
    files = [p for p in subfolder.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions]
    
    # Filter out output files from previous runs to prevent infinite loops
    files = [f for f in files if not f.stem.endswith('_out')]
    
    if not files:
        print('Warning: Subfolder "Images" is empty or contains no supported files')
        return

    print(f"Starting parallel processing of {len(files)} images...")
    
    # Launch process pool execution
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_single_file, files)
        
        for target_msg in results:
            print(target_msg)

    print("\nAll images processed parallelly!")


if __name__ == '__main__':
    main()