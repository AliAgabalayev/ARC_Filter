import sys
from pathlib import Path
import cv2
import numpy as np
import tifffile as tiff

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
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        return img, False
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return None, False


def process_color_space(img: np.ndarray, max_val: float, is_tiff: bool) -> tuple[np.ndarray, np.ndarray]:
    """Normalization and RGB to HSV conversion"""
    # Convert to RGB double [0..1]. Reverse channels only for OpenCV (BGR layout)
    img_rgb = img.astype(np.float64) / max_val if is_tiff else img[..., ::-1].astype(np.float64)
    
    # OpenCV native conversion to HSV (Hue is [0..360] for float32, Saturation/Value are [0..1])
    img_hsv = cv2.cvtColor(img_rgb.astype(np.float32), cv2.COLOR_RGB2HSV).astype(np.float64)
    
    # Extract Value channel for the ARC filter
    return img_hsv[:, :, 2], img_hsv


def save_image(out_path: Path, img_processed: np.ndarray, is_color: bool, 
               bit_depth: int, is_tiff: bool, img_hsv: np.ndarray | None = None, max_val: float = 255.0) -> None:
    if is_color and img_hsv is not None:
        # Inject the filtered brightness back
        img_hsv[:, :, 2] = img_processed
        
        # Convert back to standard RGB double [0..1]
        img_rgb_out = cv2.cvtColor(img_hsv.astype(np.float32), cv2.COLOR_HSV2RGB).astype(np.float64)
        
        # Quantize to target bit-depth
        if bit_depth > 8 and is_tiff:
            img_out = np.round(img_rgb_out * max_val).astype(np.uint16)
            tiff.imwrite(out_path, img_out)
        else:
            img_out = np.round(img_rgb_out * 255.0).astype(np.uint8)
            if is_tiff:
                tiff.imwrite(out_path, img_out)  # Tifffile expects RGB
            else:
                cv2.imwrite(str(out_path), img_out[..., ::-1])  # OpenCV expects BGR
    else:
        # Standard Grayscale quantization pathway
        if bit_depth > 8 and is_tiff:
            img_out = np.round(img_processed * max_val).astype(np.uint16)
            tiff.imwrite(out_path, img_out)
        else:
            img_out = np.round(img_processed * 255.0).astype(np.uint8)
            tiff.imwrite(out_path, img_out) if is_tiff else cv2.imwrite(str(out_path), img_out)


def main():
    supported_extensions = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    img_types = ['Gray', 'RGB']
    
    # Setup subfolder path relative to the script location
    subfolder = Path('.') / 'Images'
    if not subfolder.is_dir():
        print('Error: Subfolder "Images" is not found')
        return

    # Gather all candidate files in the specified "Images" directory
    files = [p for p in subfolder.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions]
    
    if not files:
        print('Warning: Subfolder "Images" is empty or contains no supported files')
        return

    for file_path in files:
        if file_path.stem.endswith('_out'):
            continue
            
        out_filename = file_path.parent / f"{file_path.stem}_out{file_path.suffix}"
        
        try:
            # 1. Image ingestion
            img, is_tiff = load_image(file_path)
            if img is None:
                continue
                
            # 2. Structural dimension and bit depth evaluation
            bit_depth = get_bit_depth(img)
            max_val = float((2 ** bit_depth) - 1)
            is_color = (img.ndim == 3 and img.shape[2] == 3)
            
            # 3. Brightness channel extraction and normalization
            if is_color:
                img_normalized, img_hsv = process_color_space(img, max_val, is_tiff)
            else:
                img_gray = img[:, :, 0] if img.ndim == 3 else img
                img_normalized = img_gray.astype(np.float64) / max_val
                img_hsv = None
                
            # 4. Processing via the ARC filter
            img_processed = np.clip(ARC_filter(img_normalized), 0.0, 1.0)
            
            # 5. File reconstruction and saving
            save_image(out_filename, img_processed, is_color, bit_depth, is_tiff, img_hsv, max_val)
            
            print(f"Processed successfully: {file_path.name} ({bit_depth}-bit {img_types[int(is_color)]} -> {out_filename.name})")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

    print("\nAll images processed!")


if __name__ == '__main__':
    main()