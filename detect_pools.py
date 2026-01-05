#!/usr/bin/env python3
"""
Swimming Pool Detection CLI
Detects swimming pools in aerial images using YOLO + SAM segmentation.

Usage:
    python detect_pools.py --image path/to/image.jpg [options]

Output:
    - originalname_cord.txt: Pool boundary coordinates
    - originalname_output.extension: Image with blue outlines around detected pools
"""

import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect swimming pools in aerial images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python detect_pools.py --image aerial.jpg
    python detect_pools.py --image aerial.jpg --output-dir results/
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input aerial image"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Output directory for results (default: current directory)"
    )
    
    return parser.parse_args()


def load_models(yolo_path, sam_path, verbose=False):
    """Load YOLO and SAM models."""
    try:
        from ultralytics import YOLO, SAM
    except ImportError:
        print("Error: ultralytics package not found. Install with: pip install ultralytics")
        sys.exit(1)
    
    if verbose:
        print(f"Loading YOLO model: {yolo_path}")
    
    if not os.path.exists(yolo_path):
        print(f"Error: YOLO model not found: {yolo_path}")
        sys.exit(1)
    
    yolo_model = YOLO(yolo_path)
    
    if verbose:
        print(f"Loading SAM model: {sam_path}")
    
    if not os.path.exists(sam_path):
        print(f"Error: SAM model not found: {sam_path}")
        sys.exit(1)
    
    sam_model = SAM(sam_path)
    
    return yolo_model, sam_model


def detect_pools(img, yolo_model, conf_threshold, verbose=False):
    """Detect pools using YOLO."""
    h, w = img.shape[:2]
    
    if verbose:
        print(f"\nImage size: {w}x{h}")
        print(f"Running YOLO detection (confidence >= {conf_threshold})...")
    
    # Run YOLO detection
    yolo_results = yolo_model(img, conf=conf_threshold, verbose=False)
    yolo_res = yolo_results[0]
    
    # Extract detections
    detections = []
    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
        boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
        confidences = yolo_res.boxes.conf.cpu().numpy()
        classes = yolo_res.boxes.cls.cpu().numpy() if yolo_res.boxes.cls is not None else None
        
        for i, (box_xyxy, conf) in enumerate(zip(boxes_xyxy, confidences)):
            x1, y1, x2, y2 = box_xyxy
            cls_id = int(classes[i]) if classes is not None else 0
            
            detections.append({
                'bbox_xyxy': [x1, y1, x2, y2],
                'confidence': conf,
                'class': cls_id
            })
    
    if verbose:
        print(f"Detected {len(detections)} pool(s)")
        for i, det in enumerate(detections):
            print(f"  Pool {i+1}: confidence={det['confidence']:.3f}")
    
    return detections


def segment_pools(img, detections, sam_model, padding_ratio, use_morph, verbose=False):
    """Segment pools using SAM and extract contours."""
    h, w = img.shape[:2]
    all_contours = []
    
    if verbose:
        print(f"\nRunning SAM segmentation...")
    
    for idx, det in enumerate(detections):
        if verbose:
            print(f"  Processing pool {idx+1}/{len(detections)}...")
        
        # Get bbox with optional padding
        x1, y1, x2, y2 = det['bbox_xyxy']
        
        if padding_ratio > 0:
            bw = x2 - x1
            bh = y2 - y1
            pad_x = bw * padding_ratio
            pad_y = bh * padding_ratio
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w - 1, x2 + pad_x)
            y2 = min(h - 1, y2 + pad_y)
        
        bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
        
        # Run SAM
        sam_results = sam_model(img, bboxes=bbox_xyxy, verbose=False)
        sam_res = sam_results[0]
        
        # Extract mask
        mask = None
        try:
            if getattr(sam_res, "masks", None) is not None and getattr(sam_res.masks, "data", None) is not None:
                m = sam_res.masks.data
                mask = m[0].cpu().numpy() if hasattr(m[0], "cpu") else np.array(m[0])
        except Exception as e:
            if verbose:
                print(f"    Warning: Could not extract mask - {e}")
        
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Morphological post-processing
            if use_morph:
                # Close gaps
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close, iterations=2)
                
                # Fill holes
                contours_temp, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_temp) > 0:
                    mask_filled = np.zeros_like(mask_uint8)
                    cv2.drawContours(mask_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
                    mask_uint8 = mask_filled
                
                # Remove small artifacts
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open, iterations=1)
            
            # Extract contour
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                main_contour = max(contours, key=cv2.contourArea)
                
                # Simplify contour
                epsilon = 0.002 * cv2.arcLength(main_contour, True)
                main_contour = cv2.approxPolyDP(main_contour, epsilon, True)
                
                all_contours.append(main_contour)
                
                if verbose:
                    print(f"    ✓ Extracted {len(main_contour)} points")
            else:
                if verbose:
                    print(f"    ⚠ No contour found")
        else:
            if verbose:
                print(f"    ⚠ No mask generated")
    
    return all_contours


def save_coordinates(contours, output_path, verbose=False):
    """Save contour coordinates to text file."""
    if verbose:
        print(f"\nSaving coordinates to: {output_path}")
    
    with open(output_path, 'w') as f:
        if len(contours) == 0:
            f.write("# No pools detected\n")
        else:
            for pool_idx, contour in enumerate(contours):
                f.write(f"# Pool {pool_idx + 1} ({len(contour)} points)\n")
                for point in contour:
                    x, y = point[0]
                    f.write(f"{int(x)} {int(y)}\n")
                f.write("\n")
    
    if verbose:
        total_points = sum(len(c) for c in contours)
        print(f"  Saved {len(contours)} pool(s) with {total_points} total points")


def save_output_image(img, contours, output_path, line_thickness, verbose=False):
    """Save image with blue contours."""
    if verbose:
        print(f"\nSaving output image to: {output_path}")
    
    img_output = img.copy()
    
    # Draw all contours in BLUE (BGR format - img is already in BGR)
    BLUE = (255, 0, 0)  # BGR format for OpenCV
    
    for contour in contours:
        cv2.drawContours(img_output, [contour], -1, BLUE, thickness=line_thickness)
    
    # Save image directly (already in BGR format)
    cv2.imwrite(output_path, img_output)
    
    if verbose:
        print(f"  Drew {len(contours)} pool outline(s)")


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Configuration (hard-coded defaults)
    YOLO_MODEL = "last.pt"
    SAM_MODEL = "sam2.1_l.pt"
    CONFIDENCE_THRESHOLD = 0.4
    PADDING_RATIO = 0.0
    LINE_THICKNESS = 2
    USE_MORPH = True
    VERBOSE = True
    
    # Validate input image
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Swimming Pool Detection CLI")
    print("="*60)
    print(f"Input image: {args.image}")
    print(f"Output directory: {output_dir}")
    
    # Generate output filename from input image name
    input_path = Path(args.image)
    output_image_name = f"{input_path.stem}_output{input_path.suffix}"
    output_coord_name = f"{input_path.stem}_cord.txt"
    
    # Load image
    try:
        img = np.array(Image.open(args.image).convert("RGB"))
        # Convert RGB to BGR for OpenCV compatibility
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
    
    # Load models
    yolo_model, sam_model = load_models(YOLO_MODEL, SAM_MODEL, VERBOSE)
    
    # Detect pools
    detections = detect_pools(img, yolo_model, CONFIDENCE_THRESHOLD, VERBOSE)
    
    if len(detections) == 0:
        print("\n⚠️  No pools detected.")
        # Still create empty output files
        save_coordinates([], output_dir / output_coord_name, VERBOSE)
        save_output_image(img, [], output_dir / output_image_name, LINE_THICKNESS, VERBOSE)
        return
    
    # Segment pools
    contours = segment_pools(
        img, 
        detections, 
        sam_model, 
        PADDING_RATIO, 
        USE_MORPH,
        VERBOSE
    )
    
    # Save outputs
    save_coordinates(contours, output_dir / output_coord_name, VERBOSE)
    save_output_image(img, contours, output_dir / output_image_name, LINE_THICKNESS, VERBOSE)
    
    print("\n" + "="*60)
    print("✅ Processing complete!")
    print("="*60)
    print(f"Detected pools: {len(detections)}")
    print(f"Segmented pools: {len(contours)}")
    print(f"\nOutput files:")
    print(f"  - {output_dir / output_coord_name}")
    print(f"  - {output_dir / output_image_name}")


if __name__ == "__main__":
    main()
