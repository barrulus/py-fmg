#!/usr/bin/env python3
"""
Convert base64 string to image file.
"""

import base64
import sys
from pathlib import Path


def convert_base64_to_image(base64_string, output_filename="output_image.png"):
    """
    Convert a base64 string to an image file.

    Args:
        base64_string: The base64 encoded image data
        output_filename: The filename to save the image as
    """
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            # Handle data URLs like "data:image/png;base64,..."
            header, base64_string = base64_string.split(",", 1)

            # Try to determine file extension from header
            if "image/jpeg" in header or "image/jpg" in header:
                if not output_filename.endswith((".jpg", ".jpeg")):
                    output_filename = output_filename.rsplit(".", 1)[0] + ".jpg"
            elif "image/png" in header:
                if not output_filename.endswith(".png"):
                    output_filename = output_filename.rsplit(".", 1)[0] + ".png"
            elif "image/gif" in header:
                if not output_filename.endswith(".gif"):
                    output_filename = output_filename.rsplit(".", 1)[0] + ".gif"
            elif "image/webp" in header:
                if not output_filename.endswith(".webp"):
                    output_filename = output_filename.rsplit(".", 1)[0] + ".webp"

        # Decode base64 string
        image_data = base64.b64decode(base64_string)

        # Write to file
        with open(output_filename, "wb") as f:
            f.write(image_data)

        print(f"Image saved successfully as: {output_filename}")
        print(f"File size: {len(image_data):,} bytes")

        # Try to get image dimensions using PIL if available
        try:
            from PIL import Image

            with Image.open(output_filename) as img:
                print(f"Image dimensions: {img.width} x {img.height} pixels")
                print(f"Image format: {img.format}")
        except ImportError:
            print("(Install Pillow to see image dimensions: pip install Pillow)")

        return output_filename

    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None


def main():
    """Main function to handle command line usage."""
    if len(sys.argv) < 2:
        print(
            "Usage: python convert_base64_to_image.py <base64_string_or_file> [output_filename]"
        )
        print("\nExample:")
        print(
            "  python convert_base64_to_image.py 'data:image/png;base64,iVBORw0KG...' output.png"
        )
        print("  python convert_base64_to_image.py base64.txt output.png")
        sys.exit(1)

    base64_input = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "output_image.png"

    # Check if input is a file
    if Path(base64_input).exists():
        print(f"Reading base64 from file: {base64_input}")
        with open(base64_input, "r") as f:
            base64_string = f.read().strip()
    else:
        base64_string = base64_input

    convert_base64_to_image(base64_string, output_filename)


if __name__ == "__main__":
    main()
