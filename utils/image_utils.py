from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FONT = str(_REPO_ROOT / "Super Moods.ttf")
_LLAVA_IMAGES_DIR = _REPO_ROOT / "CS_DJ_main" / "llava_images"


def text_to_art_image_pil(text, font_path='arial.ttf', font_size=100, text_color=(0, 0, 0), bg_color=(255, 255, 255), image_width=500):
    """Modified version of text_to_art_image that returns PIL Image instead of saving to file"""
    image = Image.new('RGB', (image_width, 300), color=bg_color)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font not found at {font_path}, trying arial.ttf...")
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            print("Arial font not found, using default font.")
            font = ImageFont.load_default()

    lines = []
    words = text.split()
    current_line = ""
    
    # Allow text to use up to 90% of image width for better utilization
    max_width = int(image_width * 0.9)

    for word in words:
        test_line = f"{current_line} {word}".strip()
        # Use textbbox instead of deprecated textsize
        try:
            text_bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            # Fallback for older PIL versions
            text_width = draw.textsize(test_line, font=font)[0]
        
        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:  # Only add non-empty lines
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Reduce line spacing for more compact text
    line_spacing = max(5, font_size // 10)  # Minimum 5px spacing, or 10% of font size
    total_height = len(lines) * font_size + (len(lines) - 1) * line_spacing + 20  # Add padding
    image = Image.new('RGB', (image_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    y_offset = 10  # Start with some padding
    for line in lines:
        try:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            # Fallback for older PIL versions
            text_width = draw.textsize(line, font=font)[0]
        
        text_x = (image.width - text_width) // 2
        draw.text((text_x, y_offset), line, font=font, fill=text_color)
        y_offset += font_size + line_spacing 

    return image

def concatenate_images_with_distraction(distraction_image_paths, diffusion_image, typo_image, images_per_row=3, target_size=(300, 300), fill_color=(255, 255, 255), font_size=20, rotation_angle=0):
    images = []
    
    try:
        font = ImageFont.truetype(_DEFAULT_FONT, font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

    # Process up to 7 distraction images (positions 1-7)
    # If fewer than 7 images are available, we'll just use what we have
    for idx, img_path in enumerate(distraction_image_paths[:7]):
        if img_path.startswith('./llava_images/'):
            filename = img_path.replace('./llava_images/', '')
            img_path = str(_LLAVA_IMAGES_DIR / filename)
        img = Image.open(img_path)
        img.thumbnail(target_size) 

        diagonal = int((target_size[0]**2 + target_size[1]**2)**0.5)
        expanded_img = Image.new('RGB', (diagonal, diagonal), fill_color)
        img_x, img_y = img.size
        paste_x = (diagonal - img_x) // 2
        paste_y = (diagonal - img_y) // 2
        expanded_img.paste(img, (paste_x, paste_y))

        rotated_img = expanded_img.rotate(rotation_angle, expand=True, fillcolor=fill_color)

        final_padded_img = Image.new('RGB', target_size, fill_color)
        rotated_img_x, rotated_img_y = rotated_img.size
        final_paste_x = (target_size[0] - rotated_img_x) // 2
        final_paste_y = (target_size[1] - rotated_img_y) // 2
        final_padded_img.paste(rotated_img, (final_paste_x, final_paste_y))

        final_img = Image.new('RGB', (target_size[0], target_size[1] + font_size + 10), fill_color)
        final_img.paste(final_padded_img, (0, 0))

        draw = ImageDraw.Draw(final_img)
        text = str(idx + 1)
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = (target_size[0] - text_width) // 2
        text_y = target_size[1] + (font_size // 2)
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))  

        images.append(final_img)

    # Process diffusion image (position 8)
    if diffusion_image:
        diffusion_img = diffusion_image.copy()
        diffusion_img.thumbnail(target_size)
        
        final_padded_img = Image.new('RGB', target_size, fill_color)
        img_x, img_y = diffusion_img.size
        paste_x = (target_size[0] - img_x) // 2
        paste_y = (target_size[1] - img_y) // 2
        final_padded_img.paste(diffusion_img, (paste_x, paste_y))

        final_img = Image.new('RGB', (target_size[0], target_size[1] + font_size + 10), fill_color)
        final_img.paste(final_padded_img, (0, 0))

        draw = ImageDraw.Draw(final_img)
        text = "8"
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = (target_size[0] - text_width) // 2
        text_y = target_size[1] + (font_size // 2)
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

        images.append(final_img)

    # Process typo image (position 9)
    if typo_image:
        typo_img = typo_image.copy()
        typo_img.thumbnail(target_size)
        
        final_padded_img = Image.new('RGB', target_size, fill_color)
        img_x, img_y = typo_img.size
        paste_x = (target_size[0] - img_x) // 2
        paste_y = (target_size[1] - img_y) // 2
        final_padded_img.paste(typo_img, (paste_x, paste_y))

        final_img = Image.new('RGB', (target_size[0], target_size[1] + font_size + 10), fill_color)
        final_img.paste(final_padded_img, (0, 0))

        draw = ImageDraw.Draw(final_img)
        text = "9"
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)
        
        text_x = (target_size[0] - text_width) // 2
        text_y = target_size[1] + (font_size // 2)
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

        images.append(final_img)

    width, height = target_size[0], target_size[1] + font_size + 10

    rows = (len(images) + images_per_row - 1) // images_per_row

    total_width = width * images_per_row
    total_height = height * rows
    new_image = Image.new('RGB', (total_width, total_height), fill_color)

    for index, img in enumerate(images):
        x_offset = (index % images_per_row) * width
        y_offset = (index // images_per_row) * height
        new_image.paste(img, (x_offset, y_offset))

    return new_image
