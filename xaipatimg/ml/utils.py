from PIL import Image, ImageChops, ImageOps, ImageDraw, ImageFont
import textwrap
import os

def nhwc_to_nchw(x):
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x

def nchw_to_nhwc(x):
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x

def crop_white_border(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def add_margin(img, top, right, bottom, left, color):
    width, height = img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(img.mode, (new_width, new_height), color)
    result.paste(img, (left, top))
    return result

def add_border(img):
    # Add a 1px border around the image
    return ImageOps.expand(img, border=1, fill="#f2f2f2")


def generate_llm_text_image(
    text,
    width=800,
    height=400,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    font_size=40,
    margin=20,
    line_spacing=10,
    yes_pred_img_path=None,
    no_pred_img_path=None,
    pred_img_scale=1.0,  # scale for Yes/No image size
):
    """
    Generate an image with given text, auto-wrapping to fit the width.
    The text block is vertically centered but left-aligned horizontally.
    If the text does not fit, the font size is decreased iteratively.
    'pred_img_scale' controls how large the Yes/No images appear relative to text height.
    """

    # Load replacement images if given
    yes_img = Image.open(yes_pred_img_path).convert("RGBA") if yes_pred_img_path else None
    no_img = Image.open(no_pred_img_path).convert("RGBA") if no_pred_img_path else None

    # Create base image
    image = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Load font (default fallback)
    def get_font(size):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size
            )
        except IOError:
            print("Warning: LiberationSans-Regular.ttf not found. Falling back to default font.")
            return ImageFont.load_default()

    def wrap_text_with_font(fnt):
        """Wrap text to fit width with given font."""
        wrapped_lines = []
        for paragraph in text.split('\n'):
            words = paragraph.split()
            if not words:
                wrapped_lines.append("")
                continue
            current_line = ""
            for word in words:
                test_line = (current_line + " " + word).strip()
                line_width = draw.textlength(test_line, font=fnt)
                if line_width <= (width - 2 * margin):
                    current_line = test_line
                else:
                    wrapped_lines.append(current_line)
                    current_line = word
            if current_line:
                wrapped_lines.append(current_line)
        return wrapped_lines

    # --- Font size adjustment loop ---
    current_size = font_size
    min_size = 10
    while current_size >= min_size:
        font = get_font(current_size)
        lines = wrap_text_with_font(font)
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
        total_text_height = len(lines) * (line_height + line_spacing) - line_spacing

        # Check if all lines fit horizontally & vertically
        fits_horizontally = all(
            draw.textlength(line, font=font) <= (width - 2 * margin)
            for line in lines
        )
        fits_vertically = total_text_height <= (height - 2 * margin)

        if fits_horizontally and fits_vertically:
            break
        current_size -= 2  # decrease gradually

    # Compute vertical centering
    y_start = (height - total_text_height) // 2
    y = y_start

    # Draw text line by line
    for line in lines:
        x = margin
        words = line.split(" ")
        for word in words:
            clean_word = word.strip(",.!?;:")
            if clean_word == "|YES|" and yes_img:
                img_h = int(line_height * pred_img_scale)
                aspect_ratio = yes_img.width / yes_img.height
                img_w = int(img_h * aspect_ratio)
                resized = yes_img.resize((img_w, img_h))
                image.paste(resized, (int(x), y), resized)
                x += img_w + 5
            elif clean_word == "|NO|" and no_img:
                img_h = int(line_height * pred_img_scale)
                aspect_ratio = no_img.width / no_img.height
                img_w = int(img_h * aspect_ratio)
                resized = no_img.resize((img_w, img_h))
                image.paste(resized, (int(x), y), resized)
                x += img_w + 5
            else:
                draw.text((int(x), y), word, fill=text_color, font=font)
                word_width = draw.textlength(word + " ", font=font)
                x += word_width
        y += line_height + line_spacing

    return image