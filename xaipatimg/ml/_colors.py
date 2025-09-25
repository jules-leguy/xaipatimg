import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_transparent_colormap():
    # Define the colors (RGB values from 0 to 1)
    red_color = np.array([220, 53, 69]) / 255.0    # #dc3545
    green_color = np.array([25, 135, 84]) / 255.0  # #198754

    # Create colormap dictionary
    cdict = {
        'red': [
            (0.0, red_color[0], red_color[0]),    # Start with red
            (0.5, 0.8, 0.8),                      # Middle: lighter red (for transparency effect)
            (1.0, green_color[0], green_color[0])  # End with green
        ],
        'green': [
            (0.0, red_color[1], red_color[1]),    # Start with red's green component
            (0.5, 0.8, 0.8),                      # Middle: lighter
            (1.0, green_color[1], green_color[1])  # End with green's green component
        ],
        'blue': [
            (0.0, red_color[2], red_color[2]),    # Start with red's blue component
            (0.5, 0.8, 0.8),                      # Middle: lighter
            (1.0, green_color[2], green_color[2])  # End with green's blue component
        ],
        'alpha': [
            (0.0, 1.0, 1.0),   # Opaque at start
            (0.5, 0.3, 0.3),   # Transparent in middle (30% opacity)
            (1.0, 1.0, 1.0)    # Opaque at end
        ]
    }

    return LinearSegmentedColormap('RedGreenTransparent', cdict)

# Create the colormap
red_transparent_green = create_transparent_colormap()