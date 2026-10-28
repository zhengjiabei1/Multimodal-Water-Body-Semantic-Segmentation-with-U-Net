"""Band layout and input configurations used throughout the project."""

# Channel order produced by the Google Earth Engine export script.
# The first five channels are Sentinel-2 surface reflectance bands, followed by
# two water indices computed from those bands.
BAND_NAMES = ("B2", "B3", "B4", "B8", "B11", "NDWI", "MNDWI")

# Index of each channel in the exported stack.
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}

# Torchvision-style RGB order: B4 (red), B3 (green), B2 (blue).
RGB_INDEX = (BAND_INDEX["B4"], BAND_INDEX["B3"], BAND_INDEX["B2"])

INPUT_SCHEMES = {
    "A": {
        "name": "RGB",
        "channels": RGB_INDEX,
        "description": "B4, B3, B2",
    },
    "B": {
        "name": "RGB + NIR",
        "channels": RGB_INDEX + (BAND_INDEX["B8"],),
        "description": "B4, B3, B2, B8",
    },
    "C": {
        "name": "RGB + NIR + MNDWI",
        "channels": RGB_INDEX + (BAND_INDEX["B8"], BAND_INDEX["MNDWI"]),
        "description": "B4, B3, B2, B8, MNDWI",
    },
    "D": {
        "name": "All channels",
        "channels": RGB_INDEX
        + (BAND_INDEX["B8"], BAND_INDEX["NDWI"], BAND_INDEX["MNDWI"]),
        "description": "B4, B3, B2, B8, NDWI, MNDWI",
    },
}

SCHEME_LABELS = {
    "A": "A: RGB",
    "B": "B: RGB + NIR",
    "C": "C: RGB + NIR + MNDWI",
    "D": "D: All channels",
}
