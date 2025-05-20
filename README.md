#Heatmap Generator

A desktop application for creating customizable geographic heatmaps from KML and GeoJSON data.

- Python 3.7 or higher
- Dependencies (automatically installed):
  - geopandas, folium, pandas, matplotlib, numpy
  - pillow, shapely, reportlab, customtkinter
  - pyperclip, lxml, tqdm, colorama

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-heatmap-generator.git
cd enhanced-heatmap-generator

# Install dependencies
pip install -r requirements.txt

# Run the application
python heatmap_generatorV2.py
```

## Basic Usage

1. create a.bat file and launch the application
2. Select KML data file and GeoJSON boundary file
3. Click "Load Files" to process data
4. Choose heatmap style, border style, and map style
5. Click "Create Heatmap" to generate and save the result

## Command Line Interface

```bash
python heatmap_generatorV2.py --kml data.kml --geojson boundary.geojson --style default --output heatmap.html
```

Options:
- `--kml`: Path to KML file (required)
- `--geojson`: Path to GeoJSON file (required)
- `--style`: Heatmap style
- `--border`: Border style
- `--output`: Output file path
- `--no-open`: Don't open the map in browser

## Advanced Features

- **Custom Style Editor**: Create and save your own heatmap styles (buggy)
- **Batch Processing**: Process multiple KML/GeoJSON pairs at once

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Made by BennoGHG
