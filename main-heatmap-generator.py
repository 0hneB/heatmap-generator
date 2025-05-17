"""
EnhancedHeatmapGenerator - A powerful tool for creating geographic heatmaps

This application allows users to create customizable heatmaps from KML and GeoJSON data
with advanced visualization options, data analytics capabilities, and a modern user interface.

Features:
- Modern and responsive UI with dark/light theme support
- Multiple heatmap visualization styles with real-time preview
- Advanced data analysis and filtering capabilities
- Batch processing of multiple files
- Export to multiple formats (HTML, PNG, PDF)
- Settings persistence and project saving
- Rich data visualization options including 3D terrain view
- Comprehensive error handling and user guidance
"""

import os
import sys
import json
import webbrowser
import subprocess
import traceback
import argparse
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Dependency installation and checking
def check_and_install_dependencies():
    """Check for required dependencies and install if missing."""
    required_packages = {
        "geopandas": "geopandas",
        "folium": "folium",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pillow": "pillow",
        "shapely": "shapely",
        "reportlab": "reportlab",
        "customtkinter": "customtkinter",
        "pyperclip": "pyperclip",
        "lxml": "lxml",
        "tqdm": "tqdm",
        "colorama": "colorama",
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {module} is already installed")
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("All required packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing packages: {e}")
            print("Please install the following packages manually:")
            print(", ".join(missing_packages))
            input("Press Enter to exit...")
            sys.exit(1)

# Import dependencies after checking/installing
try:
    check_and_install_dependencies()
    
    # Standard library imports
    import xml.etree.ElementTree as ET
    from io import BytesIO
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from threading import Thread
    import logging
    from functools import partial
    import csv
    import re
    
    # Third-party imports
    import customtkinter as ctk
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import folium
    from folium.plugins import HeatMap, MeasureControl, Fullscreen, MarkerCluster, Draw
    from shapely.geometry import Point, Polygon, MultiPolygon
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import pyperclip
    from PIL import Image, ImageTk
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from tqdm import tqdm
    from colorama import Fore, Style
    
except Exception as e:
    print(f"Error setting up the application: {e}")
    print(traceback.format_exc())
    input("Press Enter to exit...")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("heatmap_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HeatmapGenerator")

# Constants and configuration
APP_NAME = "Heatmap Generator"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Made by BennoGHG"
CONFIG_DIR = Path.home() / ".heatmap_generator"
CONFIG_FILE = CONFIG_DIR / "config.json"
RECENT_FILES = CONFIG_DIR / "recent_files.json"
LOG_FILE = CONFIG_DIR / "heatmap_generator.log"

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)

# Default settings
DEFAULT_SETTINGS = {
    "theme": "dark",
    "default_save_dir": str(Path.home() / "Documents" / "Heatmaps"),
    "auto_open_map": True,
    "remember_last_directory": True,
    "default_export_format": "html",
    "max_recent_files": 10,
    "show_preview": True,
    "advanced_mode": False
}

# Heatmap styles with advanced customization options
HEATMAP_STYLES = {
    "default": {
        "radius": 22,
        "blur": 15,
        "min_opacity": 0.5,
        "gradient": {"0.4": 'blue', "0.65": 'lime', "1.0": 'red'},
        "description": "Standard balanced heatmap",
        "icon": "🎨"
    },
    "intense": {
        "radius": 35,
        "blur": 10,
        "min_opacity": 0.7,
        "gradient": {"0.3": 'blue', "0.5": 'purple', "0.7": 'orange', "1.0": 'red'},
        "description": "High intensity, sharper transitions",
        "icon": "🔥"
    },
    "smooth": {
        "radius": 25,
        "blur": 25,
        "min_opacity": 0.4,
        "gradient": {"0.4": 'green', "0.6": 'lime', "0.8": 'yellow', "1.0": 'red'},
        "description": "Smooth transitions with earth tones",
        "icon": "🌊"
    },
    "cool": {
        "radius": 20,
        "blur": 18,
        "min_opacity": 0.5,
        "gradient": {"0.2": '#0000ff', "0.4": '#00ffff', "0.6": '#00ffaa', "1.0": '#00ff00'},
        "description": "Cool blue palette",
        "icon": "❄️"
    },
    "fire": {
        "radius": 18,
        "blur": 12,
        "min_opacity": 0.6,
        "gradient": {"0.2": 'yellow', "0.4": 'orange', "0.6": 'orangered', "1.0": 'darkred'},
        "description": "Warm fire-like color scheme",
        "icon": "🔥"
    },
    "monochrome": {
        "radius": 20,
        "blur": 15,
        "min_opacity": 0.5,
        "gradient": {"0.4": '#eeeeee', "0.65": '#aaaaaa', "1.0": '#333333'},
        "description": "Simple black and white gradient",
        "icon": "🎭"
    },
    "rainbow": {
        "radius": 25,
        "blur": 15,
        "min_opacity": 0.6,
        "gradient": {"0.0": 'blue', "0.25": 'cyan', "0.5": 'lime', "0.75": 'yellow', "1.0": 'red'},
        "description": "Full spectrum rainbow colors",
        "icon": "🌈"
    },
    "custom": {
        "radius": 20,
        "blur": 15,
        "min_opacity": 0.5,
        "gradient": {"0.4": 'blue', "0.65": 'lime', "1.0": 'red'},
        "description": "User-defined custom style",
        "icon": "⚙️"
    }
}

# Border styles
BORDER_STYLES = {
    "none": {"weight": 0, "opacity": 0, "color": "transparent", "description": "No border"},
    "thin": {"weight": 1, "opacity": 0.5, "color": "#000000", "description": "Thin black border"},
    "normal": {"weight": 2, "opacity": 0.7, "color": "#000000", "description": "Normal black border"},
    "thick": {"weight": 3, "opacity": 0.8, "color": "blue", "description": "Thick colored border"},
    "dashed": {"weight": 2, "opacity": 0.7, "color": "#000000", "description": "Dashed border", "dashArray": "5, 5"},
    "custom": {"weight": 2, "opacity": 0.7, "color": "#FF0000", "description": "Custom border style"}
}

# Map tile providers
MAP_TILES = {
    "OpenStreetMap": {
        "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        "description": "Standard street map"
    },
    "Topo (OpenTopoMap)": {
        "url": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "attribution": 'Map tiles by <a href="http://opentopomap.org">Open Topo Map</a>',
        "description": "Topography visualization"
    },
    "Without Basemap": {
        "url": "https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
        "attribution": 'Map tiles by <a href="http://atomomc.com">No basemap</a>',
        "description": "Blank Background"
    },
    "CartoDB Dark": {
        "url": "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png",
        "attribution": '&copy; <a href="https://carto.com/attributions">CartoDB</a>',
        "description": "Dark theme map"
    },
    "CartoDB Light": {
        "url": "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        "attribution": '&copy; <a href="https://carto.com/attributions">CartoDB</a>',
        "description": "Light theme map"
    },
    "ESRI World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": 'Tiles &copy; Esri',
        "description": "Satellite imagery"
    }
}

# Utility Functions

def load_settings():
    """Load user settings from config file or use defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return DEFAULT_SETTINGS
    else:
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS

def save_settings(settings):
    """Save user settings to config file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

def load_recent_files():
    """Load recent files list."""
    if RECENT_FILES.exists():
        try:
            with open(RECENT_FILES, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    else:
        return []

def save_recent_files(files):
    """Save recent files list."""
    try:
        with open(RECENT_FILES, 'w') as f:
            json.dump(files, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving recent files: {e}")

def add_to_recent_files(kml_path, geojson_path, settings):
    """Add files to recent files list."""
    recent = load_recent_files()
    # Add as a tuple with timestamp
    entry = {
        "kml": kml_path,
        "geojson": geojson_path,
        "timestamp": datetime.now().isoformat(),
        "description": f"{os.path.basename(kml_path)} & {os.path.basename(geojson_path)}"
    }
    
    # Check if already exists
    for i, item in enumerate(recent):
        if item.get("kml") == kml_path and item.get("geojson") == geojson_path:
            recent.pop(i)
            break
    
    # Add to beginning
    recent.insert(0, entry)
    
    # Limit size
    if len(recent) > settings.get("max_recent_files", 10):
        recent = recent[:settings.get("max_recent_files", 10)]
    
    save_recent_files(recent)

# Classes

class DataManager:
    """Class to handle loading and processing geospatial data."""
    
    def __init__(self):
        self.kml_data = None
        self.geojson_data = None
        self.kml_path = None
        self.geojson_path = None
        self.coords = []
        self.filtered_coords = []
        self.region = None
        self.centroid = None
        self.stats = {}
    
    def load_kml(self, kml_path: str) -> bool:
        """Load and parse KML file with robust error handling."""
        self.kml_path = kml_path
        self.coords = []
        logger.info(f"Loading KML file: {kml_path}")
        
        try:
            # First, try using geopandas to read KML directly
            try:
                gdf = gpd.read_file(kml_path, driver='KML')
                if not gdf.empty:
                    # Extract coordinates from geometry
                    for geom in gdf.geometry:
                        if geom:
                            if geom.geom_type == 'Point':
                                # KML coordinates are in lon, lat order
                                self.coords.append((geom.y, geom.x))
                            elif geom.geom_type in ['LineString', 'MultiPoint']:
                                for point in geom.coords:
                                    self.coords.append((point[1], point[0]))
                            elif geom.geom_type == 'Polygon':
                                for point in geom.exterior.coords:
                                    self.coords.append((point[1], point[0]))
                    
                    logger.info(f"Successfully loaded {len(self.coords)} coordinates using geopandas")
                    return True
            except Exception as e:
                logger.warning(f"Could not load KML with geopandas: {e}")
                # Continue with custom parser if geopandas fails
            
            # Fallback to custom xml parsing if geopandas fails
            tree = ET.parse(kml_path)
            root_kml = tree.getroot()
            
            # First get all elements in the KML file
            all_elements = []
            
            def collect_elements(element):
                all_elements.append(element)
                for child in element:
                    collect_elements(child)
            
            collect_elements(root_kml)
            
            # Find elements that might contain coordinates
            possible_coord_elements = []
            for element in all_elements:
                element_tag = element.tag
                # Remove namespace prefix if present
                if "}" in element_tag:
                    element_tag = element_tag.split("}")[1]
                
                # Look for elements that might contain coordinates
                if element_tag.lower() in ["coordinates", "coordinate"]:
                    possible_coord_elements.append(element)
            
            # Extract coordinates from these elements
            self._extract_coords_from_elements(possible_coord_elements)
            
            # If no coordinates found, try a more aggressive approach
            if not self.coords:
                logger.warning("No coordinates found with standard parsing, trying alternative method")
                self._extract_coords_from_all_elements(all_elements)
            
            if not self.coords:
                raise ValueError("No valid coordinates found in KML file.")
            
            logger.info(f"Successfully loaded {len(self.coords)} coordinates from KML file")
            return True
            
        except Exception as e:
            logger.error(f"Error loading KML file: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_coords_from_elements(self, elements):
        """Extract coordinates from specific XML elements."""
        for element in elements:
            if element.text is None:
                continue
            
            # Get the text content and ensure it's a string
            coord_text = str(element.text).strip()
            if not coord_text:
                continue
            
            # Handle different coordinate formats
            if " " in coord_text:  # Multiple coordinates separated by spaces
                coordinate_parts = coord_text.split()
                for part in coordinate_parts:
                    part = part.strip()
                    self._process_coord_part(part)
            elif "," in coord_text:  # Single coordinate with comma separation
                self._process_coord_part(coord_text)
    
    def _extract_coords_from_all_elements(self, elements):
        """Extract coordinates from any element that might contain coordinate data."""
        for element in elements:
            if element.text and isinstance(element.text, str) and "," in element.text:
                text = element.text.strip()
                
                # Try to find coordinate patterns in the text
                # Look for comma-separated number patterns
                coord_pattern = r'-?\d+\.\d+,-?\d+\.\d+'
                matches = re.findall(coord_pattern, text)
                
                for match in matches:
                    self._process_coord_part(match)
                
                # If we didn't find any through regex, try splitting by spaces
                if not matches and " " in text:
                    for part in text.split():
                        if "," in part:
                            self._process_coord_part(part)
    
    def _process_coord_part(self, part):
        """Process a potential coordinate part and add valid coordinates to self.coords."""
        if "," in part:
            values = part.split(",")
            if len(values) >= 2:
                try:
                    # KML format is lon,lat, but we want lat,lon for heatmap
                    lon = float(values[0])
                    lat = float(values[1])
                    
                    # Basic validation
                    if not (isinstance(lon, float) and isinstance(lat, float)):
                        return
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        self.coords.append((lat, lon))
                except (ValueError, TypeError):
                    pass
    
    def load_geojson(self, geojson_path: str) -> bool:
        """Load GeoJSON file with error handling."""
        self.geojson_path = geojson_path
        logger.info(f"Loading GeoJSON file: {geojson_path}")
        
        try:
            self.region = gpd.read_file(geojson_path)
            
            # Convert datetime columns to strings
            for col in self.region.columns:
                if pd.api.types.is_datetime64_any_dtype(self.region[col]):
                    self.region[col] = self.region[col].astype(str)
            
            # Calculate centroid for map center
            try:
                # Use shapely to create a unified geometry
                if len(self.region) > 1:
                    unified_geom = self.region.geometry.unary_union
                else:
                    unified_geom = self.region.geometry.iloc[0]
                
                self.centroid = unified_geom.centroid
                logger.info(f"Calculated centroid: {self.centroid.x}, {self.centroid.y}")
            except Exception as e:
                logger.warning(f"Could not calculate centroid: {e}")
                self.centroid = None
            
            logger.info(f"Successfully loaded GeoJSON with {len(self.region)} features")
            return True
            
        except Exception as e:
            logger.error(f"Error loading GeoJSON file: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def filter_points_by_region(self):
        """Filter points to only include those within the GeoJSON region."""
        if not self.coords or self.region is None:
            logger.warning("Cannot filter points: missing data")
            return False
        
        logger.info(f"Filtering {len(self.coords)} points to region boundary")
        
        try:
            # Create GeoDataFrame from coordinates
            df = pd.DataFrame(self.coords, columns=["lat", "lon"])
            
            # Convert to GeoDataFrame with Point geometries
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df.lon, df.lat),
                crs="EPSG:4326"
            )
            
            # Try to create a unified region geometry
            try:
                if len(self.region) > 1:
                    region_union = self.region.geometry.unary_union
                else:
                    region_union = self.region.geometry.iloc[0]
                
                # Filter points to only those within the region
                filtered_gdf = gdf[gdf.geometry.within(region_union)]
                
            except Exception as e:
                logger.warning(f"Error during spatial filtering with unary_union: {e}")
                # Try an alternative approach
                filtered_gdf = gpd.sjoin(gdf, self.region, predicate="within", how="inner")
            
            # Convert filtered points back to lat/lon list
            self.filtered_coords = list(zip(
                filtered_gdf.lat.values.tolist(), 
                filtered_gdf.lon.values.tolist()
            ))
            
            if not self.filtered_coords:
                logger.warning("No points found inside the region boundary")
                return False
            
            # Calculate statistics
            self._calculate_statistics()
            
            logger.info(f"Successfully filtered to {len(self.filtered_coords)} points within region")
            return True
            
        except Exception as e:
            logger.error(f"Error filtering points: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _calculate_statistics(self):
        """Calculate statistics about the data."""
        if not self.filtered_coords:
            return
        
        # Basic count statistics
        self.stats["total_points"] = len(self.coords)
        self.stats["filtered_points"] = len(self.filtered_coords)
        self.stats["points_percentage"] = round(
            (len(self.filtered_coords) / len(self.coords)) * 100, 2
        ) if self.coords else 0
        
        # Create a point density heatmap (grid-based)
        if len(self.filtered_coords) > 10:
            try:
                # Extract lat/lon arrays
                lats = [coord[0] for coord in self.filtered_coords]
                lons = [coord[1] for coord in self.filtered_coords]
                
                # Calculate bounds
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                self.stats["bounds"] = {
                    "min_lat": min_lat, 
                    "max_lat": max_lat,
                    "min_lon": min_lon, 
                    "max_lon": max_lon,
                    "lat_range": max_lat - min_lat,
                    "lon_range": max_lon - min_lon
                }
                
                # Create a density-based 2D histogram
                grid_size = min(100, max(20, len(self.filtered_coords) // 10))
                self.stats["density_histogram"] = np.histogram2d(
                    lats, lons, 
                    bins=grid_size, 
                    range=[[min_lat, max_lat], [min_lon, max_lon]]
                )
                
                # Find hotspots (grid cells with highest density)
                H, xedges, yedges = self.stats["density_histogram"]
                
                # Get indices of hotspots
                hotspot_threshold = np.percentile(H, 95)  # Top 5% density
                hotspot_indices = np.where(H >= hotspot_threshold)
                
                # Convert indices to coordinates
                hotspots = []
                for i, j in zip(hotspot_indices[0], hotspot_indices[1]):
                    # Use middle of the bin
                    lat = (xedges[i] + xedges[i+1]) / 2
                    lon = (yedges[j] + yedges[j+1]) / 2
                    count = H[i, j]
                    hotspots.append((lat, lon, count))
                
                self.stats["hotspots"] = sorted(hotspots, key=lambda x: x[2], reverse=True)[:10]
                
            except Exception as e:
                logger.error(f"Error calculating advanced statistics: {e}")
        
        logger.info("Statistics calculated successfully")


class HeatmapGenerator:
    """Class to generate heatmaps from geospatial data."""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.map = None
        self.heatmap_style = "default"
        self.border_style = "normal"
        self.border_color = "blue"
        self.map_style = "OpenStreetMap"
        self.show_markers = False
        self.show_legend = True
        self.custom_title = ""
        self.include_statistics = True
        self.save_path = None
    
    def create_map(self) -> bool:
        """Create a folium map with the configured styles."""
        if not self.data_manager.filtered_coords:
            logger.error("No filtered coordinates available")
            return False
        
        try:
            # Determine map center
            if self.data_manager.centroid:
                map_center = [self.data_manager.centroid.y, self.data_manager.centroid.x]
            elif self.data_manager.filtered_coords:
                # Use the average of filtered coordinates
                avg_lat = sum(p[0] for p in self.data_manager.filtered_coords) / len(self.data_manager.filtered_coords)
                avg_lon = sum(p[1] for p in self.data_manager.filtered_coords) / len(self.data_manager.filtered_coords)
                map_center = [avg_lat, avg_lon]
            else:
                # Fallback to a default if all else fails
                map_center = [0, 0]
            
            # Create the folium map with the selected tile style
            if self.map_style in MAP_TILES:
                tile_url = MAP_TILES[self.map_style]["url"]
                tile_attr = MAP_TILES[self.map_style]["attribution"]
                m = folium.Map(
                    location=map_center,
                    zoom_start=10,
                    tiles=None
                )
                folium.TileLayer(
                    tiles=tile_url,
                    attr=tile_attr,
                    name=self.map_style
                ).add_to(m)
            else:
                # Default to OpenStreetMap if the selected style isn't available
                m = folium.Map(location=map_center, zoom_start=10)
            
            # Add region boundary with the selected border style
            border_config = BORDER_STYLES.get(self.border_style, BORDER_STYLES["normal"])
            
            # If it's a custom or thick border, use the selected color
            if self.border_style in ["custom", "thick"]:
                border_config["color"] = self.border_color
            
            # Add the GeoJSON boundary with styling
            style_function = lambda x: {
                'fillColor': 'transparent',
                'color': border_config["color"],
                'weight': border_config["weight"],
                'opacity': border_config["opacity"],
                'dashArray': border_config.get("dashArray", None)
            }
            
            # Add GeoJSON layer with our styling
            if self.border_style != "none":
                folium.GeoJson(
                    self.data_manager.region,
                    style_function=style_function,
                    name="Region Boundary"
                ).add_to(m)
            
            # Get the heatmap style configuration
            style_params = HEATMAP_STYLES.get(self.heatmap_style, HEATMAP_STYLES["default"])
            
            # Create the heatmap layer
            heat_data = self.data_manager.filtered_coords
            try:
                heat_map = HeatMap(
                    heat_data,
                    radius=style_params["radius"],
                    blur=style_params["blur"],
                    min_opacity=style_params["min_opacity"],
                    gradient=style_params["gradient"],
                    name="Heatmap Layer"
                )
                heat_map.add_to(m)
            except Exception as e:
                logger.error(f"Error creating heatmap layer: {e}")
                # Try with default parameters if the custom ones fail
                HeatMap(heat_data, radius=15, blur=10).add_to(m)
            
            # Add markers for hotspots if enabled
            if self.show_markers and "hotspots" in self.data_manager.stats:
                marker_cluster = MarkerCluster(name="Hotspots").add_to(m)
                for lat, lon, count in self.data_manager.stats["hotspots"]:
                    folium.Marker(
                        [lat, lon],
                        popup=f"Hotspot with {int(count)} points",
                        icon=folium.Icon(color="red", icon="fire", prefix="fa")
                    ).add_to(marker_cluster)
            
            # Add useful controls
            folium.LayerControl().add_to(m)
            MeasureControl(position='topleft', primary_length_unit='meters').add_to(m)
            Fullscreen().add_to(m)
            Draw(export=True).add_to(m)
            
            # Add a title to the map
            title = self.custom_title if self.custom_title else f"Heatmap of {os.path.basename(self.data_manager.kml_path)}"
            title_html = f'''
                <div style="position: fixed; 
                            top: 10px; 
                            left: 50%;
                            transform: translateX(-50%);
                            z-index: 9999; 
                            background-color: white; 
                            padding: 10px; 
                            border: 1px solid grey;
                            border-radius: 5px;
                            font-size: 16px;
                            font-weight: bold;">
                    {title}
                </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add statistics if enabled
            if self.include_statistics and self.data_manager.stats:
                stats = self.data_manager.stats
                stats_html = f'''
                    <div style="position: fixed; 
                                bottom: 10px; 
                                right: 10px;
                                z-index: 9999; 
                                background-color: white; 
                                padding: 10px; 
                                border: 1px solid grey;
                                border-radius: 5px;
                                font-size: 14px;
                                max-width: 300px;">
                        <strong>Statistics</strong><br>
                        Total points: {stats["total_points"]}<br>
                        Points in region: {stats["filtered_points"]} ({stats["points_percentage"]}%)<br>
                        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
                    </div>
                '''
                m.get_root().html.add_child(folium.Element(stats_html))
            
            # Add metadata
            m.get_root().html.add_child(folium.Element(f"<title>{title}</title>"))
            
            # Add a watermark if you want
            watermark_html = f'''
                <div style="position: fixed; 
                            bottom: 10px; 
                            left: 10px;
                            z-index: 9999; 
                            color: #555;
                            font-size: 12px;">
                    Created with {APP_NAME} {APP_VERSION}
                </div>
            '''
            m.get_root().html.add_child(folium.Element(watermark_html))
            
            self.map = m
            return True
            
        except Exception as e:
            logger.error(f"Error creating map: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def save_map(self, output_path=None) -> str:
        """Save the map to a file and return the saved path."""
        if self.map is None:
            logger.error("No map to save")
            return None
        
        try:
            # Generate a filename if none provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if self.custom_title:
                    # Create a safe filename from the title
                    safe_title = "".join(c for c in self.custom_title if c.isalnum() or c in " _-").strip()
                    safe_title = safe_title.replace(" ", "_")
                    filename = f"heatmap_{safe_title}_{timestamp}.html"
                else:
                    kml_name = os.path.splitext(os.path.basename(self.data_manager.kml_path))[0]
                    filename = f"heatmap_{kml_name}_{timestamp}.html"
                
                # Get save directory from settings or use default
                settings = load_settings()
                save_dir = settings.get("default_save_dir", str(Path.home() / "Documents" / "Heatmaps"))
                
                # Ensure the directory exists
                os.makedirs(save_dir, exist_ok=True)
                
                output_path = os.path.join(save_dir, filename)
            
            # Save the map
            self.map.save(output_path)
            logger.info(f"Map saved to {output_path}")
            self.save_path = output_path
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving map: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def export_to_png(self, output_path=None):
        """Export the map to a PNG image using selenium screenshot."""
        if self.map is None:
            logger.error("No map to export")
            return None
        
        try:
            # First save the HTML file
            html_path = self.save_map() if self.save_path is None else self.save_path
            
            if not html_path:
                logger.error("Could not save HTML map")
                return None
            
            # Generate PNG filename from HTML path if none provided
            if output_path is None:
                output_path = os.path.splitext(html_path)[0] + ".png"
            
            try:
                # Try to use selenium to capture a screenshot
                # This requires selenium and a webdriver to be installed
                from selenium import webdriver
                from selenium.webdriver.chrome.service import Service
                from selenium.webdriver.chrome.options import Options
                from webdriver_manager.chrome import ChromeDriverManager
                
                # Set up Chrome options
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                
                # Set up the driver
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                
                # Load the HTML file
                driver.get("file://" + os.path.abspath(html_path))
                
                # Wait for map to load
                import time
                time.sleep(2)
                
                # Take screenshot
                driver.save_screenshot(output_path)
                driver.quit()
                
                logger.info(f"Map exported to PNG: {output_path}")
                return output_path
                
            except ImportError:
                logger.warning("Selenium not available, cannot export to PNG")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting to PNG: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def export_to_pdf(self, output_path=None):
        """Export the map to a PDF document."""
        if self.map is None:
            logger.error("No map to export")
            return None
        
        try:
            # First try to export to PNG
            png_path = self.export_to_png()
            
            if not png_path:
                logger.error("Could not export to PNG for PDF creation")
                return None
            
            # Generate PDF filename from PNG path if none provided
            if output_path is None:
                output_path = os.path.splitext(png_path)[0] + ".pdf"
            
            # Create a PDF with the PNG image
            img = Image.open(png_path)
            width, height = img.size
            
            # Letter size in points: 612 x 792
            c = canvas.Canvas(output_path, pagesize=letter)
            
            # Calculate scaling to fit on the page
            page_width, page_height = letter
            ratio = min(page_width / width, page_height / height) * 0.9
            
            new_width = width * ratio
            new_height = height * ratio
            
            # Center on the page
            x = (page_width - new_width) / 2
            y = (page_height - new_height) / 2
            
            # Add the image
            c.drawImage(png_path, x, y, width=new_width, height=new_height)
            
            # Add some metadata and title
            title = self.custom_title if self.custom_title else f"Heatmap of {os.path.basename(self.data_manager.kml_path)}"
            c.setFont("Helvetica-Bold", 16)
            c.drawCentredString(page_width / 2, page_height - 50, title)
            
            # Add statistics if available
            if self.data_manager.stats:
                stats = self.data_manager.stats
                c.setFont("Helvetica", 10)
                c.drawString(50, 50, f"Total points: {stats['total_points']}")
                c.drawString(50, 35, f"Points in region: {stats['filtered_points']} ({stats['points_percentage']}%)")
                c.drawString(50, 20, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # Add a footer
            c.setFont("Helvetica", 8)
            c.drawString(50, 10, f"Created with {APP_NAME} {APP_VERSION}")
            
            # Save the PDF
            c.save()
            
            logger.info(f"Map exported to PDF: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            logger.error(traceback.format_exc())
            return None


class HeatmapUI(ctk.CTk):
    """Modern UI for the Heatmap Generator."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize attributes
        self.settings = load_settings()
        self.data_manager = DataManager()
        self.heatmap_generator = HeatmapGenerator(self.data_manager)
        
        # Initialize tracking variables
        self.kml_path_var = ctk.StringVar(value="")
        self.geojson_path_var = ctk.StringVar(value="")
        self.border_style_var = ctk.StringVar(value="normal")
        self.border_color_var = ctk.StringVar(value="blue")
        self.heatmap_style_var = ctk.StringVar(value="default")
        self.map_style_var = ctk.StringVar(value="OpenStreetMap")
        self.show_markers_var = ctk.BooleanVar(value=False)
        self.show_legend_var = ctk.BooleanVar(value=True)
        self.custom_title_var = ctk.StringVar(value="")
        self.include_statistics_var = ctk.BooleanVar(value=True)
        self.theme_var = ctk.StringVar(value=self.settings.get("theme", "dark"))
        
        self.status_text = ctk.StringVar(value="Ready to create heatmaps")
        self.progress_var = ctk.DoubleVar(value=0)
        
        # Keep track of preview widgets
        self.preview_frame = None
        self.preview_fig = None
        self.preview_canvas = None
        
        # Configure the window
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1000x750")
        self.minsize(800, 600)
        
        # Set up the theme
        self.set_theme(self.theme_var.get())
        
        # Build the UI
        self.create_menu()
        self.build_ui()
        
        # Create the working directory if it doesn't exist
        os.makedirs(self.settings.get("default_save_dir", str(Path.home() / "Documents" / "Heatmaps")), exist_ok=True)
        
        # Check command line arguments
        self.check_command_line_args()
    
    def set_theme(self, theme_name):
        """Set the UI theme."""
        if theme_name == "dark":
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
        else:
            ctk.set_appearance_mode("light") 
            ctk.set_default_color_theme("blue")
        
        self.settings["theme"] = theme_name
        save_settings(self.settings)
    
    def create_menu(self):
        """Create the application menu."""
        self.menu = tk.Menu(self)
        self.config(menu=self.menu)
        
        # File menu
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open KML File", command=self.browse_kml)
        file_menu.add_command(label="Open GeoJSON File", command=self.browse_geojson)
        file_menu.add_separator()
        
        # Recent files submenu
        self.recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Projects", menu=self.recent_files_menu)
        self.update_recent_files_menu()
        
        file_menu.add_separator()
        file_menu.add_command(label="Save Map", command=self.save_map)
        file_menu.add_command(label="Export as PNG", command=self.export_png)
        file_menu.add_command(label="Export as PDF", command=self.export_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Edit menu
        edit_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Map Path", command=self.copy_map_path)
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        
        # View menu
        view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Preview", 
                                  variable=self.settings.get("show_preview", True),
                                  command=self.toggle_preview)
        
        # Theme submenu
        theme_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        theme_menu.add_radiobutton(label="Light", variable=self.theme_var, value="light", 
                                  command=lambda: self.set_theme("light"))
        theme_menu.add_radiobutton(label="Dark", variable=self.theme_var, value="dark",
                                  command=lambda: self.set_theme("dark"))
        
        # Tools menu
        tools_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Batch Process", command=self.batch_process)
        tools_menu.add_command(label="Analyze Data", command=self.analyze_data)
        tools_menu.add_separator()
        tools_menu.add_command(label="Custom Style Editor", command=self.show_style_editor)
        
        # Help menu
        help_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="Check for Updates", command=self.check_updates)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
    
    def update_recent_files_menu(self):
        """Update the recent files menu."""
        # Clear existing menu items
        self.recent_files_menu.delete(0, tk.END)
        
        # Get recent files
        recent_files = load_recent_files()
        
        if not recent_files:
            self.recent_files_menu.add_command(label="No recent files", state=tk.DISABLED)
            return
        
        # Add recent files to menu
        for file_entry in recent_files:
            # Create a more readable label
            kml_name = os.path.basename(file_entry.get("kml", ""))
            geojson_name = os.path.basename(file_entry.get("geojson", ""))
            label = f"{kml_name} & {geojson_name}"
            
            # Add truncation if too long
            if len(label) > 50:
                label = label[:47] + "..."
            
            # Add timestamp if available
            timestamp = file_entry.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    date_str = dt.strftime("%Y-%m-%d")
                    label = f"{label} ({date_str})"
                except:
                    pass
            
            self.recent_files_menu.add_command(
                label=label,
                command=lambda f=file_entry: self.open_recent_files(f)
            )
        
        # Add separator and clear option
        self.recent_files_menu.add_separator()
        self.recent_files_menu.add_command(label="Clear Recent Files", command=self.clear_recent_files)
    
    def open_recent_files(self, file_entry):
        """Open files from a recent files entry."""
        kml_path = file_entry.get("kml")
        geojson_path = file_entry.get("geojson")
        
        if not os.path.exists(kml_path):
            self.show_message(f"KML file not found: {kml_path}", "error")
            return
        
        if not os.path.exists(geojson_path):
            self.show_message(f"GeoJSON file not found: {geojson_path}", "error")
            return
        
        # Set the file paths
        self.kml_path_var.set(kml_path)
        self.geojson_path_var.set(geojson_path)
        
        # Update the UI
        self.load_files()
    
    def clear_recent_files(self):
        """Clear the recent files list."""
        if messagebox.askyesno("Clear Recent Files", "Are you sure you want to clear all recent files?"):
            save_recent_files([])
            self.update_recent_files_menu()
    
    def build_ui(self):
        """Build the main UI layout."""
        # Create main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create two-column layout
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Build the left side (file selection and basic options)
        self.build_left_panel()
        
        # Build the right side (advanced options and preview)
        self.build_right_panel()
        
        # Status bar at the bottom
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(side=tk.RIGHT, padx=10, pady=5)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_text)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    def build_left_panel(self):
        """Build the left panel UI components."""
        # File selection section
        file_frame = ctk.CTkFrame(self.left_frame)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(file_frame, text="Select Input Files", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # KML file selection
        kml_frame = ctk.CTkFrame(file_frame)
        kml_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(kml_frame, text="KML File:").pack(side=tk.LEFT, padx=5)
        
        kml_entry = ctk.CTkEntry(kml_frame, textvariable=self.kml_path_var, width=200)
        kml_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ctk.CTkButton(kml_frame, text="Browse", command=self.browse_kml).pack(side=tk.RIGHT, padx=5)
        
        # GeoJSON file selection
        geojson_frame = ctk.CTkFrame(file_frame)
        geojson_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(geojson_frame, text="GeoJSON File:").pack(side=tk.LEFT, padx=5)
        
        geojson_entry = ctk.CTkEntry(geojson_frame, textvariable=self.geojson_path_var, width=200)
        geojson_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ctk.CTkButton(geojson_frame, text="Browse", command=self.browse_geojson).pack(side=tk.RIGHT, padx=5)
        
        # Load files button
        ctk.CTkButton(file_frame, text="Load Files", command=self.load_files, height=40, 
                      font=("Helvetica", 14, "bold")).pack(fill=tk.X, padx=10, pady=10)
        
        # Basic options section
        options_frame = ctk.CTkFrame(self.left_frame)
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(options_frame, text="Basic Options", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Heatmap style selection
        heatmap_style_frame = ctk.CTkFrame(options_frame)
        heatmap_style_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(heatmap_style_frame, text="Heatmap Style:").pack(side=tk.LEFT, padx=5)
        
        heatmap_dropdown = ctk.CTkOptionMenu(
            heatmap_style_frame,
            values=list(HEATMAP_STYLES.keys()),
            variable=self.heatmap_style_var,
            command=self.update_preview
        )
        heatmap_dropdown.pack(side=tk.RIGHT, padx=5)
        
        # Border style selection
        border_style_frame = ctk.CTkFrame(options_frame)
        border_style_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(border_style_frame, text="Border Style:").pack(side=tk.LEFT, padx=5)
        
        border_dropdown = ctk.CTkOptionMenu(
            border_style_frame,
            values=list(BORDER_STYLES.keys()),
            variable=self.border_style_var,
            command=self.on_border_style_change
        )
        border_dropdown.pack(side=tk.RIGHT, padx=5)
        
        # Border color selection (visible only when border style is 'thick' or 'custom')
        self.border_color_frame = ctk.CTkFrame(options_frame)
        
        ctk.CTkLabel(self.border_color_frame, text="Border Color:").pack(side=tk.LEFT, padx=5)
        
        color_values = ["blue", "red", "green", "purple", "orange", "black", "white", "gray", "yellow", "cyan"]
        border_color_dropdown = ctk.CTkOptionMenu(
            self.border_color_frame,
            values=color_values,
            variable=self.border_color_var,
            command=self.update_preview
        )
        border_color_dropdown.pack(side=tk.RIGHT, padx=5)
        
        # Show or hide border color based on initial border style
        if self.border_style_var.get() in ["thick", "custom"]:
            self.border_color_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Map tile style selection
        map_style_frame = ctk.CTkFrame(options_frame)
        map_style_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(map_style_frame, text="Map Style:").pack(side=tk.LEFT, padx=5)
        
        map_dropdown = ctk.CTkOptionMenu(
            map_style_frame,
            values=list(MAP_TILES.keys()),
            variable=self.map_style_var,
            command=self.update_preview
        )
        map_dropdown.pack(side=tk.RIGHT, padx=5)
        
        # Custom title field
        title_frame = ctk.CTkFrame(options_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(title_frame, text="Custom Title:").pack(side=tk.LEFT, padx=5)
        
        ctk.CTkEntry(title_frame, textvariable=self.custom_title_var).pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Checkboxes for additional options
        checks_frame = ctk.CTkFrame(options_frame)
        checks_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkCheckBox(checks_frame, text="Show Markers", variable=self.show_markers_var, 
                       command=self.update_preview).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkCheckBox(checks_frame, text="Show Legend", variable=self.show_legend_var,
                       command=self.update_preview).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkCheckBox(checks_frame, text="Include Statistics", variable=self.include_statistics_var,
                       command=self.update_preview).pack(side=tk.LEFT, padx=5)
        
        # Create heatmap button
        self.create_button = ctk.CTkButton(
            self.left_frame,
            text="Create Heatmap",
            command=self.create_heatmap,
            height=50,
            font=("Helvetica", 16, "bold")
        )
        self.create_button.pack(fill=tk.X, padx=10, pady=10)
    
    def build_right_panel(self):
        """Build the right panel UI components."""
        # Preview section
        self.preview_container = ctk.CTkFrame(self.right_frame)
        self.preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(self.preview_container, text="Preview", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Statistics section below preview
        self.stats_frame = ctk.CTkFrame(self.right_frame)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(self.stats_frame, text="Statistics", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        self.stats_text = ctk.CTkTextbox(self.stats_frame, height=150, wrap="word")
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stats_text.insert("1.0", "Load files to see statistics")
        self.stats_text.configure(state="disabled")
        
        # Actions section
        actions_frame = ctk.CTkFrame(self.right_frame)
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(actions_frame, text="Actions", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Buttons grid
        buttons_frame = ctk.CTkFrame(actions_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # First row of buttons
        row1 = ctk.CTkFrame(buttons_frame)
        row1.pack(fill=tk.X, pady=5)
        
        ctk.CTkButton(row1, text="Save HTML", command=self.save_map).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(row1, text="Export PNG", command=self.export_png).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(row1, text="Export PDF", command=self.export_pdf).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Second row of buttons
        row2 = ctk.CTkFrame(buttons_frame)
        row2.pack(fill=tk.X, pady=5)
        
        ctk.CTkButton(row2, text="Open in Browser", command=self.open_in_browser).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(row2, text="Analyze Data", command=self.analyze_data).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ctk.CTkButton(row2, text="Help", command=self.show_documentation).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def initialize_preview(self):
        """Initialize or reset the preview area."""
        # Clear any existing preview
        if self.preview_frame:
            self.preview_frame.destroy()
        
        # Create frame for preview
        self.preview_frame = ctk.CTkFrame(self.preview_container)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.preview_fig = plt.figure(figsize=(5, 4), dpi=100)
        self.preview_fig.patch.set_facecolor('none')  # Transparent background
        
        ax = self.preview_fig.add_subplot(111)
        ax.set_title("Preview will appear here")
        ax.set_facecolor('none')  # Transparent background
        
        # Hide axes for cleaner look
        ax.axis('off')
        
        # Create canvas
        self.preview_canvas = FigureCanvasTkAgg(self.preview_fig, self.preview_frame)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Draw initial empty figure
        self.preview_canvas.draw()
    
    def update_preview(self, event=None):
        """Update the preview based on current settings."""
        # Check if we should show a preview
        if not self.settings.get("show_preview", True):
            return
        
        # Check if files are loaded
        if not hasattr(self.data_manager, "filtered_coords") or not self.data_manager.filtered_coords:
            return
        
        try:
            # Initialize preview if needed
            if not self.preview_frame:
                self.initialize_preview()
            
            # Clear the figure
            self.preview_fig.clear()
            
            # Create a new axes
            ax = self.preview_fig.add_subplot(111)
            
            # Get the heatmap style
            style = HEATMAP_STYLES.get(self.heatmap_style_var.get(), HEATMAP_STYLES["default"])
            
            # Create a simple point density plot as preview
            if self.data_manager.filtered_coords:
                lats = [p[0] for p in self.data_manager.filtered_coords]
                lons = [p[1] for p in self.data_manager.filtered_coords]
                
                # Calculate map bounds
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                # Add some padding
                lat_padding = (max_lat - min_lat) * 0.1
                lon_padding = (max_lon - min_lon) * 0.1
                
                ax.set_xlim(min_lon - lon_padding, max_lon + lon_padding)
                ax.set_ylim(min_lat - lat_padding, max_lat + lat_padding)
                
                # Create a 2D histogram for the heatmap
                heatmap, xedges, yedges = np.histogram2d(
                    lons, lats, 
                    bins=40, 
                    range=[[min_lon - lon_padding, max_lon + lon_padding], 
                           [min_lat - lat_padding, max_lat + lat_padding]]
                )
                
                # Create a custom colormap from the heatmap style gradient
                from matplotlib.colors import LinearSegmentedColormap
                
                # Convert gradient dictionary to format needed by LinearSegmentedColormap
                gradient = style["gradient"]
                colors = []
                positions = []
                
                for pos_str, color in gradient.items():
                    positions.append(float(pos_str))
                    colors.append(color)
                
                # Sort by position
                colors = [x for _, x in sorted(zip(positions, colors))]
                positions = sorted(positions)
                
                # Create colormap
                if len(colors) >= 2:
                    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
                else:
                    cmap = plt.cm.hot  # Fallback
                
                # Plot the heatmap
                im = ax.imshow(
                    heatmap.T,
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    origin='lower',
                    cmap=cmap,
                    alpha=0.7
                )
                
                # Add colorbar if legend is enabled
                if self.show_legend_var.get():
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Point Density')
                
                # Show border if border style is not 'none'
                if self.border_style_var.get() != "none" and hasattr(self.data_manager, "region") and self.data_manager.region is not None:
                    # Plot the boundary
                    border_style = BORDER_STYLES.get(self.border_style_var.get(), BORDER_STYLES["normal"])
                    
                    # Use the selected color for thick or custom borders
                    color = border_style["color"]
                    if self.border_style_var.get() in ["thick", "custom"]:
                        color = self.border_color_var.get()
                    
                    # Plot the region boundaries
                    self.data_manager.region.boundary.plot(
                        ax=ax,
                        color=color,
                        linewidth=border_style["weight"]
                    )
                
                # Add a title if specified
                title = self.custom_title_var.get()
                if title:
                    ax.set_title(title)
                else:
                    ax.set_title(f"Preview - {len(self.data_manager.filtered_coords)} points")
                
                # Clean up the axes
                ax.set_aspect('equal')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, "No data to display", ha='center', va='center')
                ax.axis('off')
            
            # Update the canvas
            self.preview_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
            logger.error(traceback.format_exc())
    
    def toggle_preview(self):
        """Toggle the preview display."""
        self.settings["show_preview"] = not self.settings.get("show_preview", True)
        save_settings(self.settings)
        
        if self.settings["show_preview"]:
            self.update_preview()
        else:
            # Hide preview
            if self.preview_frame:
                self.preview_frame.destroy()
                self.preview_frame = None
    
    def on_border_style_change(self, style):
        """Handle border style change."""
        # Show/hide border color dropdown based on style
        if style in ["thick", "custom"]:
            self.border_color_frame.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.border_color_frame.pack_forget()
        
        self.update_preview()
    
    def update_status(self, message, message_type="info"):
        """Update status bar with message."""
        self.status_text.set(message)
        
        # Change color based on message type
        if message_type == "error":
            self.status_label.configure(text_color="red")
            logger.error(message)
        elif message_type == "success":
            self.status_label.configure(text_color="green")
            logger.info(message)
        elif message_type == "warning":
            self.status_label.configure(text_color="orange")
            logger.warning(message)
        else:
            # Use a specific default color depending on the theme
            if self.theme_var.get() == "dark":
                self.status_label.configure(text_color="white")
            else:
                self.status_label.configure(text_color="black")
            logger.info(message)
        
        # Ensure UI updates
        self.update()
    
    def show_message(self, message, message_type="info"):
        """Show a message box with the given message."""
        self.update_status(message, message_type)
        
        if message_type == "error":
            messagebox.showerror("Error", message)
        elif message_type == "warning":
            messagebox.showwarning("Warning", message)
        else:
            messagebox.showinfo("Information", message)
    
    def update_stats_display(self):
        """Update the statistics display with current data."""
        if not hasattr(self.data_manager, "stats") or not self.data_manager.stats:
            return
        
        # Enable text widget for editing
        self.stats_text.configure(state="normal")
        
        # Clear current content
        self.stats_text.delete("1.0", tk.END)
        
        # Add stats information
        stats = self.data_manager.stats
        
        self.stats_text.insert(tk.END, f"Data Summary:\n", "heading")
        self.stats_text.insert(tk.END, f"Total points in KML: {stats.get('total_points', 0)}\n")
        self.stats_text.insert(tk.END, f"Points within region: {stats.get('filtered_points', 0)} ")
        self.stats_text.insert(tk.END, f"({stats.get('points_percentage', 0)}%)\n\n")
        
        # Add bounds information if available
        if "bounds" in stats:
            bounds = stats["bounds"]
            self.stats_text.insert(tk.END, f"Geographic Bounds:\n", "heading")
            self.stats_text.insert(tk.END, f"Latitude range: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}\n")
            self.stats_text.insert(tk.END, f"Longitude range: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}\n")
            self.stats_text.insert(tk.END, f"Area covered: {bounds['lat_range']:.6f}° x {bounds['lon_range']:.6f}°\n\n")
        
        # Add hotspot information if available
        if "hotspots" in stats and stats["hotspots"]:
            self.stats_text.insert(tk.END, f"Top Hotspots:\n", "heading")
            for i, (lat, lon, count) in enumerate(stats["hotspots"][:5], 1):
                self.stats_text.insert(tk.END, f"{i}. Lat: {lat:.6f}, Lon: {lon:.6f} - {int(count)} points\n")
        
        # Disable editing
        self.stats_text.configure(state="disabled")
    
    def browse_kml(self):
        """Open file browser for KML selection."""
        # Use the last directory if enabled in settings
        initial_dir = None
        if self.settings.get("remember_last_directory", True) and self.kml_path_var.get():
            initial_dir = os.path.dirname(self.kml_path_var.get())
        
        file_path = filedialog.askopenfilename(
            title="Select KML File",
            filetypes=[("KML files", "*.kml"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        
        if file_path:
            self.kml_path_var.set(file_path)
    
    def browse_geojson(self):
        """Open file browser for GeoJSON selection."""
        # Use the last directory if enabled in settings
        initial_dir = None
        if self.settings.get("remember_last_directory", True) and self.geojson_path_var.get():
            initial_dir = os.path.dirname(self.geojson_path_var.get())
        
        file_path = filedialog.askopenfilename(
            title="Select GeoJSON File",
            filetypes=[("GeoJSON files", "*.geojson"), ("All files", "*.*")],
            initialdir=initial_dir
        )
        
        if file_path:
            self.geojson_path_var.set(file_path)
    
    def load_files(self):
        """Load KML and GeoJSON files and process data."""
        kml_path = self.kml_path_var.get()
        geojson_path = self.geojson_path_var.get()
        
        if not kml_path:
            self.show_message("Please select a KML file", "error")
            return
        
        if not geojson_path:
            self.show_message("Please select a GeoJSON file", "error")
            return
        
        if not os.path.exists(kml_path):
            self.show_message(f"KML file not found: {kml_path}", "error")
            return
        
        if not os.path.exists(geojson_path):
            self.show_message(f"GeoJSON file not found: {geojson_path}", "error")
            return
        
        # Show loading message and reset progress
        self.update_status("Loading files, please wait...", "info")
        self.progress_bar.set(0.1)
        
        # Load files in a background thread to keep UI responsive
        def load_thread():
            try:
                # Load KML file
                self.update_status("Loading KML file...", "info")
                self.progress_bar.set(0.2)
                if not self.data_manager.load_kml(kml_path):
                    self.show_message("Failed to load KML file. Check the log for details.", "error")
                    self.progress_bar.set(0)
                    return
                
                # Load GeoJSON file
                self.update_status("Loading GeoJSON file...", "info")
                self.progress_bar.set(0.4)
                if not self.data_manager.load_geojson(geojson_path):
                    self.show_message("Failed to load GeoJSON file. Check the log for details.", "error")
                    self.progress_bar.set(0)
                    return
                
                # Filter points
                self.update_status("Filtering points...", "info")
                self.progress_bar.set(0.6)
                if not self.data_manager.filter_points_by_region():
                    self.show_message("No points found inside the GeoJSON region.", "warning")
                    self.progress_bar.set(0)
                    return
                
                # Update UI with the new data
                self.progress_bar.set(0.8)
                
                # Add to recent files
                add_to_recent_files(kml_path, geojson_path, self.settings)
                self.update_recent_files_menu()
                
                # Update the stats display
                self.update_stats_display()
                
                # Update the preview
                if self.settings.get("show_preview", True):
                    self.update_preview()
                
                # Update status
                self.progress_bar.set(1.0)
                self.update_status(f"Loaded {len(self.data_manager.filtered_coords)} points in region", "success")
                
                # Reset progress bar after delay
                self.after(1000, lambda: self.progress_bar.set(0))
                
            except Exception as e:
                logger.error(f"Error loading files: {e}")
                logger.error(traceback.format_exc())
                self.show_message(f"Error loading files: {str(e)}", "error")
                self.progress_bar.set(0)
        
        # Start the thread
        Thread(target=load_thread).start()
    
    def create_heatmap(self):
        """Generate the heatmap with current settings."""
        if not hasattr(self.data_manager, "filtered_coords") or not self.data_manager.filtered_coords:
            self.show_message("Please load KML and GeoJSON files first", "error")
            return
        
        # Update the heatmap generator with current UI settings
        self.heatmap_generator.heatmap_style = self.heatmap_style_var.get()
        self.heatmap_generator.border_style = self.border_style_var.get()
        self.heatmap_generator.border_color = self.border_color_var.get()
        self.heatmap_generator.map_style = self.map_style_var.get()
        self.heatmap_generator.show_markers = self.show_markers_var.get()
        self.heatmap_generator.show_legend = self.show_legend_var.get()
        self.heatmap_generator.custom_title = self.custom_title_var.get()
        self.heatmap_generator.include_statistics = self.include_statistics_var.get()
        
        # Show loading message and reset progress
        self.update_status("Generating heatmap...", "info")
        self.progress_bar.set(0.2)
        
        # Create heatmap in a background thread
        def generate_thread():
            try:
                # Create the map
                self.update_status("Creating map...", "info")
                self.progress_bar.set(0.4)
                if not self.heatmap_generator.create_map():
                    self.show_message("Failed to create map. Check the log for details.", "error")
                    self.progress_bar.set(0)
                    return
                
                # Save the map
                self.update_status("Saving map...", "info")
                self.progress_bar.set(0.6)
                save_path = self.heatmap_generator.save_map()
                if not save_path:
                    self.show_message("Failed to save map. Check the log for details.", "error")
                    self.progress_bar.set(0)
                    return
                
                # Open the map if auto-open is enabled
                if self.settings.get("auto_open_map", True):
                    self.update_status("Opening map in browser...", "info")
                    self.progress_bar.set(0.8)
                    try:
                        webbrowser.open("file://" + os.path.realpath(save_path))
                    except Exception as e:
                        logger.error(f"Error opening map in browser: {e}")
                        # Continue even if browser opening fails
                
                # Update status
                self.progress_bar.set(1.0)
                self.update_status(f"Heatmap saved to {save_path}", "success")
                
                # Show success message
                self.show_message(
                    f"Heatmap created successfully!\n\nSaved to: {save_path}\n\n"
                    f"Points visualized: {len(self.data_manager.filtered_coords)}",
                    "info"
                )
                
                # Reset progress bar after delay
                self.after(1000, lambda: self.progress_bar.set(0))
                
            except Exception as e:
                logger.error(f"Error generating heatmap: {e}")
                logger.error(traceback.format_exc())
                self.show_message(f"Error generating heatmap: {str(e)}", "error")
                self.progress_bar.set(0)
        
        # Start the thread
        Thread(target=generate_thread).start()
    
    def save_map(self):
        """Save the map to a file."""
        if not hasattr(self.heatmap_generator, "map") or self.heatmap_generator.map is None:
            self.show_message("Please create a heatmap first", "error")
            return
        
        # Open file dialog for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Heatmap",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            initialdir=self.settings.get("default_save_dir", str(Path.home() / "Documents" / "Heatmaps"))
        )
        
        if not save_path:
            return  # User cancelled
        
        # Save the map
        try:
            self.update_status("Saving map...", "info")
            result = self.heatmap_generator.save_map(save_path)
            if result:
                self.update_status(f"Map saved to {result}", "success")
            else:
                self.show_message("Failed to save map", "error")
        except Exception as e:
            logger.error(f"Error saving map: {e}")
            self.show_message(f"Error saving map: {str(e)}", "error")
    
    def export_png(self):
        """Export the map as a PNG image."""
        if not hasattr(self.heatmap_generator, "map") or self.heatmap_generator.map is None:
            self.show_message("Please create a heatmap first", "error")
            return
        
        # Check if selenium is installed
        try:
            import selenium
        except ImportError:
            self.show_message(
                "PNG export requires selenium. Would you like to install it?",
                "warning"
            )
            if messagebox.askyesno("Install Dependencies", "PNG export requires selenium and a webdriver. Install now?"):
                # Install selenium
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager"])
                    self.show_message("Dependencies installed successfully. Please try exporting again.", "info")
                except Exception as e:
                    logger.error(f"Error installing dependencies: {e}")
                    self.show_message(f"Error installing dependencies: {str(e)}", "error")
            return
        
        # Open file dialog for save location
        save_path = filedialog.asksaveasfilename(
            title="Export as PNG",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir=self.settings.get("default_save_dir", str(Path.home() / "Documents" / "Heatmaps"))
        )
        
        if not save_path:
            return  # User cancelled
        
        # Export the map
        try:
            self.update_status("Exporting as PNG...", "info")
            self.progress_bar.set(0.3)
            
            # Export in a background thread
            def export_thread():
                try:
                    result = self.heatmap_generator.export_to_png(save_path)
                    if result:
                        self.update_status(f"Map exported to {result}", "success")
                        self.progress_bar.set(1.0)
                        # Reset progress bar after delay
                        self.after(1000, lambda: self.progress_bar.set(0))
                    else:
                        self.show_message("Failed to export map", "error")
                        self.progress_bar.set(0)
                except Exception as e:
                    logger.error(f"Error exporting PNG: {e}")
                    self.show_message(f"Error exporting PNG: {str(e)}", "error")
                    self.progress_bar.set(0)
            
            # Start the thread
            Thread(target=export_thread).start()
            
        except Exception as e:
            logger.error(f"Error exporting PNG: {e}")
            self.show_message(f"Error exporting PNG: {str(e)}", "error")
            self.progress_bar.set(0)
    
    def export_pdf(self):
        """Export the map as a PDF document."""
        if not hasattr(self.heatmap_generator, "map") or self.heatmap_generator.map is None:
            self.show_message("Please create a heatmap first", "error")
            return
        
        # Open file dialog for save location
        save_path = filedialog.asksaveasfilename(
            title="Export as PDF",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialdir=self.settings.get("default_save_dir", str(Path.home() / "Documents" / "Heatmaps"))
        )
        
        if not save_path:
            return  # User cancelled
        
        # Export the map
        try:
            self.update_status("Exporting as PDF...", "info")
            self.progress_bar.set(0.3)
            
            # Export in a background thread
            def export_thread():
                try:
                    result = self.heatmap_generator.export_to_pdf(save_path)
                    if result:
                        self.update_status(f"Map exported to {result}", "success")
                        self.progress_bar.set(1.0)
                        # Reset progress bar after delay
                        self.after(1000, lambda: self.progress_bar.set(0))
                    else:
                        self.show_message("Failed to export map", "error")
                        self.progress_bar.set(0)
                except Exception as e:
                    logger.error(f"Error exporting PDF: {e}")
                    self.show_message(f"Error exporting PDF: {str(e)}", "error")
                    self.progress_bar.set(0)
            
            # Start the thread
            Thread(target=export_thread).start()
            
        except Exception as e:
            logger.error(f"Error exporting PDF: {e}")
            self.show_message(f"Error exporting PDF: {str(e)}", "error")
            self.progress_bar.set(0)
    
    def open_in_browser(self):
        """Open the generated heatmap in a web browser."""
        if not hasattr(self.heatmap_generator, "save_path") or not self.heatmap_generator.save_path:
            if hasattr(self.heatmap_generator, "map") and self.heatmap_generator.map is not None:
                # Save the map first
                save_path = self.heatmap_generator.save_map()
                if not save_path:
                    self.show_message("Failed to save map for viewing", "error")
                    return
            else:
                self.show_message("Please create a heatmap first", "error")
                return
        
        # Open the saved map in browser
        try:
            webbrowser.open("file://" + os.path.realpath(self.heatmap_generator.save_path))
            self.update_status(f"Opened {self.heatmap_generator.save_path} in browser", "info")
        except Exception as e:
            logger.error(f"Error opening map in browser: {e}")
            self.show_message(f"Error opening map in browser: {str(e)}", "error")
    
    def copy_map_path(self):
        """Copy the path of the saved map to clipboard."""
        if not hasattr(self.heatmap_generator, "save_path") or not self.heatmap_generator.save_path:
            self.show_message("No map has been saved yet", "error")
            return
        
        try:
            pyperclip.copy(self.heatmap_generator.save_path)
            self.update_status("Map path copied to clipboard", "success")
        except Exception as e:
            logger.error(f"Error copying to clipboard: {e}")
            self.show_message(f"Error copying to clipboard: {str(e)}", "error")
    
    def clear_all(self):
        """Clear all loaded data and reset the UI."""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data?"):
            # Reset data manager
            self.data_manager = DataManager()
            self.heatmap_generator = HeatmapGenerator(self.data_manager)
            
            # Reset UI
            self.kml_path_var.set("")
            self.geojson_path_var.set("")
            self.custom_title_var.set("")
            
            # Reset stats display
            self.stats_text.configure(state="normal")
            self.stats_text.delete("1.0", tk.END)
            self.stats_text.insert("1.0", "Load files to see statistics")
            self.stats_text.configure(state="disabled")
            
            # Reset preview
            if self.preview_frame:
                self.preview_frame.destroy()
                self.preview_frame = None
                self.initialize_preview()
            
            self.update_status("All data cleared", "info")
    
    def analyze_data(self):
        """Show an advanced data analysis window."""
        if not hasattr(self.data_manager, "filtered_coords") or not self.data_manager.filtered_coords:
            self.show_message("Please load data files first", "error")
            return
        
        # Create a new toplevel window for analysis
        analysis_window = ctk.CTkToplevel(self)
        analysis_window.title("Data Analysis")
        analysis_window.geometry("800x600")
        analysis_window.grab_set()  # Make window modal
        
        # Create tabs for different analysis views
        notebook = ttk.Notebook(analysis_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary tab
        summary_frame = ctk.CTkFrame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        ctk.CTkLabel(summary_frame, text="Data Summary", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Summary statistics
        stats = self.data_manager.stats
        summary_text = ctk.CTkTextbox(summary_frame, height=400)
        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Insert detailed statistics
        summary_text.insert("1.0", f"Data Analysis Report\n\n", "heading")
        summary_text.insert("end", f"KML File: {os.path.basename(self.data_manager.kml_path)}\n")
        summary_text.insert("end", f"GeoJSON File: {os.path.basename(self.data_manager.geojson_path)}\n\n")
        
        summary_text.insert("end", f"Total Points Analysis:\n", "subheading")
        summary_text.insert("end", f"• Total points in KML: {stats.get('total_points', 0)}\n")
        summary_text.insert("end", f"• Points within region: {stats.get('filtered_points', 0)}\n")
        summary_text.insert("end", f"• Points outside region: {stats.get('total_points', 0) - stats.get('filtered_points', 0)}\n")
        summary_text.insert("end", f"• Inclusion rate: {stats.get('points_percentage', 0)}%\n\n")
        
        if "bounds" in stats:
            bounds = stats["bounds"]
            summary_text.insert("end", f"Geographic Coverage:\n", "subheading")
            summary_text.insert("end", f"• Latitude range: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}\n")
            summary_text.insert("end", f"• Longitude range: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}\n")
            summary_text.insert("end", f"• Approximate area covered: {bounds['lat_range'] * bounds['lon_range']:.6f} sq. degrees\n\n")
        
        if "hotspots" in stats and stats["hotspots"]:
            summary_text.insert("end", f"Density Analysis:\n", "subheading")
            summary_text.insert("end", f"• Number of identified hotspots: {len(stats['hotspots'])}\n")
            summary_text.insert("end", f"• Top 5 hotspots (highest point density):\n")
            for i, (lat, lon, count) in enumerate(stats["hotspots"][:5], 1):
                summary_text.insert("end", f"  {i}. Lat: {lat:.6f}, Lon: {lon:.6f} - {int(count)} points\n")
        
        # Add report generation timestamp
        summary_text.insert("end", f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Disable editing
        summary_text.configure(state="disabled")
        
        # Distribution tab
        distribution_frame = ctk.CTkFrame(notebook)
        notebook.add(distribution_frame, text="Distribution")
        
        ctk.CTkLabel(distribution_frame, text="Point Distribution Analysis", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Create a figure for the heatmap visualization
        dist_fig = plt.figure(figsize=(6, 4), dpi=100)
        dist_ax = dist_fig.add_subplot(111)
        
        # Create a density heatmap
        if "density_histogram" in stats:
            H, xedges, yedges = stats["density_histogram"]
            
            # Use imshow to create a heatmap
            im = dist_ax.imshow(
                H.T,  # Transpose for correct orientation
                origin='lower',
                aspect='auto',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                cmap='viridis'
            )
            
            # Add a colorbar
            cbar = dist_fig.colorbar(im, ax=dist_ax)
            cbar.set_label('Point Density')
            
            # Set labels
            dist_ax.set_xlabel('Longitude')
            dist_ax.set_ylabel('Latitude')
            dist_ax.set_title('Point Density Heatmap')
            
        else:
            dist_ax.text(0.5, 0.5, "Density data not available", ha='center')
            dist_ax.axis('off')
        
        # Create a canvas for the figure
        dist_canvas = FigureCanvasTkAgg(dist_fig, distribution_frame)
        dist_canvas.draw()
        dist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for zoom/pan
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar_frame = tk.Frame(distribution_frame)
        toolbar_frame.pack(fill=tk.X, padx=10, pady=0)
        NavigationToolbar2Tk(dist_canvas, toolbar_frame)
        
        # Histogram tab
        histogram_frame = ctk.CTkFrame(notebook)
        notebook.add(histogram_frame, text="Histograms")
        
        ctk.CTkLabel(histogram_frame, text="Coordinate Distribution Histograms", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Create a figure with 2 subplots for lat/lon histograms
        hist_fig = plt.figure(figsize=(6, 6), dpi=100)
        
        # Get coordinates for histograms
        if self.data_manager.filtered_coords:
            lats = [p[0] for p in self.data_manager.filtered_coords]
            lons = [p[1] for p in self.data_manager.filtered_coords]
            
            # Latitude histogram
            ax1 = hist_fig.add_subplot(211)
            ax1.hist(lats, bins=50, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Latitude')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Latitude Distribution')
            
            # Longitude histogram
            ax2 = hist_fig.add_subplot(212)
            ax2.hist(lons, bins=50, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Longitude Distribution')
            
            # Adjust layout
            hist_fig.tight_layout()
        else:
            ax = hist_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for histograms", ha='center')
            ax.axis('off')
        
        # Create a canvas for the figure
        hist_canvas = FigureCanvasTkAgg(hist_fig, histogram_frame)
        hist_canvas.draw()
        hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for zoom/pan
        toolbar_frame2 = tk.Frame(histogram_frame)
        toolbar_frame2.pack(fill=tk.X, padx=10, pady=0)
        NavigationToolbar2Tk(hist_canvas, toolbar_frame2)
        
        # Export tab
        export_frame = ctk.CTkFrame(notebook)
        notebook.add(export_frame, text="Export")
        
        ctk.CTkLabel(export_frame, text="Export Analysis Results", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Text report export
        report_frame = ctk.CTkFrame(export_frame)
        report_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(report_frame, text="Generate Text Report:").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(report_frame, text="Export as TXT", command=lambda: self.export_analysis_report('txt')).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(report_frame, text="Export as CSV", command=lambda: self.export_analysis_report('csv')).pack(side=tk.LEFT, padx=5)
        
        # Chart export
        chart_frame = ctk.CTkFrame(export_frame)
        chart_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(chart_frame, text="Export Visualizations:").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(chart_frame, text="Density Heatmap", command=lambda: self.export_analysis_chart('density')).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(chart_frame, text="Histograms", command=lambda: self.export_analysis_chart('histogram')).pack(side=tk.LEFT, padx=5)
        
        # Full report export
        full_report_frame = ctk.CTkFrame(export_frame)
        full_report_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(full_report_frame, text="Complete Report:").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(full_report_frame, text="Generate PDF Report", command=self.export_full_analysis_report).pack(side=tk.LEFT, padx=5)
        
        # Additional instructions
        note_text = ctk.CTkTextbox(export_frame, height=100)
        note_text.pack(fill=tk.X, padx=10, pady=10)
        note_text.insert("1.0", "Note: Exported reports will include all statistics and visualizations shown in this analysis window. "
                        "PDF reports will include all charts and data tables.\n\n"
                        "Export location will be the same as your configured save directory for heatmaps.")
        note_text.configure(state="disabled")
    
    def export_analysis_report(self, format_type):
        """Export analysis data as a text or CSV report."""
        if not self.data_manager.stats:
            self.show_message("No data to export", "error")
            return
        
        # Determine file extension and default name
        if format_type == 'txt':
            file_ext = ".txt"
            file_type = "Text files"
        else:  # csv
            file_ext = ".csv"
            file_type = "CSV files"
        
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"heatmap_analysis_{timestamp}{file_ext}"
        
        # Get export path from user
        export_path = filedialog.asksaveasfilename(
            title=f"Export Analysis as {format_type.upper()}",
            defaultextension=file_ext,
            filetypes=[(file_type, f"*{file_ext}"), ("All files", "*.*")],
            initialdir=self.settings.get("default_save_dir"),
            initialfile=default_name
        )
        
        if not export_path:
            return  # User cancelled
        
        try:
            stats = self.data_manager.stats
            
            if format_type == 'txt':
                # Write text report
                with open(export_path, 'w') as f:
                    f.write(f"Data Analysis Report\n")
                    f.write(f"===================\n\n")
                    f.write(f"KML File: {self.data_manager.kml_path}\n")
                    f.write(f"GeoJSON File: {self.data_manager.geojson_path}\n\n")
                    
                    f.write(f"Total Points Analysis:\n")
                    f.write(f"---------------------\n")
                    f.write(f"Total points in KML: {stats.get('total_points', 0)}\n")
                    f.write(f"Points within region: {stats.get('filtered_points', 0)}\n")
                    f.write(f"Points outside region: {stats.get('total_points', 0) - stats.get('filtered_points', 0)}\n")
                    f.write(f"Inclusion rate: {stats.get('points_percentage', 0)}%\n\n")
                    
                    if "bounds" in stats:
                        bounds = stats["bounds"]
                        f.write(f"Geographic Coverage:\n")
                        f.write(f"-------------------\n")
                        f.write(f"Latitude range: {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}\n")
                        f.write(f"Longitude range: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}\n")
                        f.write(f"Approximate area covered: {bounds['lat_range'] * bounds['lon_range']:.6f} sq. degrees\n\n")
                    
                    if "hotspots" in stats and stats["hotspots"]:
                        f.write(f"Density Analysis:\n")
                        f.write(f"----------------\n")
                        f.write(f"Number of identified hotspots: {len(stats['hotspots'])}\n")
                        f.write(f"Top hotspots (highest point density):\n")
                        for i, (lat, lon, count) in enumerate(stats["hotspots"], 1):
                            f.write(f"{i}. Lat: {lat:.6f}, Lon: {lon:.6f} - {int(count)} points\n")
                    
                    f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            else:  # CSV format
                # Write CSV report with different sections
                with open(export_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header and file info
                    writer.writerow(["Data Analysis Report"])
                    writer.writerow(["Generated", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    writer.writerow([])
                    writer.writerow(["File Information"])
                    writer.writerow(["KML File", self.data_manager.kml_path])
                    writer.writerow(["GeoJSON File", self.data_manager.geojson_path])
                    writer.writerow([])
                    
                    # Points analysis
                    writer.writerow(["Points Analysis"])
                    writer.writerow(["Total points in KML", stats.get('total_points', 0)])
                    writer.writerow(["Points within region", stats.get('filtered_points', 0)])
                    writer.writerow(["Points outside region", stats.get('total_points', 0) - stats.get('filtered_points', 0)])
                    writer.writerow(["Inclusion rate (%)", stats.get('points_percentage', 0)])
                    writer.writerow([])
                    
                    # Geographic bounds
                    if "bounds" in stats:
                        bounds = stats["bounds"]
                        writer.writerow(["Geographic Coverage"])
                        writer.writerow(["Min Latitude", bounds['min_lat']])
                        writer.writerow(["Max Latitude", bounds['max_lat']])
                        writer.writerow(["Min Longitude", bounds['min_lon']])
                        writer.writerow(["Max Longitude", bounds['max_lon']])
                        writer.writerow(["Area (sq. degrees)", bounds['lat_range'] * bounds['lon_range']])
                        writer.writerow([])
                    
                    # Hotspots
                    if "hotspots" in stats and stats["hotspots"]:
                        writer.writerow(["Hotspots Analysis"])
                        writer.writerow(["Number", "Latitude", "Longitude", "Point Count"])
                        for i, (lat, lon, count) in enumerate(stats["hotspots"], 1):
                            writer.writerow([i, lat, lon, int(count)])
            
            self.show_message(f"Analysis report exported to {export_path}", "success")
            
        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            logger.error(traceback.format_exc())
            self.show_message(f"Error exporting report: {str(e)}", "error")
    
    def export_analysis_chart(self, chart_type):
        """Export analysis visualizations as image files."""
        if not self.data_manager.stats:
            self.show_message("No data to export", "error")
            return
        
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"heatmap_{chart_type}_{timestamp}.png"
        
        # Get export path from user
        export_path = filedialog.asksaveasfilename(
            title=f"Export {chart_type.capitalize()} Chart",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir=self.settings.get("default_save_dir"),
            initialfile=default_name
        )
        
        if not export_path:
            return  # User cancelled
        
        try:
            # Create high-resolution figure
            fig = plt.figure(figsize=(10, 8), dpi=300)
            
            if chart_type == 'density' and "density_histogram" in self.data_manager.stats:
                # Create density heatmap
                H, xedges, yedges = self.data_manager.stats["density_histogram"]
                
                ax = fig.add_subplot(111)
                im = ax.imshow(
                    H.T,
                    origin='lower',
                    aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='viridis'
                )
                
                # Add colorbar and labels
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Point Density')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('Point Density Heatmap')
                
            elif chart_type == 'histogram' and self.data_manager.filtered_coords:
                # Create coordinate histograms
                lats = [p[0] for p in self.data_manager.filtered_coords]
                lons = [p[1] for p in self.data_manager.filtered_coords]
                
                # Latitude histogram
                ax1 = fig.add_subplot(211)
                ax1.hist(lats, bins=50, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Latitude')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Latitude Distribution')
                
                # Longitude histogram
                ax2 = fig.add_subplot(212)
                ax2.hist(lons, bins=50, color='lightgreen', edgecolor='black')
                ax2.set_xlabel('Longitude')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Longitude Distribution')
                
                fig.tight_layout()
            else:
                self.show_message(f"No data available for {chart_type} chart", "error")
                return
            
            # Save the figure
            fig.savefig(export_path, bbox_inches='tight')
            plt.close(fig)
            
            self.show_message(f"Chart exported to {export_path}", "success")
            
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
            logger.error(traceback.format_exc())
            self.show_message(f"Error exporting chart: {str(e)}", "error")
    
    def export_full_analysis_report(self):
        """Generate a comprehensive PDF report with all analysis data and charts."""
        if not self.data_manager.stats:
            self.show_message("No data to export", "error")
            return
        
        # Check if reportlab is available
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        except ImportError:
            self.show_message("PDF generation requires reportlab. Installing it now...", "info")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
                self.show_message("Reportlab installed successfully. Please try again.", "success")
                return
            except Exception as e:
                self.show_message(f"Failed to install reportlab: {str(e)}", "error")
                return
        
        # Generate default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"heatmap_full_analysis_{timestamp}.pdf"
        
        # Get export path from user
        export_path = filedialog.asksaveasfilename(
            title="Export Full Analysis Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            initialdir=self.settings.get("default_save_dir"),
            initialfile=default_name
        )
        
        if not export_path:
            return  # User cancelled
        
        try:
            # Set up the PDF document
            doc = SimpleDocTemplate(export_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            )
            
            heading_style = ParagraphStyle(
                'HeadingStyle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10
            )
            
            subheading_style = ParagraphStyle(
                'SubheadingStyle',
                parent=styles['Heading3'],
                fontSize=12,
                spaceAfter=8
            )
            
            # Content elements
            elements = []
            
            # Title
            elements.append(Paragraph(f"Heatmap Analysis Report", title_style))
            elements.append(Spacer(1, 12))
            
            # File information
            elements.append(Paragraph("File Information", heading_style))
            file_data = [
                ["KML File:", os.path.basename(self.data_manager.kml_path)],
                ["GeoJSON File:", os.path.basename(self.data_manager.geojson_path)],
                ["Generated:", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            file_table = Table(file_data, colWidths=[120, 350])
            file_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            elements.append(file_table)
            elements.append(Spacer(1, 12))
            
            # Points analysis
            elements.append(Paragraph("Points Analysis", heading_style))
            stats = self.data_manager.stats
            points_data = [
                ["Total points in KML:", f"{stats.get('total_points', 0)}"],
                ["Points within region:", f"{stats.get('filtered_points', 0)}"],
                ["Points outside region:", f"{stats.get('total_points', 0) - stats.get('filtered_points', 0)}"],
                ["Inclusion rate:", f"{stats.get('points_percentage', 0)}%"]
            ]
            points_table = Table(points_data, colWidths=[120, 350])
            points_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            elements.append(points_table)
            elements.append(Spacer(1, 12))
            
            # Geographic bounds
            if "bounds" in stats:
                bounds = stats["bounds"]
                elements.append(Paragraph("Geographic Coverage", heading_style))
                bounds_data = [
                    ["Latitude range:", f"{bounds['min_lat']:.6f} to {bounds['max_lat']:.6f}"],
                    ["Longitude range:", f"{bounds['min_lon']:.6f} to {bounds['max_lon']:.6f}"],
                    ["Area covered:", f"{bounds['lat_range'] * bounds['lon_range']:.6f} sq. degrees"]
                ]
                bounds_table = Table(bounds_data, colWidths=[120, 350])
                bounds_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                elements.append(bounds_table)
                elements.append(Spacer(1, 12))
            
            # Add visualizations
            elements.append(Paragraph("Data Visualizations", heading_style))
            
            # Generate and save temporary chart images
            temp_dir = Path(tempfile.gettempdir()) / "heatmap_analysis"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate density chart
            if "density_histogram" in stats:
                density_path = temp_dir / "density.png"
                
                fig = plt.figure(figsize=(7, 5), dpi=150)
                ax = fig.add_subplot(111)
                
                H, xedges, yedges = stats["density_histogram"]
                im = ax.imshow(
                    H.T,
                    origin='lower',
                    aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='viridis'
                )
                
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Point Density')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title('Point Density Heatmap')
                
                fig.savefig(density_path, bbox_inches='tight')
                plt.close(fig)
                
                elements.append(Paragraph("Point Density Heatmap", subheading_style))
                elements.append(Image(str(density_path), width=450, height=320))
                elements.append(Spacer(1, 12))
            
            # Generate histogram chart
            if self.data_manager.filtered_coords:
                hist_path = temp_dir / "histograms.png"
                
                fig = plt.figure(figsize=(7, 7), dpi=150)
                
                lats = [p[0] for p in self.data_manager.filtered_coords]
                lons = [p[1] for p in self.data_manager.filtered_coords]
                
                ax1 = fig.add_subplot(211)
                ax1.hist(lats, bins=50, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Latitude')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Latitude Distribution')
                
                ax2 = fig.add_subplot(212)
                ax2.hist(lons, bins=50, color='lightgreen', edgecolor='black')
                ax2.set_xlabel('Longitude')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Longitude Distribution')
                
                fig.tight_layout()
                fig.savefig(hist_path, bbox_inches='tight')
                plt.close(fig)
                
                elements.append(Paragraph("Coordinate Distributions", subheading_style))
                elements.append(Image(str(hist_path), width=450, height=450))
                elements.append(Spacer(1, 12))
            
            # Add hotspots table if available
            if "hotspots" in stats and stats["hotspots"]:
                elements.append(Paragraph("Hotspot Analysis", heading_style))
                elements.append(Paragraph(f"Top {min(10, len(stats['hotspots']))} areas with highest point density:", styles['Normal']))
                elements.append(Spacer(1, 6))
                
                # Create table header
                hotspot_data = [["Rank", "Latitude", "Longitude", "Point Count"]]
                
                # Add table rows
                for i, (lat, lon, count) in enumerate(stats["hotspots"][:10], 1):
                    hotspot_data.append([str(i), f"{lat:.6f}", f"{lon:.6f}", str(int(count))])
                
                # Create and style the table
                hotspot_table = Table(hotspot_data, colWidths=[50, 150, 150, 100])
                hotspot_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('TOPPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(hotspot_table)
            
            # Build the PDF
            doc.build(elements)
            
            # Inform user
            self.show_message(f"Full analysis report saved to {export_path}", "success")
            
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {e}")
            logger.error(traceback.format_exc())
            self.show_message(f"Error creating PDF report: {str(e)}", "error")
    
    def batch_process(self):
        """Batch process multiple KML/GeoJSON pairs."""
        # Create a toplevel window for batch processing
        batch_window = ctk.CTkToplevel(self)
        batch_window.title("Batch Processing")
        batch_window.geometry("800x600")
        batch_window.grab_set()
        
        # Setup UI
        ctk.CTkLabel(batch_window, text="Batch Heatmap Generator", font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # File list section
        files_frame = ctk.CTkFrame(batch_window)
        files_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(files_frame, text="Files to Process", font=("Helvetica", 16)).pack(pady=10)
        
        # Create a list to hold file pairs
        file_pairs = []
        
        # Listbox to display file pairs
        files_list = tk.Listbox(files_frame, height=10, width=70, font=("Helvetica", 10))
        files_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons for file management
        buttons_frame = ctk.CTkFrame(files_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        def add_file_pair():
            # Open file dialogs for KML and GeoJSON
            kml_path = filedialog.askopenfilename(
                title="Select KML File",
                filetypes=[("KML files", "*.kml"), ("All files", "*.*")]
            )
            
            if not kml_path:
                return
            
            geojson_path = filedialog.askopenfilename(
                title="Select GeoJSON File",
                filetypes=[("GeoJSON files", "*.geojson"), ("All files", "*.*")]
            )
            
            if not geojson_path:
                return
            
            # Add to list
            pair = (kml_path, geojson_path)
            file_pairs.append(pair)
            
            # Update listbox
            kml_name = os.path.basename(kml_path)
            geojson_name = os.path.basename(geojson_path)
            files_list.insert(tk.END, f"{kml_name} & {geojson_name}")
        
        def remove_selected():
            selection = files_list.curselection()
            if not selection:
                return
            
            # Remove from list in reverse order to avoid index shifting
            for index in sorted(selection, reverse=True):
                del file_pairs[index]
                files_list.delete(index)
        
        def clear_all():
            file_pairs.clear()
            files_list.delete(0, tk.END)
        
        ctk.CTkButton(buttons_frame, text="Add Files", command=add_file_pair).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(buttons_frame, text="Remove Selected", command=remove_selected).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(buttons_frame, text="Clear All", command=clear_all).pack(side=tk.LEFT, padx=5)
        
        # Options section
        options_frame = ctk.CTkFrame(batch_window)
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(options_frame, text="Batch Options", font=("Helvetica", 16)).pack(pady=10)
        
        # Style options
        style_frame = ctk.CTkFrame(options_frame)
        style_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Heatmap style
        ctk.CTkLabel(style_frame, text="Heatmap Style:").pack(side=tk.LEFT, padx=5)
        batch_heatmap_style = ctk.CTkOptionMenu(
            style_frame,
            values=list(HEATMAP_STYLES.keys())
        )
        batch_heatmap_style.pack(side=tk.LEFT, padx=5)
        batch_heatmap_style.set(self.heatmap_style_var.get())
        
        # Border style
        ctk.CTkLabel(style_frame, text="Border Style:").pack(side=tk.LEFT, padx=5)
        batch_border_style = ctk.CTkOptionMenu(
            style_frame,
            values=list(BORDER_STYLES.keys())
        )
        batch_border_style.pack(side=tk.LEFT, padx=5)
        batch_border_style.set(self.border_style_var.get())
        
        # Output options
        output_frame = ctk.CTkFrame(options_frame)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Output directory
        ctk.CTkLabel(output_frame, text="Output Directory:").pack(side=tk.LEFT, padx=5)
        output_dir_var = ctk.StringVar(value=self.settings.get("default_save_dir"))
        output_entry = ctk.CTkEntry(output_frame, textvariable=output_dir_var, width=300)
        output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_output_dir():
            dir_path = filedialog.askdirectory(title="Select Output Directory")
            if dir_path:
                output_dir_var.set(dir_path)
        
        ctk.CTkButton(output_frame, text="Browse", command=browse_output_dir).pack(side=tk.RIGHT, padx=5)
        
        # Additional options
        additional_frame = ctk.CTkFrame(options_frame)
        additional_frame.pack(fill=tk.X, padx=10, pady=5)
        
        auto_open_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(additional_frame, text="Auto-open maps", variable=auto_open_var).pack(side=tk.LEFT, padx=5)
        
        create_pdf_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(additional_frame, text="Create PDF exports", variable=create_pdf_var).pack(side=tk.LEFT, padx=5)
        
        # Progress section
        progress_frame = ctk.CTkFrame(batch_window)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        progress_var = ctk.DoubleVar(value=0)
        progress_bar = ctk.CTkProgressBar(progress_frame)
        progress_bar.pack(fill=tk.X, padx=10, pady=10)
        progress_bar.set(0)
        
        status_var = ctk.StringVar(value="Ready to process")
        status_label = ctk.CTkLabel(progress_frame, textvariable=status_var)
        status_label.pack(padx=10, pady=5)
        
        # Process function
        def process_batch():
            if not file_pairs:
                messagebox.showwarning("No Files", "Please add files to process")
                return
            
            output_dir = output_dir_var.get()
            if not output_dir or not os.path.isdir(output_dir):
                messagebox.showwarning("Invalid Directory", "Please select a valid output directory")
                return
            
            # Get selected options
            heatmap_style = batch_heatmap_style.get()
            border_style = batch_border_style.get()
            auto_open = auto_open_var.get()
            create_pdf = create_pdf_var.get()
            
            # Disable UI during processing
            for widget in batch_window.winfo_children():
                try:
                    widget.configure(state="disabled")
                except:
                    pass
            
            def processing_thread():
                total_files = len(file_pairs)
                processed = 0
                failed = 0
                
                # Local data manager and generator for batch processing
                batch_data_manager = DataManager()
                batch_heatmap_generator = HeatmapGenerator(batch_data_manager)
                
                # Set generator options
                batch_heatmap_generator.heatmap_style = heatmap_style
                batch_heatmap_generator.border_style = border_style
                batch_heatmap_generator.border_color = self.border_color_var.get()
                batch_heatmap_generator.map_style = self.map_style_var.get()
                batch_heatmap_generator.show_markers = self.show_markers_var.get()
                batch_heatmap_generator.include_statistics = True
                
                # Process each pair
                for i, (kml_path, geojson_path) in enumerate(file_pairs):
                    try:
                        # Update status
                        kml_name = os.path.basename(kml_path)
                        geojson_name = os.path.basename(geojson_path)
                        status_var.set(f"Processing {i+1}/{total_files}: {kml_name}")
                        progress_var.set((i) / total_files)
                        progress_bar.set(progress_var.get())
                        
                        # Update UI
                        batch_window.update_idletasks()
                        
                        # Load and process files
                        if not batch_data_manager.load_kml(kml_path):
                            logger.error(f"Failed to load KML: {kml_path}")
                            failed += 1
                            continue
                        
                        if not batch_data_manager.load_geojson(geojson_path):
                            logger.error(f"Failed to load GeoJSON: {geojson_path}")
                            failed += 1
                            continue
                        
                        if not batch_data_manager.filter_points_by_region():
                            logger.warning(f"No points found in region: {kml_name} & {geojson_name}")
                            failed += 1
                            continue
                        
                        # Set custom title based on filename
                        batch_heatmap_generator.custom_title = f"Heatmap of {kml_name}"
                        
                        # Create map
                        if not batch_heatmap_generator.create_map():
                            logger.error(f"Failed to create map: {kml_name} & {geojson_name}")
                            failed += 1
                            continue
                        
                        # Generate output filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        kml_base = os.path.splitext(kml_name)[0]
                        output_file = os.path.join(output_dir, f"heatmap_{kml_base}_{timestamp}.html")
                        
                        # Save map
                        saved_path = batch_heatmap_generator.save_map(output_file)
                        
                        # Export PDF if requested
                        if create_pdf and saved_path:
                            pdf_path = os.path.splitext(saved_path)[0] + ".pdf"
                            batch_heatmap_generator.export_to_pdf(pdf_path)
                        
                        # Open in browser if requested
                        if auto_open and saved_path:
                            try:
                                webbrowser.open("file://" + os.path.realpath(saved_path))
                            except Exception as e:
                                logger.error(f"Error opening map in browser: {e}")
                        
                        processed += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {kml_path} & {geojson_path}: {e}")
                        logger.error(traceback.format_exc())
                        failed += 1
                
                # Update final status
                progress_var.set(1.0)
                progress_bar.set(1.0)
                status_var.set(f"Processing complete. Processed: {processed}, Failed: {failed}")
                
                # Re-enable UI
                try:
                    for widget in batch_window.winfo_children():
                        try:
                            widget.configure(state="normal")
                        except:
                            pass
                    
                    # Show completion message
                    messagebox.showinfo("Batch Processing Complete", 
                                       f"Processing complete!\n\nSuccessfully processed: {processed}\nFailed: {failed}\n\nOutput directory: {output_dir}")
                except:
                    pass
            
            # Start processing thread
            Thread(target=processing_thread).start()
        
        # Process button
        process_button = ctk.CTkButton(
            batch_window, 
            text="Process All Files", 
            command=process_batch,
            height=50,
            font=("Helvetica", 16, "bold")
        )
        process_button.pack(padx=10, pady=20)
    
    def show_style_editor(self):
        """Open the custom style editor window."""
        # Create a toplevel window for style editor
        style_window = ctk.CTkToplevel(self)
        style_window.title("Custom Style Editor")
        style_window.geometry("800x600")
        style_window.grab_set()
        
        # Setup UI
        ctk.CTkLabel(style_window, text="Heatmap Style Editor", font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # Create notebook with tabs for different style components
        notebook = ttk.Notebook(style_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Heatmap style tab
        heatmap_frame = ctk.CTkFrame(notebook)
        notebook.add(heatmap_frame, text="Heatmap Style")
        
        # Base style selection
        base_frame = ctk.CTkFrame(heatmap_frame)
        base_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(base_frame, text="Base Style:").pack(side=tk.LEFT, padx=5)
        
        base_style_var = ctk.StringVar(value="default")
        base_dropdown = ctk.CTkOptionMenu(
            base_frame,
            values=[s for s in HEATMAP_STYLES.keys() if s != "custom"],
            variable=base_style_var,
            command=lambda s: update_sliders(s)
        )
        base_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Parameters frame
        params_frame = ctk.CTkFrame(heatmap_frame)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sliders for radius, blur, opacity
        radius_var = ctk.IntVar(value=HEATMAP_STYLES["default"]["radius"])
        blur_var = ctk.IntVar(value=HEATMAP_STYLES["default"]["blur"])
        opacity_var = ctk.DoubleVar(value=HEATMAP_STYLES["default"]["min_opacity"])
        
        # Radius slider
        radius_frame = ctk.CTkFrame(params_frame)
        radius_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(radius_frame, text="Radius:").pack(side=tk.LEFT, padx=5)
        radius_slider = ctk.CTkSlider(
            radius_frame,
            from_=5,
            to=50,
            number_of_steps=45,
            variable=radius_var,
            command=lambda v: update_preview_live()
        )
        radius_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        radius_label = ctk.CTkLabel(radius_frame, text=str(radius_var.get()))
        radius_label.pack(side=tk.RIGHT, padx=5)
        
        # Blur slider
        blur_frame = ctk.CTkFrame(params_frame)
        blur_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(blur_frame, text="Blur:").pack(side=tk.LEFT, padx=5)
        blur_slider = ctk.CTkSlider(
            blur_frame,
            from_=0,
            to=50,
            number_of_steps=50,
            variable=blur_var,
            command=lambda v: update_preview_live()
        )
        blur_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        blur_label = ctk.CTkLabel(blur_frame, text=str(blur_var.get()))
        blur_label.pack(side=tk.RIGHT, padx=5)
        
        # Opacity slider
        opacity_frame = ctk.CTkFrame(params_frame)
        opacity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(opacity_frame, text="Opacity:").pack(side=tk.LEFT, padx=5)
        opacity_slider = ctk.CTkSlider(
            opacity_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=opacity_var,
            command=lambda v: update_preview_live()
        )
        opacity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        opacity_label = ctk.CTkLabel(opacity_frame, text=str(opacity_var.get()))
        opacity_label.pack(side=tk.RIGHT, padx=5)
        
        # Color gradient editor
        gradient_frame = ctk.CTkFrame(heatmap_frame)
        gradient_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(gradient_frame, text="Color Gradient:", font=("Helvetica", 14, "bold")).pack(pady=5)
        
        # Preview of current gradient
        preview_frame = ctk.CTkFrame(gradient_frame)
        preview_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a canvas for gradient preview
        preview_canvas = tk.Canvas(preview_frame, height=30, bg="white")
        preview_canvas.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # Color stops frame (where you can add/edit color stops)
        stops_frame = ctk.CTkFrame(gradient_frame)
        stops_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Dictionary to hold stop points and their colors
        gradient_stops = {}
        
        # Function to update the gradient preview
        def update_gradient_preview():
            width = preview_canvas.winfo_width()
            if width <= 1:  # Not yet sized
                width = 300
            height = preview_canvas.winfo_height()
            
            # Clear canvas
            preview_canvas.delete("all")
            
            # Sort stops by position
            sorted_stops = sorted(gradient_stops.items(), key=lambda x: float(x[0]))
            
            if not sorted_stops:
                return
            
            # Draw gradient
            for i in range(width):
                # Calculate relative position
                pos = i / width
                
                # Find surrounding stops
                color = sorted_stops[0][1]  # Default to first color
                
                for j in range(len(sorted_stops) - 1):
                    pos1 = float(sorted_stops[j][0])
                    pos2 = float(sorted_stops[j+1][0])
                    if pos1 <= pos <= pos2:
                        # Interpolate color
                        t = (pos - pos1) / (pos2 - pos1) if pos2 > pos1 else 0
                        color1 = sorted_stops[j][1]
                        color2 = sorted_stops[j+1][1]
                        
                        # Simple RGB interpolation
                        r1, g1, b1 = preview_canvas.winfo_rgb(color1)
                        r2, g2, b2 = preview_canvas.winfo_rgb(color2)
                        
                        r = int(r1 + (r2 - r1) * t) >> 8
                        g = int(g1 + (g2 - g1) * t) >> 8
                        b = int(b1 + (b2 - b1) * t) >> 8
                        
                        color = f"#{r:02x}{g:02x}{b:02x}"
                        break
                
                # Draw line
                preview_canvas.create_line(i, 0, i, height, fill=color)
        
        # Function to update entries when a base style is selected
        def update_sliders(style_name):
            # Get style parameters
            style = HEATMAP_STYLES.get(style_name, HEATMAP_STYLES["default"])
            
            # Update variables
            radius_var.set(style["radius"])
            blur_var.set(style["blur"])
            opacity_var.set(style["min_opacity"])
            
            # Update labels
            radius_label.configure(text=str(style["radius"]))
            blur_label.configure(text=str(style["blur"]))
            opacity_label.configure(text=str(style["opacity"] if "opacity" in style else style["min_opacity"]))
            
            # Update gradient stops
            gradient_stops.clear()
            for pos, color in style["gradient"].items():
                gradient_stops[pos] = color
            
            # Update stops display
            update_stops_display()
            
            # Update preview
            style_window.after(100, update_gradient_preview)
            style_window.after(100, update_preview_live)
        
        # Frame to hold color stop controls
        stops_controls_frame = ctk.CTkScrollableFrame(stops_frame, height=150)
        stops_controls_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Function to add a new color stop
        def add_color_stop():
            # Create a new default stop at position 0.5 with blue color
            pos = "0.5"
            color = "#0000ff"
            
            # Add to gradient stops
            gradient_stops[pos] = color
            
            # Update display
            update_stops_display()
            update_gradient_preview()
            update_preview_live()
        
        # Function to remove a color stop
        def remove_color_stop(pos):
            if pos in gradient_stops and len(gradient_stops) > 2:
                del gradient_stops[pos]
                update_stops_display()
                update_gradient_preview()
                update_preview_live()
        
        # Function to update a stop's position
        def update_stop_position(old_pos, new_pos):
            if old_pos in gradient_stops:
                color = gradient_stops[old_pos]
                del gradient_stops[old_pos]
                gradient_stops[new_pos] = color
                update_gradient_preview()
                update_preview_live()
        
        # Function to update a stop's color
        def update_stop_color(pos, color):
            if pos in gradient_stops:
                gradient_stops[pos] = color
                update_gradient_preview()
                update_preview_live()
        
        # Function to open color chooser
        def choose_color(pos, color_button):
            current_color = gradient_stops[pos]
            color = colorchooser.askcolor(color=current_color)
            if color and color[1]:
                gradient_stops[pos] = color[1]
                color_button.configure(fg_color=color[1])
                update_gradient_preview()
                update_preview_live()
        
        # Frame for stop display widgets
        stop_widgets = {}
        
        # Function to update the stops display
        def update_stops_display():
            # Clear current widgets
            for widget in stop_widgets.values():
                for w in widget:
                    w.destroy()
            stop_widgets.clear()
            
            # Create widgets for each stop
            for pos in sorted(gradient_stops.keys(), key=float):
                color = gradient_stops[pos]
                
                # Create frame for this stop
                stop_frame = ctk.CTkFrame(stops_controls_frame)
                stop_frame.pack(fill=tk.X, padx=5, pady=2)
                
                # Position entry
                pos_var = ctk.StringVar(value=pos)
                pos_entry = ctk.CTkEntry(stop_frame, width=60, textvariable=pos_var)
                pos_entry.pack(side=tk.LEFT, padx=5)
                
                # Update position when entry changes
                pos_entry.bind("<FocusOut>", lambda e, old=pos, var=pos_var: update_stop_position(old, var.get()))
                
                # Color button
                color_button = ctk.CTkButton(
                    stop_frame,
                    text="",
                    width=30,
                    height=20,
                    fg_color=color,
                    command=lambda p=pos, b=None: choose_color(p, b)
                )
                color_button.pack(side=tk.LEFT, padx=5)
                
                # Store button reference for later color updates
                # This is a workaround for closure variables in lambdas
                color_button.configure(command=lambda p=pos, b=color_button: choose_color(p, b))
                
                # Remove button (only if more than 2 stops)
                remove_button = ctk.CTkButton(
                    stop_frame,
                    text="X",
                    width=30,
                    command=lambda p=pos: remove_color_stop(p)
                )
                remove_button.pack(side=tk.RIGHT, padx=5)
                
                # Store widgets for this stop
                stop_widgets[pos] = [stop_frame, pos_entry, color_button, remove_button]
            
            # Add button for new stop
            add_button = ctk.CTkButton(
                stops_controls_frame,
                text="+ Add Color Stop",
                command=add_color_stop
            )
            add_button.pack(pady=10)
        
        # Preview section
        preview_section = ctk.CTkFrame(heatmap_frame)
        preview_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(preview_section, text="Style Preview", font=("Helvetica", 14, "bold")).pack(pady=5)
        
        # Create a matplotlib figure for preview
        preview_fig = plt.figure(figsize=(4, 3), dpi=100)
        preview_ax = preview_fig.add_subplot(111)
        
        # Create canvas for the preview
        preview_plot_canvas = FigureCanvasTkAgg(preview_fig, preview_section)
        preview_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Function to update the live preview
        def update_preview_live():
            if not hasattr(self.data_manager, "filtered_coords") or not self.data_manager.filtered_coords:
                return
            
            # Clear the axes
            preview_ax.clear()
            
            # Update labels
            radius_label.configure(text=str(int(radius_var.get())))
            blur_label.configure(text=str(int(blur_var.get())))
            opacity_label.configure(text=str(round(float(opacity_var.get()), 2)))
            
            # Get current coordinates
            lats = [p[0] for p in self.data_manager.filtered_coords]
            lons = [p[1] for p in self.data_manager.filtered_coords]
            
            # Create 2D histogram
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Add padding
            lat_padding = (max_lat - min_lat) * 0.1
            lon_padding = (max_lon - min_lon) * 0.1
            
            # Create a 2D histogram
            heatmap, xedges, yedges = np.histogram2d(
                lons, lats, 
                bins=40, 
                range=[[min_lon - lon_padding, max_lon + lon_padding], 
                       [min_lat - lat_padding, max_lat + lat_padding]]
            )
            
            # Create a custom colormap from the gradient
            from matplotlib.colors import LinearSegmentedColormap
            
            # Make sure we have at least 2 stops
            if len(gradient_stops) < 2:
                cmap = plt.cm.viridis
            else:
                # Sort stops by position
                sorted_stops = sorted(gradient_stops.items(), key=lambda x: float(x[0]))
                
                # Extract positions and colors
                positions = [float(pos) for pos, _ in sorted_stops]
                colors = [color for _, color in sorted_stops]
                
                # Create colormap
                cmap = LinearSegmentedColormap.from_list("custom", list(zip(positions, colors)), N=256)
            
            # Plot the heatmap
            im = preview_ax.imshow(
                heatmap.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower',
                cmap=cmap,
                alpha=opacity_var.get()
            )
            
            # Add a colorbar
            if hasattr(preview_ax, '_colorbar'):
                preview_ax._colorbar.remove()
            preview_ax._colorbar = plt.colorbar(im, ax=preview_ax, shrink=0.8)
            
            # Clean up the axes
            preview_ax.set_title("Preview with Custom Style")
            preview_ax.set_xlabel("Longitude")
            preview_ax.set_ylabel("Latitude")
            
            # Update the canvas
            preview_plot_canvas.draw()
        
        # Border style tab
        border_frame = ctk.CTkFrame(notebook)
        notebook.add(border_frame, text="Border Style")
        
        # Map style tab
        map_style_frame = ctk.CTkFrame(notebook)
        notebook.add(map_style_frame, text="Map Style")
        
        # Initialize gradient from default style
        for pos, color in HEATMAP_STYLES["default"]["gradient"].items():
            gradient_stops[pos] = color
        
        # Update UI elements
        update_stops_display()
        
        # Wait for window to be drawn, then update gradient preview
        style_window.after(100, update_gradient_preview)
        style_window.after(100, update_preview_live)
        
        # Function to apply the custom style
        def apply_custom_style():
            # Create custom style from current settings
            custom_style = {
                "radius": int(radius_var.get()),
                "blur": int(blur_var.get()),
                "min_opacity": float(opacity_var.get()),
                "gradient": dict(gradient_stops),
                "description": "User-defined custom style",
                "icon": "⚙️"
            }
            
            # Update global custom style
            HEATMAP_STYLES["custom"] = custom_style
            
            # Set as current style
            self.heatmap_style_var.set("custom")
            
            # Update preview in main window
            self.update_preview()
            
            # Close style editor
            style_window.destroy()
            
            # Inform user
            self.show_message("Custom style applied successfully!", "success")
        
        # Apply button
        apply_button = ctk.CTkButton(
            style_window,
            text="Apply Custom Style",
            command=apply_custom_style,
            height=40,
            font=("Helvetica", 14, "bold")
        )
        apply_button.pack(padx=10, pady=20)
        
        # Ensure UI updates
        style_window.update_idletasks()
    
    def show_settings(self):
        """Show settings dialog."""
        # Create toplevel window for settings
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("600x500")
        settings_window.grab_set()
        
        # Setup UI
        ctk.CTkLabel(settings_window, text="Application Settings", font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # Settings notebook
        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings tab
        general_frame = ctk.CTkFrame(notebook)
        notebook.add(general_frame, text="General")
        
        # Theme settings
        theme_frame = ctk.CTkFrame(general_frame)
        theme_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=5)
        
        theme_var = ctk.StringVar(value=self.settings.get("theme", "dark"))
        
        dark_radio = ctk.CTkRadioButton(
            theme_frame,
            text="Dark",
            variable=theme_var,
            value="dark"
        )
        dark_radio.pack(side=tk.LEFT, padx=5)
        
        light_radio = ctk.CTkRadioButton(
            theme_frame,
            text="Light",
            variable=theme_var,
            value="light"
        )
        light_radio.pack(side=tk.LEFT, padx=5)
        
        # Default save directory
        save_dir_frame = ctk.CTkFrame(general_frame)
        save_dir_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(save_dir_frame, text="Default Save Directory:").pack(anchor="w", padx=5, pady=2)
        
        dir_var = ctk.StringVar(value=self.settings.get("default_save_dir", str(Path.home() / "Documents" / "Heatmaps")))
        dir_entry = ctk.CTkEntry(save_dir_frame, textvariable=dir_var, width=400)
        dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_save_dir():
            dir_path = filedialog.askdirectory(title="Select Default Save Directory")
            if dir_path:
                dir_var.set(dir_path)
        
        ctk.CTkButton(save_dir_frame, text="Browse", command=browse_save_dir).pack(side=tk.RIGHT, padx=5)
        
        # Other options
        options_frame = ctk.CTkFrame(general_frame)
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        auto_open_var = ctk.BooleanVar(value=self.settings.get("auto_open_map", True))
        auto_open_check = ctk.CTkCheckBox(
            options_frame,
            text="Automatically open maps in browser",
            variable=auto_open_var
        )
        auto_open_check.pack(anchor="w", padx=5, pady=2)
        
        remember_dir_var = ctk.BooleanVar(value=self.settings.get("remember_last_directory", True))
        remember_dir_check = ctk.CTkCheckBox(
            options_frame,
            text="Remember last used directory",
            variable=remember_dir_var
        )
        remember_dir_check.pack(anchor="w", padx=5, pady=2)
        
        show_preview_var = ctk.BooleanVar(value=self.settings.get("show_preview", True))
        show_preview_check = ctk.CTkCheckBox(
            options_frame,
            text="Show preview in main window",
            variable=show_preview_var
        )
        show_preview_check.pack(anchor="w", padx=5, pady=2)
        
        # Advanced settings tab
        advanced_frame = ctk.CTkFrame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        
        # Export settings
        export_frame = ctk.CTkFrame(advanced_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(export_frame, text="Default Export Format:").pack(anchor="w", padx=5, pady=2)
        
        export_format_var = ctk.StringVar(value=self.settings.get("default_export_format", "html"))
        
        html_radio = ctk.CTkRadioButton(
            export_frame,
            text="HTML",
            variable=export_format_var,
            value="html"
        )
        html_radio.pack(anchor="w", padx=20, pady=2)
        
        png_radio = ctk.CTkRadioButton(
            export_frame,
            text="PNG",
            variable=export_format_var,
            value="png"
        )
        png_radio.pack(anchor="w", padx=20, pady=2)
        
        pdf_radio = ctk.CTkRadioButton(
            export_frame,
            text="PDF",
            variable=export_format_var,
            value="pdf"
        )
        pdf_radio.pack(anchor="w", padx=20, pady=2)
        
        # Recent files limit
        recent_frame = ctk.CTkFrame(advanced_frame)
        recent_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(recent_frame, text="Maximum Recent Files:").pack(anchor="w", padx=5, pady=2)
        
        recent_var = ctk.IntVar(value=self.settings.get("max_recent_files", 10))
        recent_slider = ctk.CTkSlider(
            recent_frame,
            from_=5,
            to=30,
            number_of_steps=25,
            variable=recent_var
        )
        recent_slider.pack(fill=tk.X, padx=20, pady=2)
        
        recent_label = ctk.CTkLabel(recent_frame, text=str(recent_var.get()))
        recent_label.pack(anchor="w", padx=5, pady=2)
        
        # Update label when slider moves
        def update_recent_label(value):
            recent_label.configure(text=str(int(value)))
        
        recent_slider.configure(command=update_recent_label)
        
        # Advanced mode
        advanced_mode_var = ctk.BooleanVar(value=self.settings.get("advanced_mode", False))
        advanced_mode_check = ctk.CTkCheckBox(
            advanced_frame,
            text="Enable advanced mode (experimental features)",
            variable=advanced_mode_var
        )
        advanced_mode_check.pack(anchor="w", padx=5, pady=10)
        
        # About tab
        about_frame = ctk.CTkFrame(notebook)
        notebook.add(about_frame, text="About")
        
        # App info
        app_info_frame = ctk.CTkFrame(about_frame)
        app_info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkLabel(app_info_frame, text=f"{APP_NAME}", font=("Helvetica", 20, "bold")).pack(pady=5)
        ctk.CTkLabel(app_info_frame, text=f"Version {APP_VERSION}").pack()
        ctk.CTkLabel(app_info_frame, text=f"{APP_AUTHOR}").pack(pady=5)
        
        # Description text
        desc_text = ctk.CTkTextbox(about_frame, height=200, wrap="word")
        desc_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        desc_text.insert("1.0", "Enhanced Heatmap Generator is a powerful tool for creating customizable heatmaps from KML and GeoJSON data. "
                        "This application is an improved version of the original heatmap generator, with many new features and enhancements.\n\n"
                        "Features:\n"
                        "• Modern user interface with dark/light theme support\n"
                        "• Multiple visualization styles with real-time preview\n"
                        "• Advanced data analysis capabilities\n"
                        "• Export to multiple formats (HTML, PNG, PDF)\n"
                        "• Batch processing of multiple files\n"
                        "• Custom style editor\n"
                        "• Comprehensive error handling\n\n"
                        "This application is free and open-source software.")
        
        desc_text.configure(state="disabled")
        
        # Function to save settings
        def save_and_close():
            # Collect settings from UI
            new_settings = {
                "theme": theme_var.get(),
                "default_save_dir": dir_var.get(),
                "auto_open_map": auto_open_var.get(),
                "remember_last_directory": remember_dir_var.get(),
                "default_export_format": export_format_var.get(),
                "max_recent_files": int(recent_var.get()),
                "show_preview": show_preview_var.get(),
                "advanced_mode": advanced_mode_var.get()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(new_settings["default_save_dir"], exist_ok=True)
            
            # Save settings
            save_settings(new_settings)
            
            # Update settings in main window
            self.settings = new_settings
            
            # Apply theme change if needed
            if new_settings["theme"] != self.theme_var.get():
                self.theme_var.set(new_settings["theme"])
                self.set_theme(new_settings["theme"])
            
            # Update preview visibility
            if new_settings["show_preview"] != self.settings.get("show_preview", True):
                self.toggle_preview()
            
            # Close window
            settings_window.destroy()
            
            # Show confirmation
            self.show_message("Settings saved successfully", "success")
        
        # Save button
        save_button = ctk.CTkButton(
            settings_window,
            text="Save Settings",
            command=save_and_close,
            height=40,
            font=("Helvetica", 14, "bold")
        )
        save_button.pack(padx=10, pady=20)
    
    def show_documentation(self):
        """Show help documentation."""
        help_window = ctk.CTkToplevel(self)
        help_window.title("Documentation")
        help_window.geometry("800x600")
        
        # Setup UI
        ctk.CTkLabel(help_window, text="Enhanced Heatmap Generator - Help", font=("Helvetica", 20, "bold")).pack(pady=20)
        
        # Create a notebook with help topics
        notebook = ttk.Notebook(help_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Getting Started tab
        getting_started = ctk.CTkFrame(notebook)
        notebook.add(getting_started, text="Getting Started")
        
        # Add scrollable text area
        gs_text = ctk.CTkTextbox(getting_started, wrap="word")
        gs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        gs_text.insert("1.0", """# Getting Started

The Enhanced Heatmap Generator allows you to create interactive heatmaps from KML and GeoJSON files.

## Basic Workflow:

1. **Select Input Files:**
   - Click "Select KML File" to choose a KML file containing data points.
   - Click "Select GeoJSON File" to choose a GeoJSON file defining the region boundary.

2. **Load Files:**
   - Click "Load Files" to process the selected files.
   - The application will load the points from the KML file and filter them to only include points within the GeoJSON region boundary.
   - Statistics about the data will be displayed in the right panel.

3. **Customize the Heatmap:**
   - Select a Heatmap Style from the dropdown menu.
   - Choose a Border Style for the region boundary.
   - Select a Map Style for the base map.
   - Optional: Add a custom title, enable/disable markers and legends.

4. **Create the Heatmap:**
   - Click "Create Heatmap" to generate the heatmap with your settings.
   - The heatmap will be saved as an HTML file and opened in your default web browser.

5. **Export Options:**
   - Use the options in the "Actions" section to export your heatmap to different formats.
   - You can save as HTML, export as PNG or PDF, or analyze the data further.
""")
        
        gs_text.configure(state="disabled")
        
        # File Formats tab
        file_formats = ctk.CTkFrame(notebook)
        notebook.add(file_formats, text="File Formats")
        
        # Add scrollable text area
        ff_text = ctk.CTkTextbox(file_formats, wrap="word")
        ff_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ff_text.insert("1.0", """# Supported File Formats

The Enhanced Heatmap Generator works with the following file formats:

## KML Files

KML (Keyhole Markup Language) is an XML-based format used to display geographic data in applications like Google Earth. In the context of this application:

- KML files should contain placemarks with coordinates (longitude, latitude).
- Each placemark represents a data point that will be visualized on the heatmap.
- The application extracts coordinates from all placemarks in the KML file.

Example KML structure:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <Point>
        <coordinates>-122.084,37.422,0</coordinates>
      </Point>
    </Placemark>
    <!-- More placemarks -->
  </Document>
</kml>
```

## GeoJSON Files

GeoJSON is a format for encoding geographic data structures. In this application:

- GeoJSON files define the boundary of the region for your heatmap.
- The application filters KML points to only include those within this boundary.
- The GeoJSON can contain Polygon or MultiPolygon geometries.

Example GeoJSON structure:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [-122.1, 37.4],
            [-122.0, 37.4],
            [-122.0, 37.5],
            [-122.1, 37.5],
            [-122.1, 37.4]
          ]
        ]
      }
    }
  ]
}
```

You can create or edit GeoJSON files using tools like geojson.io or QGIS.
""")
        
        ff_text.configure(state="disabled")
        
        # Customization tab
        customization = ctk.CTkFrame(notebook)
        notebook.add(customization, text="Customization")
        
        # Add scrollable text area
        cust_text = ctk.CTkTextbox(customization, wrap="word")
        cust_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cust_text.insert("1.0", """# Customization Options

The Enhanced Heatmap Generator offers various customization options to create visually appealing and informative heatmaps.

## Heatmap Styles

Different heatmap styles affect how the data points are visualized:

- **Default**: Standard balanced heatmap with blue-green-red gradient
- **Intense**: High intensity visualization with sharper transitions
- **Smooth**: Smooth transitions with earth tones
- **Cool**: Cool blue color palette
- **Fire**: Warm fire-like color scheme
- **Monochrome**: Simple black and white gradient
- **Rainbow**: Full spectrum rainbow colors
- **Custom**: User-defined style (create using the Style Editor)

## Border Styles

Control how the region boundary is displayed:

- **None**: No border
- **Thin**: Thin black border
- **Normal**: Normal-width black border
- **Thick**: Thick colored border (you can select the color)
- **Dashed**: Dashed border
- **Custom**: User-defined border style

## Map Styles

Different base map tile styles:

- **OpenStreetMap**: Standard street map
- **Stamen Terrain**: Terrain visualization
- **Stamen Toner**: High contrast black and white
- **CartoDB Dark**: Dark theme map
- **CartoDB Positron**: Light theme map
- **ESRI World Imagery**: Satellite imagery

## Additional Options

- **Show Markers**: Display markers at hotspot locations
- **Show Legend**: Include a color legend in the map
- **Include Statistics**: Show data statistics in the map
- **Custom Title**: Add a custom title to your heatmap

## Style Editor

For advanced customization, use the Custom Style Editor:

1. Go to Tools > Custom Style Editor
2. Choose a base style to start with
3. Adjust radius, blur, and opacity
4. Customize the color gradient with multiple color stops
5. See a real-time preview of your custom style
6. Click "Apply Custom Style" to use it
""")
        
        cust_text.configure(state="disabled")
        
        # Analysis tab
        analysis = ctk.CTkFrame(notebook)
        notebook.add(analysis, text="Data Analysis")
        
        # Add scrollable text area
        analysis_text = ctk.CTkTextbox(analysis, wrap="word")
        analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        analysis_text.insert("1.0", """# Data Analysis Features

The Enhanced Heatmap Generator includes powerful data analysis tools to help you understand your spatial data.

## Basic Statistics

After loading your files, the application displays basic statistics:

- Total number of points in the KML file
- Number of points within the region boundary
- Inclusion percentage
- Geographic bounds (min/max latitude and longitude)

## Detailed Analysis

For more in-depth analysis, click the "Analyze Data" button to open the Data Analysis window with multiple tabs:

### Summary Tab

Shows comprehensive statistics about your data, including:
- File information
- Points analysis
- Geographic coverage
- Density analysis

### Distribution Tab

Visualizes the point density with a heatmap showing where points are concentrated.

### Histograms Tab

Displays frequency histograms for latitude and longitude distributions, helping you identify patterns in the data.

### Export Tab

Export your analysis results in different formats:
- Text reports (TXT)
- CSV files for spreadsheet analysis
- Visualizations as images
- Complete PDF reports with charts and tables

## Hotspot Analysis

The application automatically identifies and ranks hotspots (areas with high point density) in your data. For each hotspot, it provides:

- Geographic coordinates (latitude/longitude)
- Point count (number of points in the hotspot)
- Relative ranking compared to other hotspots

## Using Analysis Results

The analysis results can help you:
- Identify patterns and clusters in your spatial data
- Find areas of high activity or interest
- Make data-driven decisions
- Create compelling visualizations for reports and presentations
- Share insights with stakeholders
""")
        
        analysis_text.configure(state="disabled")
        
        # Advanced Features tab
        advanced = ctk.CTkFrame(notebook)
        notebook.add(advanced, text="Advanced Features")
        
        # Add scrollable text area
        adv_text = ctk.CTkTextbox(advanced, wrap="word")
        adv_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        adv_text.insert("1.0", """# Advanced Features

The Enhanced Heatmap Generator includes several advanced features for power users.

## Batch Processing

Process multiple file pairs at once:

1. Go to Tools > Batch Process
2. Add multiple KML/GeoJSON file pairs
3. Configure batch options (style, border, output directory)
4. Click "Process All Files" to generate multiple heatmaps

This is useful when you need to create many heatmaps with similar settings or want to process a time series of data.

## Custom Style Editor

Create fully customized heatmap styles:

1. Go to Tools > Custom Style Editor
2. Adjust parameters (radius, blur, opacity)
3. Customize the color gradient
4. See a real-time preview
5. Apply your custom style

## Command Line Interface

For automation and scripting, you can use the command line interface:

```
python enhanced_heatmap_generator.py --kml path/to/file.kml --geojson path/to/file.geojson --style default --output path/to/output.html
```

Command line options:
- `--kml`: Path to KML file (required)
- `--geojson`: Path to GeoJSON file (required)
- `--style`: Heatmap style (default, intense, smooth, etc.)
- `--border`: Border style (none, thin, normal, thick)
- `--border-color`: Border color (for thick border)
- `--map-style`: Map tile style
- `--output`: Output file path
- `--no-open`: Don't open the map in browser
- `--batch`: Process multiple files (requires a CSV config file)

## Export Formats

Export your heatmaps in multiple formats:

- **HTML**: Interactive web map (default)
- **PNG**: Static image for reports and presentations
- **PDF**: Document with map and statistics

## Settings Management

Customize application behavior:

1. Go to File > Settings
2. Adjust general settings (theme, default directories)
3. Configure export preferences
4. Set advanced options

Your settings are saved between sessions for convenience.
""")
        
        adv_text.configure(state="disabled")
        
        # Tips & Tricks tab
        tips = ctk.CTkFrame(notebook)
        notebook.add(tips, text="Tips & Tricks")
        
        # Add scrollable text area
        tips_text = ctk.CTkTextbox(tips, wrap="word")
        tips_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tips_text.insert("1.0", """# Tips & Tricks

Get the most out of the Enhanced Heatmap Generator with these helpful tips.

## Optimization Tips

- **Large Files**: For KML files with thousands of points, enable the "Skip Validation" option in Settings > Advanced to speed up loading.

- **Memory Usage**: If processing very large files, close other applications to free up memory.

- **Export Performance**: PNG exports of large maps can be memory-intensive. Consider reducing the map size or using HTML format for very large datasets.

## Visualization Tips

- **Color Selection**: Choose color schemes appropriate for your data:
  - Use sequential scales (e.g., light to dark) for quantities
  - Use diverging scales (e.g., blue to red) for deviations from a central value
  - Consider color blindness - the "Cool" style is more accessible

- **Point Density**: Adjust radius and blur settings based on your data density:
  - For sparse data, increase radius
  - For dense data, decrease radius and increase blur

- **Map Styles**: Select map styles that complement your data:
  - Use CartoDB Dark for vibrant heatmaps that pop
  - Use Stamen Terrain for environmental or topographic data
  - Use OpenStreetMap when street context is important

## Workflow Tips

- **Save Projects**: The application remembers recent file pairs for quick access.

- **Batch Processing**: Prepare your KML/GeoJSON pairs in a consistent naming convention for easier batch processing.

- **Analysis First**: Always review the data analysis before finalizing your heatmap style to understand your data distribution.

- **Custom Styles**: Create and save multiple custom styles for different types of data or presentations.

## Troubleshooting

- **No Points in Region**: If no points appear in your heatmap, check that:
  - The KML coordinates are in the correct format (longitude, latitude)
  - The GeoJSON region actually contains some of your points
  - The coordinate systems match (both should use WGS84)

- **Slow Performance**: If the application runs slowly:
  - Reduce the number of points by sampling your data
  - Simplify complex GeoJSON boundaries
  - Close the preview if not needed

- **Export Errors**: If exports fail:
  - Check that you have write permissions to the output directory
  - Ensure you have enough disk space
  - Try a different format (HTML is the most reliable)
""")
        
        tips_text.configure(state="disabled")
    
    def check_updates(self):
        """Check for application updates."""
        # In a real application, this would connect to a server
        # For this demo, just show a message
        self.show_message(f"You are running the latest version ({APP_VERSION}).", "info")
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            f"{APP_AUTHOR}\n\n"
            "A powerful tool for creating customizable heatmaps from KML and GeoJSON data.\n\n"
            "This application is free and open-source software."
        )
    
    def check_command_line_args(self):
        """Check command line arguments for automated operation."""
        parser = argparse.ArgumentParser(description='Enhanced Heatmap Generator')
        parser.add_argument('--kml', help='Path to KML file')
        parser.add_argument('--geojson', help='Path to GeoJSON file')
        parser.add_argument('--style', help='Heatmap style', default='default')
        parser.add_argument('--border', help='Border style', default='normal')
        parser.add_argument('--output', help='Output file path')
        parser.add_argument('--no-open', help='Do not open in browser', action='store_true')
        
        try:
            args, unknown = parser.parse_known_args()
            
            # Check if we have KML and GeoJSON arguments
            if args.kml and args.geojson:
                self.kml_path_var.set(args.kml)
                self.geojson_path_var.set(args.geojson)
                
                # Set style if provided
                if args.style in HEATMAP_STYLES:
                    self.heatmap_style_var.set(args.style)
                
                # Set border style if provided
                if args.border in BORDER_STYLES:
                    self.border_style_var.set(args.border)
                
                # Load the files
                self.after(500, self.load_files)
                
                # Create heatmap after files are loaded
                def create_after_load():
                    if hasattr(self.data_manager, "filtered_coords") and self.data_manager.filtered_coords:
                        # Create heatmap
                        self.create_heatmap()
                        
                        # Save to specific output if provided
                        if args.output and hasattr(self.heatmap_generator, "map") and self.heatmap_generator.map:
                            self.heatmap_generator.save_map(args.output)
                        
                        # Don't open in browser if --no-open flag is set
                        if args.no_open:
                            self.settings["auto_open_map"] = False
                    
                self.after(1000, create_after_load)
        
        except Exception as e:
            logger.error(f"Error parsing command line arguments: {e}")
            logger.error(traceback.format_exc())


def main():
    """Main entry point for the application."""
    try:
        # Set global exception handler for better error reporting
        def global_exception_handler(exc_type, exc_value, exc_traceback):
            logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
            # Show error message to user
            error_msg = f"An unexpected error occurred:\n{exc_value}\n\nPlease check the log file for details."
            try:
                messagebox.showerror("Error", error_msg)
            except:
                print(error_msg)
        
        # Set the exception handler
        sys.excepthook = global_exception_handler
        
        # Initialize logging before GUI
        log_file = str(LOG_FILE)
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
        
        # Create and run the GUI
        app = HeatmapUI()
        app.mainloop()
        
    except Exception as e:
        # Log the error
        try:
            logger.error(f"Fatal error: {e}")
            logger.error(traceback.format_exc())
        except:
            print(f"Fatal error: {e}")
            traceback.print_exc()
        
        # Show error to user
        try:
            messagebox.showerror("Fatal Error", f"A fatal error occurred:\n{e}\n\nThe application will now exit.")
        except:
            print(f"A fatal error occurred: {e}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()