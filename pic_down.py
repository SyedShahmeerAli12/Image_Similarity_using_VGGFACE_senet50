from bing_image_downloader import downloader
import os

# List of 20 Hollywood actors
actors = [
    "Leonardo DiCaprio", "Brad Pitt", "Johnny Depp", "Tom Cruise", "Robert Downey Jr.",
    "Chris Hemsworth", "Chris Evans", "Scarlett Johansson", "Angelina Jolie", "Jennifer Lawrence",
    "Emma Stone", "Tom Hanks", "Morgan Freeman", "Denzel Washington", "Will Smith",
    "Hugh Jackman", "Matthew McConaughey", "Ryan Reynolds", "Mark Ruffalo", "Anne Hathaway"
]

# Directory to save images
output_dir = 'Dataset'

# Download 50 photos for each actor
for actor in actors:
    print(f"Downloading images for {actor}...")
    downloader.download(actor, limit=50, output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=60)

print("Download completed.")
