import sqlite3
import os
import shutil
from datetime import datetime

def add_penguin(name, species, weight, height, location, image_path, tags):
    # Connect to database
    conn = sqlite3.connect('penguins.db')
    c = conn.cursor()
    
    # Copy image to static/images directory if it exists
    if os.path.exists(image_path):
        # Get filename from path
        filename = os.path.basename(image_path)
        # Create new path in static/images
        new_image_path = os.path.join('static', 'images', filename)
        # Copy file
        shutil.copy2(image_path, new_image_path)
        # Update image_path to be relative to static folder
        image_path = f"/static/images/{filename}"
    else:
        print(f"Warning: Image file {image_path} not found. Using default image.")
        image_path = "/static/images/default.jpg"
    
    # Add penguin to database
    c.execute('''INSERT INTO penguins 
                 (name, species, weight, height, location, image_path, tags, date_added)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (name, species, weight, height, location, image_path, tags, datetime.now()))
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print(f"Successfully added {name} to the database!")

if __name__ == "__main__":
    # Example usage:
    # add_penguin(
    #     name="African Penguin",
    #     species="African",
    #     weight=3.5,
    #     height=60.0,
    #     location="South Africa",
    #     image_path="path/to/your/image.jpg",  # Full path to your image file
    #     tags="african,south africa,small"
    # )
    
    print("To add a penguin, use the add_penguin function with these parameters:")
    print("add_penguin(name, species, weight, height, location, image_path, tags)")
    print("\nExample:")
    print('''add_penguin(
    name="African Penguin",
    species="African",
    weight=3.5,
    height=60.0,
    location="South Africa",
    image_path="path/to/your/image.jpg",
    tags="african,south africa,small"
)''') 