import sqlite3
from datetime import datetime

sample_penguins = [
    {
        "name": "Emperor Penguin",
        "species": "Emperor",
        "weight": 30.0,
        "height": 115.0,
        "location": "Antarctica",
        "image_path": "/static/images/emperor.jpg",
        "tags": "emperor,antarctica,large"
    },
    {
        "name": "Adelie Penguin",
        "species": "Adelie",
        "weight": 5.0,
        "height": 70.0,
        "location": "Antarctic Peninsula",
        "image_path": "/static/images/adelie.jpg",
        "tags": "adelie,small,peninsula"
    },
    {
        "name": "King Penguin",
        "species": "King",
        "weight": 15.0,
        "height": 90.0,
        "location": "Sub-Antarctic islands",
        "image_path": "/static/images/king.jpg",
        "tags": "king,sub-antarctic,medium"
    },
    {
        "name": "Gentoo Penguin",
        "species": "Gentoo",
        "weight": 6.5,
        "height": 80.0,
        "location": "Antarctic Peninsula",
        "image_path": "/static/images/gentoo.jpg",
        "tags": "gentoo,antarctic,medium"
    }
]

# Connect to database
conn = sqlite3.connect('penguins.db')
c = conn.cursor()

# Clear existing data
c.execute('DELETE FROM penguins')

# Add new penguins
for penguin in sample_penguins:
    c.execute('''INSERT INTO penguins 
                 (name, species, weight, height, location, image_path, tags, date_added)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (penguin['name'], penguin['species'], penguin['weight'], penguin['height'],
               penguin['location'], penguin['image_path'], penguin['tags'], datetime.now()))

# Commit changes and close connection
conn.commit()
conn.close()

print("Sample penguin data added successfully!") 