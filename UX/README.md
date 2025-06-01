# Penguin Database

A local web application for managing and visualizing penguin data. Features include:
- Display of penguin information with images
- Search functionality
- Statistical visualizations
- Tagging system

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Adding Penguin Data

To add penguin data, you can use the Python shell:

```python
from app import app, db, Penguin

with app.app_context():
    new_penguin = Penguin(
        name="Emperor Penguin",
        species="Emperor",
        weight=30.0,
        height=115.0,
        location="Antarctica",
        image_path="/static/images/emperor.jpg",
        tags="emperor,antarctica,large"
    )
    db.session.add(new_penguin)
    db.session.commit()
```

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: Frontend template
- `static/images/`: Directory for penguin images
- `penguins.db`: SQLite database (created automatically)

## Features

1. **Search**: Real-time search by name, species, or tags
2. **Statistics**: Visual representation of penguin data
3. **Responsive Design**: Works on desktop and mobile devices
4. **Interactive Cards**: Hover effects and clean layout 