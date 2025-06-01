import sqlite3
import os
from tabulate import tabulate

def view_all_penguins():
    conn = sqlite3.connect('penguins.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM penguins')
    penguins = cursor.fetchall()
    
    if not penguins:
        print("\nNo penguins in the database.")
        return
    
    print("\nAll Penguins:")
    print("-" * 50)
    for penguin in penguins:
        print(f"ID: {penguin[0]}")
        print(f"Name: {penguin[1]}")
        print(f"Weight: {penguin[2]} kg")
        print(f"Tags: {penguin[3]}")
        print(f"Image Path: {penguin[4]}")
        print("-" * 50)
    
    conn.close()

def delete_penguin(penguin_id):
    conn = sqlite3.connect('penguins.db')
    cursor = conn.cursor()
    
    # Check if penguin exists
    cursor.execute('SELECT * FROM penguins WHERE id = ?', (penguin_id,))
    if not cursor.fetchone():
        print(f"\nNo penguin found with ID {penguin_id}")
        return
    
    # Delete the penguin
    cursor.execute('DELETE FROM penguins WHERE id = ?', (penguin_id,))
    conn.commit()
    print(f"\nPenguin with ID {penguin_id} has been deleted")
    conn.close()

def main_menu():
    while True:
        print("\nPenguin Database Manager")
        print("1. View all penguins")
        print("2. Delete a penguin")
        print("3. View database contents")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            view_all_penguins()
        elif choice == '2':
            penguin_id = input("Enter the ID of the penguin to delete: ")
            try:
                delete_penguin(int(penguin_id))
            except ValueError:
                print("Please enter a valid number")
        elif choice == '3':
            view_database()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def view_database():
    # Connect to the database
    conn = sqlite3.connect('penguins.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\nDatabase Contents:")
    print("=" * 50)
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        print("-" * 50)
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get all rows
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        # Print table contents
        if rows:
            print(tabulate(rows, headers=columns, tablefmt="grid"))
        else:
            print("No data in this table")
    
    conn.close()

if __name__ == '__main__':
    main_menu() 