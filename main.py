import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess

# Fonction pour exécuter un fichier Python correspondant à la section
def execute_script(script_name):
    try:
        subprocess.Popen(["python", script_name])
    except FileNotFoundError:
        print(f"Fichier introuvable : {script_name}")

# Création de la fenêtre principale
app = tk.Tk()
app.title("Simulateur des Lois")
app.geometry("1000x600")
app.configure(bg="#f9f9f9")

# Diviser la fenêtre en deux parties
left_frame = tk.Frame(app, bg="#429596", width=700)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH)

right_frame = tk.Frame(app, bg="#ffffff", width=300)
right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

# Contenu de la partie gauche
# Titre
title_label = tk.Label(left_frame, text="Simulateur des Lois", bg="#429596", fg="white", 
                       font=("Arial", 20, "bold"))
title_label.pack(pady=60)

# Image
try:
    img = Image.open("mathLogo.png")  # Remplacez par le chemin de votre image
    img = img.resize((200, 200), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(left_frame, image=img, bg="#429596")
    img_label.image = img  # Garder une référence pour éviter le garbage collection
    img_label.pack(pady=10)
except FileNotFoundError:
    tk.Label(left_frame, text="Image introuvable", bg="#429596", fg="white", font=("Arial", 12, "italic")).pack(pady=10)

# Citation
quote_label = tk.Label(left_frame, text="“Les mathématiques sont la poésie des sciences.”", 
                        bg="#429596", fg="white", font=("Arial", 12, "italic"), wraplength=250)
quote_label.pack(pady=20)

# Contenu de la partie droite
# Sections avec noms, scripts et icônes associées
sections = [
    {"name": "Lois Continues", "script": "projet_continues.py", "icon": "mathIcon.png"},
    {"name": "Lois Discrètes", "script": "loidiscre.py", "icon": "mathIcon.png"},
    {"name": "Inégalités", "script": "inegalite.py", "icon": "mathIcon.png"},
    {"name": "Processus", "script": "processus.py", "icon": "mathIcon.png"},
    {"name": "Exemples", "script": "exemples.py", "icon": "mathIcon.png"}
]

for section in sections:
    button_frame = tk.Frame(right_frame, bg="#ffffff")
    button_frame.pack(pady=40, padx=100, anchor="w", fill=tk.X)

    # Icône
    try:
        icon = Image.open(section["icon"])
        icon = icon.resize((30, 30), Image.Resampling.LANCZOS)
        icon = ImageTk.PhotoImage(icon)
        icon_label = tk.Label(button_frame, image=icon, bg="#ffffff")
        icon_label.image = icon  # Garder une référence pour éviter le garbage collection
        icon_label.pack(side=tk.LEFT, padx=10)
    except FileNotFoundError:
        tk.Label(button_frame, text="❌", bg="#ffffff", fg="red", font=("Arial", 14)).pack(side=tk.LEFT, padx=10)

    # Bouton
    btn = ttk.Button(button_frame, text=section["name"], command=lambda s=section["script"]: execute_script(s))
    btn.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

# Lancer l'application
app.mainloop()
