import tkinter as tk
from tkinter import ttk, Label, Entry, Button
from tkinter import StringVar, OptionMenu
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
import math

chernoff_definition = """
Chernoff se réfère souvent aux inégalités de Chernoff,utilisées en probabilités et en statistiques. 
L'inégalité de Chernoff est une estimation probabiliste de la probabilité qu'une somme de variables
aléatoires indépendantes dépasse un certain seuil. Sa formule dépend du contexte, mais une forme 
générique est l'inégalité de Chernoff exponentielle, qui peut s'écrire comme suit :

P(X ≥ t) ≤ e^{-t}*ϕ(t) 

où (X) est la somme des variables aléatoires,(t) est le seuil, et (ϕ(t)) est la fonction caractéristique exponentielle.
"""

jensen_definition = """
Jensen est souvent associé à l'inégalité de Jensen,
 une propriété importante en théorie de l'optimisation et de l'information.
L'inégalité de Jensen stipule que pour toute fonction
 convexe f, la moyenne de la fonction appliquée 
 à une variable aléatoire est supérieure ou égale à la fonction 
 appliquée à la moyenne de cette variable aléatoire. La formule générale 
 de l'inégalité de Jensen est la suivante :

f(E[X]) ≤ E[f(X)]

où (X) est la variable aléatoire,(E[X]) est l'espérance 
de (X), et f est une fonction convexe.
"""

chebyshev_definition = """
L'inégalité de Bienaymé-Tchebychev est utilisée en statistiques pour fournir
 une borne supérieure à la probabilité que la valeur d'une variable 
 aléatoire s'écarte de sa moyenne. La formule générale est :

P(|X - μ| ≥ kσ) ≤ 1/k^2

où (X) est la variable aléatoire, (μ) est la moyenne, 
(σ) est l'écart type, et (k) est un nombre réel positif.
"""
markov_definition = """
Les inégalités de Markov sont des résultats mathématiques qui fournissent d
es bornes supérieures pour la probabilité qu'une variable aléatoire positive 
prenne une valeur au moins égale à un multiple positif de son espérance.
La formule générale de l'inégalité de Markov est la suivante :

P(X ≥ a) ≤ E[X]/a

où (X) est une variable aléatoire positive et (a) est un nombre 
réel positif. Cette inégalité est utile pour obtenir des bornes supérieures
 sur la probabilité d'événements rares en fonction de l'espérance de la variable aléatoire.

"""


# Program 1
def chernoff_bound(n, p, delta):
    mean = n * p
    bound = mean * (1 + delta)
    return bound

def plot_chernoff(n, p, delta):
    x = np.arange(0, n + 1)
    y = binom.pmf(x, n, p)

    fig, ax = plt.subplots()
    ax.bar(x, y, label='Distribution Binomiale')

    bound = chernoff_bound(n, p, delta)
    ax.axvline(bound, color='red', linestyle='dashed', label='borne supérieure')

    ax.set_title(f'Chernoff graphe Binomiale (n={n}, p={p}, δ={delta},borne={bound})')
    ax.set_xlabel('X')
    ax.set_ylabel('Probabilité')
    ax.legend()

    return fig


def on_button_click():
    n_value = int(n_entry.get())
    p_value = float(p_entry.get())
    delta_value = float(delta_entry.get())

    figure = plot_chernoff(n_value, p_value, delta_value)

    canvas = FigureCanvasTkAgg(figure, master=tab2)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=5, column=0, columnspan=3)


# Program 2
def esperance(X, p):
    # Calcul de l'espérance de X
    return np.dot(X, p)

def fonction_convexe(x, convex_type='exp'):
    # Fonction convexe
    if convex_type == 'exp':
        return np.exp(x)
    elif convex_type == 'carré':
        return x**2
    elif convex_type == 'X':
        return x
    elif convex_type == 'log':
        return np.log1p(x)
    elif convex_type == 'racine':
        return np.sqrt(x)
    elif convex_type == 'cube':
        return x**3
    elif convex_type == 'sin':
        return np.sin(x)
    elif convex_type == 'cos':
        return np.cos(x)
    elif convex_type == 'tan':
        return np.tan(x)
    elif convex_type == 'arctan':
        return np.arctan(x)
    elif convex_type == 'arcsin':
        return np.arcsin(x)
    elif convex_type == 'arccos':
        return np.arccos(x)
    else:
        raise ValueError(f"Type de fonction convexe non pris en charge : {convex_type}")



def jensen_inequality(convex_type, values, weights):
    # Calcul de l'espérance de X
    esp_X = esperance(values, weights)
    # Application de la fonction convexe à l'espérance de X
    esp_convexe_X = fonction_convexe(esp_X, convex_type)
    # Espérance de la fonction convexe de X
    esp_convexe = np.dot(fonction_convexe(values, convex_type), weights)

    return esp_convexe - esp_convexe_X



def calculer_inegalite():
    valeurs = np.array([float(entry.get()) for entry in entries_values])
    poids = np.array([float(entry.get()) for entry in entries_weights])

    try:
        resultat_inegalite = jensen_inequality(selected_convex_type.get(),valeurs,poids)
        if (resultat_inegalite >= 0):
            label_resultat.config(
                text=f"Résultat de l'inégalité de Jensen : {resultat_inegalite} d'ou elle est verrifiee")
        else:
            label_resultat.config(
                text=f"Résultat de l'inégalité de Jensen : {resultat_inegalite} d'ou elle n'est pas verrifiee")
    except ValueError as e:
        label_resultat.config(text=str(e))

def update_convex_type():
    selected_type = selected_convex_type.get()

# Program 3 (Integrated with Program 4)
def markov_inequality(X, a):
    expectation_X = np.mean(X)
    probability = np.sum(X >= a) / len(X)
    return probability, expectation_X / a


def chebyshev_inequality(X, k, sigma):
    mean_X = np.mean(X)
    std_dev_X = np.std(X)
    probability = np.sum(np.abs(X - mean_X) >= k * sigma) / len(X)
    return probability, 1 / k ** 2


def update_markov():
    a_value = float(markov_a_entry.get())
    n_value = int(markov_n_entry.get())
    data = np.random.normal(loc=10, scale=5, size=n_value)
    markov_probability, markov_bound = markov_inequality(data, a_value)

    markov_label.config(text=f"Inégalité de Markov: P(X ≥ {a_value})={markov_probability:.3f} ≤ {markov_bound:.3f}")

    markov_ax.clear()
    markov_ax.hist(data, bins=20, color='blue', alpha=0.7, density=True)
    markov_ax.set_title("Inégalité de Markov")
    markov_ax.set_xlabel("Valeurs")
    markov_ax.set_ylabel("Fréquence relative")
    markov_ax.axvline(x=a_value, color='red', linestyle='--', label=f'a = {a_value}')
    markov_ax.legend()
    markov_canvas.draw()


def update_chebyshev():
    k_value = float(chebyshev_k_entry.get())
    n_value = int(chebyshev_n_entry.get())
    sigma_value = float(chebyshev_sigma_entry.get())
    data = np.random.normal(loc=10, scale=sigma_value, size=n_value)

    chebyshev_probability, chebyshev_bound = chebyshev_inequality(data, k_value, sigma_value)

    chebyshev_label.config(
        text=f"Inégalité de Bienaymé-Tchebychev: P(|X - μ| ≥ {k_value}*σ)={chebyshev_probability:.3f} ≤ {chebyshev_bound:.3f}")
    mean_value = np.mean(data)
    std_dev_value = np.std(data)

    lower_bound = mean_value - k_value * std_dev_value
    upper_bound = mean_value + k_value * std_dev_value

    chebyshev_ax.clear()  # Efface le contenu du graphique
    chebyshev_ax.plot(np.sort(data), norm.pdf(np.sort(data), np.mean(data), np.std(data)), color='blue',
             label='Densité de probabilité')
    chebyshev_ax.axvline(x=np.mean(data) - k_value * np.std(data), color='red', linestyle='--', label=f'Moyenne - {k_value}*σ')
    chebyshev_ax.axvline(x=np.mean(data) + k_value * np.std(data), color='green', linestyle='--', label=f'Moyenne + {k_value}*σ')
    chebyshev_ax.set_title("Inégalité de Bienaymé-Tchebychev")
    chebyshev_ax.set_xlabel("Valeurs")
    chebyshev_ax.set_ylabel("Densité de probabilité")
    chebyshev_ax.legend()
    chebyshev_canvas.draw()
def plot_cdf():
    k_value = float(chebyshev_k_entry.get())
    n_value = int(chebyshev_n_entry.get())
    sigma_value = float(chebyshev_sigma_entry.get())
    data = np.random.normal(loc=10, scale=sigma_value, size=n_value)
    
    chebyshev_probability, chebyshev_bound = chebyshev_inequality(data, k_value, sigma_value)

    chebyshev_label.config(
        text=f"Inégalité de Bienaymé-Tchebychev: P(|X - μ| ≥ {k_value}*σ)={chebyshev_probability:.3f} ≤ {chebyshev_bound:.3f}")
    chebyshev_ax.clear()  # Efface le contenu du graphique
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, n_value + 1) / n_value
    chebyshev_ax.plot(sorted_data, cdf_values, color='orange', label='Fonction de Répartition (CDF)')
    chebyshev_ax.set_title("Fonction de Répartition - Inégalité de Bienaymé-Tchebychev")
    chebyshev_ax.set_xlabel("Valeurs")
    chebyshev_ax.set_ylabel("Probabilité cumulative")
    chebyshev_ax.legend()
    chebyshev_canvas.draw()



# Program 3 (Cont.)
# Génération de données aléatoires pour les tests
np.random.seed(42)
data = np.random.normal(loc=10, scale=5, size=1000)

# Création de l'interface graphique
window = tk.Tk()
window.title("Inégalités")

# Notebook (Tabs)
notebook = ttk.Notebook(window)
notebook.pack(pady=10, padx=10)

# Tab 4 - Chebyshev Inequality
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Inégalité de Bienaymé-Tchebychev")

chebyshev_label_definition = Label(tab1, text=chebyshev_definition, font=("Helvetica", 15))
chebyshev_label_definition.grid(row=0, column=0, columnspan=2)

chebyshev_k_label = Label(tab1, text="Valeur de k:", font=("Helvetica", 15))
chebyshev_k_label.grid(row=2, column=0, padx=5, pady=5)

chebyshev_k_entry = Entry(tab1)
chebyshev_k_entry.grid(row=2, column=1, padx=5, pady=5)

chebyshev_n_label = Label(tab1, text="Taille de l'échantillon:", font=("Helvetica", 15))
chebyshev_n_label.grid(row=3, column=0, padx=5, pady=5)

chebyshev_n_entry = Entry(tab1)
chebyshev_n_entry.grid(row=3, column=1, padx=5, pady=5)

chebyshev_sigma_label = Label(tab1, text="Écart type (σ):", font=("Helvetica", 15))
chebyshev_sigma_label.grid(row=4, column=0, padx=5, pady=5)

chebyshev_sigma_entry = Entry(tab1)
chebyshev_sigma_entry.grid(row=4, column=1, padx=5, pady=5)

chebyshev_update_button = Button(tab1, text="densite", command=update_chebyshev, font=("Helvetica", 15))
chebyshev_update_button.grid(row=5, column=0, columnspan=2, pady=10)

cdf_button = Button(tab1, text="repartition", command=plot_cdf, font=("Helvetica", 15))
cdf_button.grid(row=6, column=0,columnspan=2,pady=10)

chebyshev_label = Label(tab1, text="", font=("Helvetica", 15))
chebyshev_label.grid(row=7, column=0, columnspan=2, pady=10)

chebyshev_fig, chebyshev_ax = plt.subplots(figsize=(6, 4))
chebyshev_canvas = FigureCanvasTkAgg(chebyshev_fig, master=tab1)
chebyshev_canvas_widget = chebyshev_canvas.get_tk_widget()
chebyshev_canvas_widget.grid(row=8, column=0, columnspan=2, pady=10)

tab1.columnconfigure(0, weight=1)

# Tab 2 - Chernoff Bound Calculator
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Inégalité de Chernoff")

n_label = Label(tab2, text="n:", font=("Helvetica", 15))
n_label.grid(row=1, column=0)
n_entry = Entry(tab2)
n_entry.grid(row=1, column=1)

p_label = Label(tab2, text="p:", font=("Helvetica", 15))
p_label.grid(row=2, column=0)
p_entry = Entry(tab2)
p_entry.grid(row=2, column=1)

delta_label = Label(tab2, text="δ:", font=("Helvetica", 15))
delta_label.grid(row=3, column=0)
delta_entry = Entry(tab2)
delta_entry.grid(row=3, column=1)

# Ajout des explications sous forme de libellés d'aide
n_help_label = Label(tab2, text="Taille de l'échantillon (nombre total d'observations)", font=("Helvetica", 15))
n_help_label.grid(row=1, column=2, columnspan=2)

p_help_label = Label(tab2, text="Probabilité de succès dans une seule observation", font=("Helvetica", 15))
p_help_label.grid(row=2, column=2, columnspan=2)

delta_help_label = Label(tab2, text="Paramètre de réglage pour déterminer la largeur de la borne",
                         font=("Helvetica", 15))
delta_help_label.grid(row=3, column=2, columnspan=2)

calculate_button = Button(tab2, text=" Calculer", command=on_button_click, font=("Helvetica", 15))
calculate_button.grid(row=4, column=0, columnspan=3)

chernoff_label = Label(tab2, text=chernoff_definition, font=("Helvetica", 15))
chernoff_label.grid(row=0, column=0, columnspan=3)

selected_convex_type = StringVar()
selected_convex_type.set('exp')  # Valeur par défaut


convex_options = ['exp', 'carré', 'X', 'log', 'racine', 'cube', 'sin', 'cos', 'tan', 'arctan', 'arcsin', 'arccos']

# Tab 3 - Jensen's Inequality Calculator
tab3 = ttk.Frame(notebook)
notebook.add(tab3, text="Inégalité de Jensen")
jensen_label = Label(tab3, text=jensen_definition, font=("Helvetica", 15))
jensen_label.pack()
Label(tab3, text="Valeurs (séparées par des points):", font=("Helvetica", 15)).pack()
entries_values = [Entry(tab3) for _ in range(3)]
for entry in entries_values:
    entry.pack()

Label(tab3, text="Probabilite (séparés par des points):", font=("Helvetica", 15)).pack()
entries_weights = [Entry(tab3) for _ in range(3)]
for entry in entries_weights:
    entry.pack()

convex_dropdown = OptionMenu(tab3, selected_convex_type, *convex_options, command=update_convex_type)
convex_dropdown.pack()


Button(tab3, text="Calculer", command=calculer_inegalite, font=("Helvetica", 15)).pack()

label_resultat = Label(tab3, text="")
label_resultat.pack()

# Tab 4 - Markov Inequality
tab4 = ttk.Frame(notebook)
notebook.add(tab4, text="Inégalité de Markov")

markov_a_label = Label(tab4, text="Valeur de a:", font=("Helvetica", 15))
markov_a_label.grid(row=2, column=0, padx=0, pady=0)

markov_label_definition = Label(tab4, text=markov_definition, font=("Helvetica", 15))
markov_label_definition.grid(row=0, column=0, columnspan=2)

markov_a_entry = Entry(tab4)
markov_a_entry.grid(row=3, column=0, padx=0, pady=0)

markov_n_label = Label(tab4, text="Taille de l'échantillon:", font=("Helvetica", 15))
markov_n_label.grid(row=4, column=0, padx=5, pady=5)

markov_n_entry = Entry(tab4)
markov_n_entry.grid(row=5, column=0, padx=5, pady=5)

markov_update_button = Button(tab4, text="Calculer", command=update_markov, font=("Helvetica", 15))
markov_update_button.grid(row=6, column=0, columnspan=2, pady=10)

markov_label = Label(tab4, text="", font=("Helvetica", 15))
markov_label.grid(row=7, column=0, columnspan=2, pady=10)

markov_fig, markov_ax = plt.subplots(figsize=(6, 4))
markov_canvas = FigureCanvasTkAgg(markov_fig, master=tab4)
markov_canvas_widget = markov_canvas.get_tk_widget()
markov_canvas_widget.grid(row=8, column=0, columnspan=2, pady=10)

tab4.columnconfigure(0, weight=1)


# Configure global resizing behavior
window.columnconfigure(0, weight=1)
notebook.grid(row=0, column=0, sticky="nsew")
window.rowconfigure(0, weight=1)

# Launch the GUI
window.mainloop()

