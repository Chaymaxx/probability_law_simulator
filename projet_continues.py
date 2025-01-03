import tkinter as tk
from tkinter import ttk, scrolledtext
from scipy.stats import norm, expon, weibull_min, cauchy, chi2, beta
from scipy.stats import f as fisher, gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import simpledialog

class SimulatorApp:
    def __init__(self, master):
        self.master = master
        master.title("Simulateur de Loi")

        self.tabControl = ttk.Notebook(master)
        self.tabControl.pack(expand=1, fill="both")

        self.add_tab(SimulateurLoiNormale, "Loi Normale")
        self.add_tab(SimulateurLoiNormaleCentreeReduite, "Loi Normale Centrée Réduite")
        self.add_tab(SimulateurLoiExponentielle, "Loi Exponentielle")
        self.add_tab(SimulateurLoiExponentielle, "Loi de Weibull")
        self.add_tab(SimulateurLoiCauchy, "Loi de Cauchy")
        self.add_tab(SimulateurLoiChiCarre, "Loi de hi2")
        self.add_tab(SimulateurLoiFisher, "Loi de Fisher")
        self.add_tab(SimulateurLoiGamma, "Loi de Gamma")
        self.add_tab(SimulateurLoiBeta, "Loi de Beta")

    def add_tab(self, tab_class, tab_title):
        tab = ttk.Frame(self.tabControl)
        self.tabControl.add(tab, text=tab_title)
        tab_class(tab)

class SimulateurLoiNormale:
    def __init__(self, master):
        self.master = master
        #self.master.tk_setPalette(background="black")
        #master.title("Simulateur de Loi Normale")

        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_text = """
        Définition:
            La loi normale, également appelée distribution gaussienne, est une distribution de probabilité continue symétrique par rapport à sa moyenne. Elle est caractérisée par deux paramètres : la moyenne (μ) et l'écart-type (σ). La forme de la courbe est une cloche, centrée autour de la moyenne.

            Formule de la loi normale :
            f(x|μ,σ) = (1 / (σ * √(2π))) * e^(-((x-μ)^2) / (2σ^2))

            où :
            - μ est la moyenne,
            - σ est l'écart-type.
            """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi normale
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.mean_label = tk.Label(self.parameters_frame, text="Moyenne (μ):")
        self.mean_label.grid(row=0, column=0, sticky="e")

        self.mean_entry = tk.Entry(self.parameters_frame)
        self.mean_entry.grid(row=0, column=1, sticky="w")

        self.stddev_label = tk.Label(self.parameters_frame, text="Écart-type (σ):")
        self.stddev_label.grid(row=1, column=0, sticky="e")

        self.stddev_entry = tk.Entry(self.parameters_frame)
        self.stddev_entry.grid(row=1, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=2, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=2, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            mean = float(self.mean_entry.get())
            stddev = float(self.stddev_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = norm.cdf(x_value, loc=mean, scale=stddev)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        try:
            mean = float(self.mean_entry.get())
            self.result_label.config(text=f"L'espérance (moyenne) est : {mean:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer une valeur numérique valide pour la moyenne.")

    def calculate_variance(self):
        try:
            stddev = float(self.stddev_entry.get())
            variance = stddev**2
            self.result_label.config(text=f"La variance est : {variance:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer une valeur numérique valide pour l'écart-type.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            mean = float(self.mean_entry.get())
            stddev = float(self.stddev_entry.get())

            x_values = np.linspace(mean - 4 * stddev, mean + 4 * stddev, 1000)
            cdf_values = norm.cdf(x_values, loc=mean, scale=stddev)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            mean = float(self.mean_entry.get())
            stddev = float(self.stddev_entry.get())

            x_values = np.linspace(mean - 4 * stddev, mean + 4 * stddev, 1000)
            y_values = norm.pdf(x_values, loc=mean, scale=stddev)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution normale")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")



'''
class SimulateurLoiNormaleCentreeReduite:
    def __init__(self, master):
        self.master = master

        self.description_text = """
               La loi normale centrée réduite est une distribution normale standardisée avec une moyenne de 0 et un écart-type de 1.
               Sa fonction de densité de probabilité (PDF) est donnée par :

               Formule de la loi normale centrée réduite :
               f(x|0,1) = (1 / √(2π)) * e^(-x^2 / 2)

               où :
               - π est la constante mathématique π,
               - e est la base du logarithme naturel.
               """

        wrap_length = 600

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.pack(padx=10, pady=10)

        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability, bg="white", fg="black")
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="", bg="white", fg="black")
        self.result_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph, bg="white", fg="black")
        self.graph_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            x_value = float(tk.simpledialog.askstring("Valeur de x", "Entrez la valeur de x:"))

            probability = norm.cdf(x_value, loc=0, scale=1)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer une valeur numérique valide.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            x_values = np.linspace(-4, 4, 1000)
            y_values = norm.pdf(x_values, loc=0, scale=1)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution normale centrée réduite")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Une erreur s'est produite lors de la création du graphique.")
            
'''
class SimulateurLoiNormaleCentreeReduite:
    def __init__(self, master):
        self.master = master

        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_text = """
        Définition:
        La loi normale centrée réduite est une distribution normale standardisée avec une moyenne de 0 et un écart-type de 1.
        Sa fonction de densité de probabilité (PDF) est donnée par :

        Formule de la loi normale centrée réduite :
        f(x|0,1) = (1 / √(2π)) * e^(-x^2 / 2)

        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi normale
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=0, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=0, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            x_value = float(self.x_value_entry.get())

            probability = norm.cdf(x_value, loc=0, scale=1)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer une valeur numérique valide.")

    def calculate_mean(self):
        mean = norm.mean(loc=0, scale=1)
        result_text = f"L'espérance de la distribution normale centrée réduite est : {mean:.4f}"
        self.result_label.config(text=result_text)

    def calculate_variance(self):
        variance = norm.var(loc=0, scale=1)
        result_text = f"La variance de la distribution normale centrée réduite est : {variance:.4f}"
        self.result_label.config(text=result_text)

    def plot_cdf(self):
        try:
            self.ax.clear()

            x_values = np.linspace(-4, 4, 1000)
            cdf_values = norm.cdf(x_values, loc=0, scale=1)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Une erreur s'est produite lors de la création du graphique de la CDF.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            x_values = np.linspace(-4, 4, 1000)
            y_values = norm.pdf(x_values, loc=0, scale=1)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution normale centrée réduite")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Une erreur s'est produite lors de la création du graphique.")


class SimulateurLoiExponentielle:
    def __init__(self, master):
        self.master = master


        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_text = """
        La distribution exponentielle est une distribution de probabilité continue qui décrit le temps entre des événements
        qui se produisent indépendamment à un taux constant. Elle est caractérisée par un seul paramètre lambda (λ), qui
        est le taux de défaillance moyen.

        Formule de la loi exponentielle :
        f(x|λ) = λ * e^(-λx)

        où :
        - λ est le taux de défaillance moyen,
        - e est la base du logarithme naturel.
        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi exponentielle
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.rate_label = tk.Label(self.parameters_frame, text="Taux de défaillance moyen (λ):")
        self.rate_label.grid(row=0, column=0, sticky="e")

        self.rate_entry = tk.Entry(self.parameters_frame)
        self.rate_entry.grid(row=0, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=1, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=1, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            rate = float(self.rate_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = expon.cdf(x_value, scale=1/rate)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        try:
            rate = float(self.rate_entry.get())
            mean = 1 / rate
            self.result_label.config(text=f"L'espérance (moyenne) est : {mean:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer une valeur numérique valide pour le taux de défaillance moyen.")

    def calculate_variance(self):
        try:
            rate = float(self.rate_entry.get())
            variance = 1 / (rate ** 2)
            self.result_label.config(text=f"La variance est : {variance:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer une valeur numérique valide pour le taux de défaillance moyen.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            rate = float(self.rate_entry.get())

            x_values = np.linspace(0, 10 / rate, 1000)
            cdf_values = expon.cdf(x_values, scale=1/rate)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            rate = float(self.rate_entry.get())

            x_values = np.linspace(0, 10 / rate, 1000)
            y_values = expon.pdf(x_values, scale=1/rate)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution Exponentielle")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

class SimulateurLoiCauchy:
    def __init__(self, master):
        self.master = master


        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_text = """
        La distribution de Cauchy, également appelée distribution de Lorentz, est une distribution de probabilité continue
        caractérisée par sa fonction de densité de probabilité (PDF) asymétrique et ses queues lourdes. Elle n'a pas de
        moyenne ou de variance finie.

        Formule de la loi de Cauchy :
        f(x|x0,γ) = (1 / (π * γ * (1 + ((x - x0) / γ)^2)))

        où :
        - x0 est la médiane de la distribution,
        - γ est la demi-largeur à mi-hauteur de la distribution,
        - π est le nombre pi.
        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi de Cauchy
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.median_label = tk.Label(self.parameters_frame, text="Médiane (x0):")
        self.median_label.grid(row=0, column=0, sticky="e")

        self.median_entry = tk.Entry(self.parameters_frame)
        self.median_entry.grid(row=0, column=1, sticky="w")

        self.gamma_label = tk.Label(self.parameters_frame, text="Demi-largeur à mi-hauteur (γ):")
        self.gamma_label.grid(row=1, column=0, sticky="e")

        self.gamma_entry = tk.Entry(self.parameters_frame)
        self.gamma_entry.grid(row=1, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=2, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=2, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            median = float(self.median_entry.get())
            gamma = float(self.gamma_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = cauchy.cdf(x_value, loc=median, scale=gamma)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        self.result_label.config(text="La distribution de Cauchy n'a pas de valeur d'espérance finie.")

    def calculate_variance(self):
        self.result_label.config(text="La distribution de Cauchy n'a pas de valeur de variance finie.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            median = float(self.median_entry.get())
            gamma = float(self.gamma_entry.get())

            x_values = np.linspace(median - 10 * gamma, median + 10 * gamma, 1000)
            cdf_values = cauchy.cdf(x_values, loc=median, scale=gamma)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            median = float(self.median_entry.get())
            gamma = float(self.gamma_entry.get())

            x_values = np.linspace(median - 10 * gamma, median + 10 * gamma, 1000)
            y_values = cauchy.pdf(x_values, loc=median, scale=gamma)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution de Cauchy")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")
class SimulateurLoiChiCarre:
    def __init__(self, master):
        self.master = master


        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_text = """
        Définition :
            La loi du Chi-carré (χ²) est une distribution de probabilité continue qui apparaît souvent dans les tests statistiques tels que le test du chi-carré d'indépendance et le test d'ajustement du chi-carré.

            Elle est définie par un paramètre k appelé "degrés de liberté". La densité de probabilité de la distribution du chi-carré est donnée par :

            f(x|k) = (1 / (2^(k/2) * Γ(k/2))) * x^(k/2 - 1) * e^(-x/2)

            où :
            - x est la variable aléatoire,
            - k est le nombre de degrés de liberté,
            - Γ est la fonction gamma.
        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi du Chi-carré
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.degrees_label = tk.Label(self.parameters_frame, text="Degrés de liberté (k):")
        self.degrees_label.grid(row=0, column=0, sticky="e")

        self.degrees_entry = tk.Entry(self.parameters_frame)
        self.degrees_entry.grid(row=0, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=1, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=1, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            degrees = float(self.degrees_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = chi2.cdf(x_value, degrees)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        try:
            degrees = float(self.degrees_entry.get())
            mean = degrees
            self.result_label.config(text=f"L'espérance (moyenne) est : {mean:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_variance(self):
        try:
            degrees = float(self.degrees_entry.get())
            variance = 2 * degrees
            self.result_label.config(text=f"La variance est : {variance:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            degrees = float(self.degrees_entry.get())

            x_values = np.linspace(0, 10 * degrees, 1000)
            cdf_values = chi2.cdf(x_values, degrees)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            degrees = float(self.degrees_entry.get())

            x_values = np.linspace(0, 10 * degrees, 1000)
            y_values = chi2.pdf(x_values, degrees)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution du Chi-carré")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

class SimulateurLoiWeibull:
    def __init__(self, master):
        self.master = master
        master.title("Simulateur de Loi de Weibull")

        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_text = """
        La distribution de Weibull est une distribution de probabilité continue qui modélise le temps jusqu'à la
        défaillance d'un composant. Elle est caractérisée par sa fonction de densité de probabilité (PDF) donnée par :

        Formule de la loi de Weibull :
        f(x|c, λ) = (c / λ) * ((x / λ)^(c-1)) * e^(-(x / λ)^c)

        où :
        - c est le paramètre de forme,
        - λ est le paramètre d'échelle,
        - e est la base du logarithme naturel.
        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi de Weibull
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.shape_label = tk.Label(self.parameters_frame, text="Paramètre de forme (c):")
        self.shape_label.grid(row=0, column=0, sticky="e")

        self.shape_entry = tk.Entry(self.parameters_frame)
        self.shape_entry.grid(row=0, column=1, sticky="w")

        self.scale_label = tk.Label(self.parameters_frame, text="Paramètre d'échelle (λ):")
        self.scale_label.grid(row=1, column=0, sticky="e")

        self.scale_entry = tk.Entry(self.parameters_frame)
        self.scale_entry.grid(row=1, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=2, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=2, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            shape = float(self.shape_entry.get())
            scale = float(self.scale_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = weibull_min.cdf(x_value, shape, loc=0, scale=scale)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        try:
            shape = float(self.shape_entry.get())
            scale = float(self.scale_entry.get())

            mean = scale * np.exp(1/shape)
            self.result_label.config(text=f"L'espérance (moyenne) est : {mean:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_variance(self):
        try:
            shape = float(self.shape_entry.get())
            scale = float(self.scale_entry.get())

            variance = scale**2 * (np.exp(2/shape) * (np.exp(1/shape) - 1))
            self.result_label.config(text=f"La variance est : {variance:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            shape = float(self.shape_entry.get())
            scale = float(self.scale_entry.get())

            x_values = np.linspace(0, 10 * scale, 1000)
            cdf_values = weibull_min.cdf(x_values, shape, loc=0, scale=scale)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            shape = float(self.shape_entry.get())
            scale = float(self.scale_entry.get())

            x_values = np.linspace(0, 10 * scale, 1000)
            y_values = weibull_min.pdf(x_values, shape, loc=0, scale=scale)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution de Weibull")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")



class SimulateurLoiFisher:
    def __init__(self, master):
        self.master = master
        self.description_text = """
            La loi de Fisher (ou distribution F) est une distribution de probabilité continue qui modélise la distribution des rapports de deux variables aléatoires chi2 indépendantes.

            Formule de la loi de Fisher :
            f(x|d1, d2) = (Γ((d1+d2)/2) * (d1/d2)^(d1/2) * x^(d1/2 - 1)) / (Γ(d1/2) * Γ(d2/2) * (1 + (d1/d2)*x)^( (d1+d2)/2 ))

            où :
            - d1 et d2 sont les degrés de liberté,
            - Γ est la fonction gamma.
            """

        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, rowspan=2, sticky="nsew")

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=600, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi de Fisher
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.degrees1_label = tk.Label(self.parameters_frame, text="Degrés de liberté (d1):")
        self.degrees1_label.grid(row=0, column=0, sticky="e")

        self.degrees1_entry = tk.Entry(self.parameters_frame)
        self.degrees1_entry.grid(row=0, column=1, sticky="w")

        self.degrees2_label = tk.Label(self.parameters_frame, text="Degrés de liberté (d2):")
        self.degrees2_label.grid(row=1, column=0, sticky="e")

        self.degrees2_entry = tk.Entry(self.parameters_frame)
        self.degrees2_entry.grid(row=1, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=2, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=2, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=1)

        self.expectation_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_expectation)
        self.expectation_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            degrees1 = float(self.degrees1_entry.get())
            degrees2 = float(self.degrees2_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = fisher.cdf(x_value, dfn=degrees1, dfd=degrees2)
            result_text = f"La probabilité que X ≤ {x_value} est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_variance(self):
        try:
            degrees1 = float(self.degrees1_entry.get())
            degrees2 = float(self.degrees2_entry.get())
            variance = fisher.var(dfn=degrees1, dfd=degrees2)
            result_text = f"La variance est : {variance:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_expectation(self):
        try:
            degrees1 = float(self.degrees1_entry.get())
            degrees2 = float(self.degrees2_entry.get())
            expectation = fisher.mean(dfn=degrees1, dfd=degrees2)
            result_text = f"L'espérance (moyenne) est : {expectation:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            degrees1 = float(self.degrees1_entry.get())
            degrees2 = float(self.degrees2_entry.get())

            x_values = np.linspace(0, 5, 1000)
            cdf_values = fisher.cdf(x_values, dfn=degrees1, dfd=degrees2)

            self.ax.plot(x_values, cdf_values, label='CDF')
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.legend()
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            degrees1 = float(self.degrees1_entry.get())
            degrees2 = float(self.degrees2_entry.get())

            x_values = np.linspace(0, 5, 1000)
            y_values = fisher.pdf(x_values, dfn=degrees1, dfd=degrees2)

            self.ax.plot(x_values, y_values, label='PDF')
            self.ax.set_title("Distribution de Fisher")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.legend()
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")




class SimulateurLoiGamma:
    def __init__(self, master):
        self.master = master

        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.description_text = """
        La distribution Gamma est une distribution de probabilité continue sur l'intervalle [0, +∞). Elle est souvent utilisée pour modéliser des durées de temps ou des processus de Poisson. La distribution Gamma est définie par deux paramètres : k (shape) et θ (scale).

        Formule de la distribution Gamma :
        f(x|k,θ) = (x^(k-1) * e^(-x/θ)) / (θ^k * Γ(k))

        où :
        - k est le paramètre de forme,
        - θ est le paramètre d'échelle,
        - Γ est la fonction gamma.
        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi Gamma
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.k_label = tk.Label(self.parameters_frame, text="Paramètre de forme (k):")
        self.k_label.grid(row=0, column=0, sticky="e")

        self.k_entry = tk.Entry(self.parameters_frame)
        self.k_entry.grid(row=0, column=1, sticky="w")

        self.theta_label = tk.Label(self.parameters_frame, text="Paramètre d'échelle (θ):")
        self.theta_label.grid(row=1, column=0, sticky="e")

        self.theta_entry = tk.Entry(self.parameters_frame)
        self.theta_entry.grid(row=1, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=2, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=2, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            k = float(self.k_entry.get())
            theta = float(self.theta_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = gamma.pdf(x_value, k, scale=theta)
            result_text = f"La probabilité de densité pour x est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        try:
            k = float(self.k_entry.get())
            theta = float(self.theta_entry.get())

            mean = k * theta
            self.result_label.config(text=f"L'espérance (moyenne) est : {mean:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_variance(self):
        try:
            k = float(self.k_entry.get())
            theta = float(self.theta_entry.get())

            variance = k * (theta ** 2)
            self.result_label.config(text=f"La variance est : {variance:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            k = float(self.k_entry.get())
            theta = float(self.theta_entry.get())

            x_values = np.linspace(0, 10 * theta, 1000)
            cdf_values = gamma.cdf(x_values, k, scale=theta)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            k = float(self.k_entry.get())
            theta = float(self.theta_entry.get())

            x_values = np.linspace(0, 10 * theta, 1000)
            y_values = gamma.pdf(x_values, k, scale=theta)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution Gamma")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")


class SimulateurLoiBeta:
    def __init__(self, master):
        self.master = master

        # Cadre pour la description
        self.description_frame = tk.Frame(master)
        self.description_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.description_text = """
        La distribution Beta est une distribution de probabilité continue sur l'intervalle [0, 1]. Elle est souvent utilisée pour modéliser des proportions ou des taux de réussite. La distribution Beta est définie par deux paramètres : alpha (α) et beta (β).

        Formule de la distribution Beta :
        f(x|α,β) = (x^(α-1) * (1-x)^(β-1)) / B(α,β)

        où :
        - α est le paramètre de forme,
        - β est le paramètre de forme,
        - B est la fonction bêta.
        """

        wrap_length = 600

        self.description_label = tk.Label(self.description_frame, text="", justify=tk.LEFT, wraplength=wrap_length, font=("Time New Roman", 12))
        self.description_label.grid(row=0, column=0, padx=10, pady=10)
        self.update_description_label()

        # Cadre pour les paramètres de la loi Beta
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.alpha_label = tk.Label(self.parameters_frame, text="Paramètre de forme (α):")
        self.alpha_label.grid(row=0, column=0, sticky="e")

        self.alpha_entry = tk.Entry(self.parameters_frame)
        self.alpha_entry.grid(row=0, column=1, sticky="w")

        self.beta_label = tk.Label(self.parameters_frame, text="Paramètre de forme (β):")
        self.beta_label.grid(row=1, column=0, sticky="e")

        self.beta_entry = tk.Entry(self.parameters_frame)
        self.beta_entry.grid(row=1, column=1, sticky="w")

        self.x_value_label = tk.Label(self.parameters_frame, text="Valeur de x:")
        self.x_value_label.grid(row=2, column=0, sticky="e")

        self.x_value_entry = tk.Entry(self.parameters_frame)
        self.x_value_entry.grid(row=2, column=1, sticky="w")

        # Boutons d'action
        self.actions_frame = tk.Frame(master)
        self.actions_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.mean_button = tk.Button(self.actions_frame, text="Calculer l'espérance", command=self.calculate_mean)
        self.mean_button.grid(row=0, column=1)

        self.variance_button = tk.Button(self.actions_frame, text="Calculer la variance", command=self.calculate_variance)
        self.variance_button.grid(row=0, column=2)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=3)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=4)

        # Résultat
        self.result_frame = tk.Frame(master)
        self.result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(master)
        self.graph_frame.grid(row=0, column=2, rowspan=5, sticky="nsew")

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=2)
        master.rowconfigure(0, weight=2)
        master.rowconfigure(1, weight=2)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=2)
        master.rowconfigure(4, weight=2)

    def update_description_label(self):
        self.description_label.config(text=self.description_text)

    def calculate_probability(self):
        try:
            alpha = float(self.alpha_entry.get())
            beta_val = float(self.beta_entry.get())
            x_value = float(self.x_value_entry.get())

            probability = beta.pdf(x_value, alpha, beta_val)
            result_text = f"La probabilité de densité pour x est : {probability:.4f}"
            self.result_label.config(text=result_text)
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_mean(self):
        try:
            alpha = float(self.alpha_entry.get())
            beta_val = float(self.beta_entry.get())

            mean = alpha / (alpha + beta_val)
            self.result_label.config(text=f"L'espérance (moyenne) est : {mean:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def calculate_variance(self):
        try:
            alpha = float(self.alpha_entry.get())
            beta_val = float(self.beta_entry.get())

            variance = (alpha * beta_val) / ((alpha + beta_val)**2 * (alpha + beta_val + 1))
            self.result_label.config(text=f"La variance est : {variance:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()

            alpha = float(self.alpha_entry.get())
            beta_val = float(self.beta_entry.get())

            x_values = np.linspace(0, 1, 1000)
            cdf_values = beta.cdf(x_values, alpha, beta_val)

            self.ax.plot(x_values, cdf_values)
            self.ax.set_title("Fonction de Répartition Cumulative (CDF)")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Probabilité cumulée")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")

    def plot_graph(self):
        try:
            self.ax.clear()

            alpha = float(self.alpha_entry.get())
            beta_val = float(self.beta_entry.get())

            x_values = np.linspace(0, 1, 1000)
            y_values = beta.pdf(x_values, alpha, beta_val)

            self.ax.plot(x_values, y_values)
            self.ax.set_title("Distribution Beta")
            self.ax.set_xlabel("Valeur de x")
            self.ax.set_ylabel("Densité de probabilité")
            self.ax.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs numériques valides.")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorApp(root)
    root.mainloop()