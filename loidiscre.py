import tkinter as tk
from tkinter import Label, Entry, Button, ttk
from scipy.stats import uniform
from tkinter import messagebox

from scipy.stats import bernoulli, binom, poisson, geom, hypergeom, randint, multinomial, nbinom, skellam, yulesimon,skellam 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tkinter import scrolledtext


class SimulateurApp():
    def __init__(self, master):
        self.master = master
        master.title("Lois de probabilités discrètes")

        # Création d'un style pour les onglets
        self.style = ttk.Style()
        self.style.configure('Tab.TNotebook', tabposition='n', tabmargins=(10, 5, 10, 0))
        self.style.configure('Tab.TNotebook.Tab', font=('Arial', '13', 'bold'))

        self.tabControl = ttk.Notebook(master, style='Tab.TNotebook')
        self.tabControl.pack(expand=1, fill="both")
        scroll_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10)

        # scroll_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=10)
        scroll_text.pack(expand=True, fill='both')

        self.add_tab(SimulateurLoiBenoulli, "Loi Benoulli")
        self.add_tab(SimulateurLoiBinomiale, "Loi Binomiale")
        self.add_tab(SimulateurLoiUniforme, "Loi Uniforme")
        self.add_tab(SimulateurLoiPoisson, "Loi de Poisson")
        self.add_tab(SimulationLoiGeometrique, "Loi Geometrique")
        self.add_tab(SimulationLoiHypergeometrique, "Loi Hypergeometrique")
        self.add_tab(SimulationLoiNegativeBinomiale, "Loi Negative Binomiale")
        self.add_tab(SimulationLoiPoissonBinomiale, "Loi Poisson Binomiale")
        self.add_tab(SimulationYulesimon, "Loi de Yulesimon")
        self.add_tab(SimulationPascal, "Loi de Pascal")
        self.add_tab(SimulationLoiSkellam, "Loi de Skellam")

    
    def add_tab(self, tab_class, tab_title):
        tab = ttk.Frame(self.tabControl)
        self.tabControl.add(tab, text=tab_title)
        tab_class(tab)


class SimulateurLoiBenoulli(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()

    def create_widgets(self):
        # Création d'un canevas
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Ajout d'un défilement pour le canevas
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Ajout d'un cadre pour contenir tous les widgets
        self.main_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW)

        # Description
        self.description_text = """
           La loi de Bernoulli est une distribution de probabilité discrète qui modélise une variable
           aléatoire binaire, c'est-à-dire une variable aléatoire prenant seulement deux valeurs 
           possibles, généralement notées 0 et 1. Cette loi est nommée d'après le mathématicien suisse
           Jacob Bernoulli.
            la formule de Loi de Bernoulli:
             P(X=k)=p^k ⋅ (1−p)^(1−k)
            où :
            p est la probabilité de succès dans un seul essai
            k est la valeur que peut prendre la variable aléatoire (0 ou 1).

            L'espérance E[X] :
            E[X]=p

            La variance V(X) :
            Var(X)=p(1−p)
            """

        self.description_label = tk.Label(self.main_frame, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        # Champ de saisie pour la probabilité
        self.probability_label = tk.Label(self.main_frame, text="Probabilité de succès p :")
        self.probability_label.pack()

        self.probability_value = tk.DoubleVar(value=0.5)  # Valeur initiale par défaut de p
        self.probability_entry = tk.Entry(self.main_frame, textvariable=self.probability_value)
        self.probability_entry.pack()

        # Bouton pour calculer
        self.calculate_button = tk.Button(self.main_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        # Label pour afficher les résultats
        self.result_label = tk.Label(self.main_frame, text="")
        self.result_label.pack()

        # Bouton pour afficher le graphe
        self.graph_button = tk.Button(self.main_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        # Bouton pour afficher la fonction de répartition
        self.cdf_button = tk.Button(self.main_frame, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas_graph = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Création d'une nouvelle figure pour le graphe de la fonction de répartition
        self.figure_cdf, self.ax_cdf = plt.subplots()
        self.canvas_cdf = FigureCanvasTkAgg(self.figure_cdf, master=self.main_frame)
        self.canvas_cdf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Mise à jour de la région de défilement après l'ajout des widgets
        self.main_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def calculate_probability(self):
        try:
            probability = self.probability_value.get()
            if 0 <= probability <= 1:
                # Créer une distribution bernoullienne avec la probabilité donnée
                rv = bernoulli(probability)
                # Calculer la probabilité de succès (k=1)
                success_probability = rv.pmf(1)
                # Calculer l'espérance et la variance
                expectation = probability
                variance = probability * (1 - probability)
                self.result_label.config(text=f"Probabilité de succès (k=1): {success_probability:.4f}\n"
                                              f"Espérance: {expectation:.4f}\n"
                                              f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une probabilité valide.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            probability = self.probability_value.get()
            if 0 <= probability <= 1:
                # Créer une distribution bernoullienne avec la probabilité donnée
                rv = bernoulli(probability)
                # Générer les valeurs possibles (0 et 1)
                values = np.arange(2)
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xticklabels(["Échec (0)", "Succès (1)"])
                self.ax.set_ylabel("Probabilité")

                self.ax.grid(True)
                self.canvas_graph.draw()
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une probabilité valide.")

    def plot_cdf(self):
        try:
            self.ax_cdf.clear()  # Nettoie le graphique précédent

            probability = self.probability_value.get()
            if 0 <= probability <= 1:
                # Créer une distribution bernoullienne avec la probabilité donnée
                rv = bernoulli(probability)
                # Générer les valeurs possibles (0 et 1)
                values = np.arange(2)
                # Calculer les valeurs de la fonction de répartition (CDF)
                cdf_values = rv.cdf(values)
                # Tracer la fonction de répartition
                self.ax_cdf.step(values, cdf_values, where='post')
                self.ax_cdf.set_xticks(values)
                self.ax_cdf.set_xticklabels(["Échec (0)", "Succès (1)"])
                self.ax_cdf.set_ylabel("Fonction de répartition")

                self.ax_cdf.grid(True)
                self.canvas_cdf.draw()
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une probabilité valide.")

class SimulateurLoiBinomiale(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()

    def create_widgets(self):
        # Création d'un canevas
        self.canvas = tk.Canvas(self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Ajout d'un défilement pour le canevas
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Ajout d'un cadre pour contenir tous les widgets
        self.main_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW)

        # Description
        self.description_text = """
           La loi binomiale est une distribution de probabilité discrète qui modélise le nombre de succès dans 
           une séquence fixe d'essais indépendants et identiquement distribués. Elle est souvent utilisée dans
           le contexte d'expériences aléatoires où chaque essai a seulement deux résultats possibles, 
           généralement étiquetés comme succès et échec.
            la formule de loi Binomiale:
             P(X=k)=n!/(k!(n−k)!) ​× p^k × (1−p)^(n−k)
            où :
            n est Nombre d'essais
            p est Probabilité de succès
            k est la valeur que peut prendre la variable aléatoire (0 ou 1).

            L'espérance (ou moyenne) d'une distribution binomiale est calculée comme suit :
            E(X)=n⋅p

            La variance d'une distribution binomiale est calculée comme suit :
            Var(X)=n⋅p⋅(1−p)
            """

        self.description_label = tk.Label(self.main_frame, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.n_label = tk.Label(self.main_frame, text="Nombre d'essais:")
        self.n_label.pack()

        self.trials_entry = tk.Entry(self.main_frame)
        self.trials_entry.insert(tk.END, "20")  # Valeur par défaut
        self.trials_entry.pack()

        self.probability_label = tk.Label(self.main_frame, text="Probabilité de succès:")
        self.probability_label.pack()

        self.probability_entry = tk.Entry(self.main_frame)
        self.probability_entry.insert(tk.END, "0.4")  # Valeur par défaut
        self.probability_entry.pack()

        self.calculate_button = tk.Button(self.main_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(self.main_frame, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(self.main_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        # Bouton pour afficher la fonction de répartition
        self.cdf_button = tk.Button(self.main_frame, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.matplotlib_canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.matplotlib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Création d'une nouvelle figure pour le graphe de la fonction de répartition
        self.figure_cdf, self.ax_cdf = plt.subplots()
        self.canvas_cdf = FigureCanvasTkAgg(self.figure_cdf, master=self.main_frame)
        self.canvas_cdf.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Mise à jour de la région de défilement après l'ajout des widgets
        self.main_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def calculate_probability(self):
        try:
            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            n = int(self.trials_entry.get())  # Nombre d'essais
            probability = float(self.probability_entry.get())  # Probabilité de succès
            if 0 <= probability <= 1:
                # Créer une distribution binomiale avec le nombre d'essais et la probabilité donnés
                rv = binom(n, probability)
                # Calculer la probabilité de succès (k=1)
                success_probability = rv.pmf(1)
                # Calculer l'espérance et la variance
                expectation = n * probability
                variance = n * probability * (1 - probability)
                self.result_label.config(text=f"Probabilité de succès (k=1): {success_probability:.4f}\n"
                                              f"Espérance: {expectation:.4f}\n"
                                              f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_graph(self):
        try:
            if self.ax is not None:
                self.ax.clear()  # Nettoie le graphique précédent
            else:
                self.ax = self.figure.add_subplot(111)

            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            n = int(self.trials_entry.get())  # Nombre d'essais
            probability = float(self.probability_entry.get())  # Probabilité de succès
            if 0 <= probability <= 1:
                # Créer une distribution binomiale avec le nombre d'essais et la probabilité donnés
                rv = binom(n, probability)
                # Générer les valeurs possibles (de 0 à n)
                values = np.arange(n + 1)
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre de succès")
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title("Distribution de la loi binomiale")

                self.ax.grid(True)
                self.matplotlib_canvas.draw()
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_cdf(self):
        try:
            self.ax_cdf.clear()  # Nettoie le graphique précédent

            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            n = int(self.trials_entry.get())  # Nombre d'essais
            probability = float(self.probability_entry.get())  # Probabilité de succès
            if 0 <= probability <= 1:
                # Créer une distribution binomiale avec le nombre d'essais et la probabilité donnés
                rv = binom(n, probability)
                # Générer les valeurs possibles (de 0 à n)
                values = np.arange(n + 1)
                # Calculer les valeurs de la fonction de répartition (CDF)
                cdf_values = rv.cdf(values)
                # Tracer la fonction de répartition
                self.ax_cdf.step(values, cdf_values, where='post')
                self.ax_cdf.set_xlabel("Nombre de succès")
                self.ax_cdf.set_ylabel("Fonction de répartition")
                self.ax_cdf.set_title("Fonction de répartition de la loi binomiale")

                self.ax_cdf.grid(True)
                self.canvas_cdf.draw()
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")



class SimulateurLoiUniforme:
    def __init__(self, master):
        self.master = master
        
        self.description_text = """
           La loi uniforme discrète est une distribution de probabilité où chaque valeur possible a 
           la même probabilité d'occurrence. C'est une distribution symétrique où chaque résultat 
           est également probable.
           Pour une variable aléatoire X qui suit une loi uniforme discrète sur l'ensemble 
           {x1, x2, ..., xn}, la fonction de masse de probabilité (pmf) est définie comme suit :
            P(X=x) = 1/n pour x dans {x1, x2, ..., xn}
            où n est le nombre total de valeurs possibles.

            L'espérance E[X] :
            E[X]=(a+b)/2

            La variance V(X) :
            V(X)=(b−a)^2/12

            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.num_values_label = tk.Label(master, text="Nombre de valeurs possibles:")
        self.num_values_label.pack()
        

        self.num_values_entry = tk.Entry(master)
        self.num_values_entry.insert(tk.END, "6")  # Valeur par défaut
        self.num_values_entry.pack()
       
        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.mean_label = tk.Label(master, text="")
        self.mean_label.pack()

        self.variance_label = tk.Label(master, text="")
        self.variance_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.graph_cdf_button = tk.Button(master, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.graph_cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            num_values = int(self.num_values_entry.get())
            if num_values > 0:
                probability = 1 / num_values  # Probabilité pour chaque valeur possible
                self.result_label.config(text=f"Probabilité pour chaque valeur possible : {probability:.4f}")
                
                # Calculer l'espérance et la variance pour une loi uniforme
                mean = (num_values + 1) / 2
                variance = ((num_values - 1) ** 2 - 1) / 12
                
                self.mean_label.config(text=f"Espérance : {mean:.4f}")
                self.variance_label.config(text=f"Variance : {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre de valeurs positives.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre valide.")

    def plot_graph(self):
        try:
            self.ax[0].clear()  # Nettoie le graphique précédent

            num_values = int(self.num_values_entry.get())
            if num_values > 0:
                probabilities = np.ones(num_values) / num_values  # Probabilité pour chaque valeur possible
                
                # Créer une ligne horizontale avec les probabilités
                self.ax[0].hlines(probabilities, xmin=np.arange(1, num_values + 1), xmax=np.arange(2, num_values + 2), color='b', lw=2)
                self.ax[0].set_yticks(probabilities)
                self.ax[0].set_yticklabels([f"{p:.4f}" for p in probabilities])
                self.ax[0].set_xlabel("Valeurs possibles")
                self.ax[0].set_ylabel("Probabilité")
                self.ax[0].set_title("Distribution de la loi uniforme discrète")

                self.ax[0].grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre de valeurs positives.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre valide.")

    def plot_cdf(self):
        try:
            self.ax[1].clear()  # Nettoie le graphique précédent

            num_values = int(self.num_values_entry.get())
            if num_values > 0:
                probabilities = np.ones(num_values) / num_values  # Probabilité pour chaque valeur possible
                cdf_values = np.cumsum(probabilities)  # Fonction de répartition
                
                # Tracer la fonction de répartition
                self.ax[1].step(np.arange(1, num_values + 1), cdf_values, where='mid')
                self.ax[1].set_xlabel("Valeurs possibles")
                self.ax[1].set_ylabel("Fonction de répartition")
                self.ax[1].set_title("Fonction de répartition de la loi uniforme discrète")

                self.ax[1].grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Veuillez entrer un nombre de valeurs positives.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre valide.")

class SimulateurLoiPoisson(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)

        self.description_text = """
            La loi de Poisson est une distribution de probabilité discrète qui
            modélise le nombre d'événements rares se produisant dans un 
            intervalle fixe de temps ou d'espace. Elle est caractérisée par un
            seul paramètre, lambda (λ), qui représente le taux moyen d'occurrence
            des événements. La loi de Poisson est souvent utilisée pour modéliser
            des phénomènes tels que le nombre d'appels reçus par un centre d'appels
            en une heure donnée, le nombre de défaillances d'un système, ou d'autres
            occurrences rares et indépendantes.
            la formule de Loi de Poisson:
             P(X=k)= (e^(−λ) ​× λ^k​) / K!
            où :
            X est la variable aléatoire représentant le nombre d'événements,
            k est le nombre spécifique d'événements
            λ est le taux moyen d'occurrence.

            Espérance (E[X]) :
            E[X] = λ

            Variance (V[X]) :
            V[X] = λ
            """

        self.description_label = tk.Label(self, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.lambda_label = tk.Label(self, text="Paramètre lambda:")
        self.lambda_label.pack()

        self.probability_entry = tk.Entry(self)
        self.probability_entry.insert(tk.END, "40")  # Valeur par défaut
        self.probability_entry.pack()

        self.calculate_button = tk.Button(self, text="Calculer (Poisson)", command=self.calculate_probability_poisson)
        self.calculate_button.pack()

        self.result_label = tk.Label(self, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(self, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.graph_cdf_button = tk.Button(self, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.graph_cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack()

    def calculate_probability_poisson(self):
        try:
            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            lambda_value = float(self.probability_entry.get())  # Paramètre lambda de la loi de Poisson

            # Vérifier que le paramètre lambda est positif
            if lambda_value <= 0:
                messagebox.showerror("Erreur", "Le paramètre lambda doit être supérieur à zéro.")
                return

            # Créer une distribution de Poisson avec le paramètre lambda donné
            rv = poisson(lambda_value)

            # Calculer la probabilité pour un certain nombre d'événements (par exemple, k=1)
            k_value = 1
            poisson_probability = rv.pmf(k_value)

            # Calculer l'espérance et la variance
            expectation = lambda_value
            variance = lambda_value

            # Afficher la probabilité calculée
            self.result_label.config(text=f"Probabilité pour {k_value} événement(s): {poisson_probability:.4f}\n"
                                          f"Espérance: {expectation:.4f}\n"
                                          f"Variance: {variance:.4f}")

        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un paramètre lambda valide.")

    def plot_graph(self):
        try:
            if self.ax[0] is not None:
                self.ax[0].clear()  # Nettoie le graphique précédent
            else:
                self.ax[0] = self.figure.add_subplot(211)

            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            lambda_value = float(self.probability_entry.get())  # Paramètre lambda de la loi de Poisson

            # Vérifier que le paramètre lambda est positif
            if lambda_value <= 0:
                messagebox.showerror("Erreur", "Le paramètre lambda doit être supérieur à zéro.")
                return

            # Créer une distribution de Poisson avec le paramètre lambda donné
            rv = poisson(lambda_value)

            # Générer les valeurs possibles (de 0 à n, où n est un nombre raisonnable)
            values = np.arange(0, int(lambda_value * 5) + 1)
            # Calculer les probabilités associées
            probabilities = rv.pmf(values)

            # Créer un graphe à barres
            self.ax[0].bar(values, probabilities, align="center", alpha=0.75)
            self.ax[0].set_xticks(values)
            self.ax[0].set_xlabel("Nombre d'événements")
            self.ax[0].set_ylabel("Probabilité")
            self.ax[0].set_title("Distribution de la loi de Poisson")

            self.ax[0].grid(True)

            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un paramètre lambda valide.")

    def plot_cdf(self):
        try:
            if self.ax[1] is not None:
                self.ax[1].clear()  # Nettoie le graphique précédent
            else:
                self.ax[1] = self.figure.add_subplot(212)

            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            lambda_value = float(self.probability_entry.get())  # Paramètre lambda de la loi de Poisson

            # Vérifier que le paramètre lambda est positif
            if lambda_value <= 0:
                messagebox.showerror("Erreur", "Le paramètre lambda doit être supérieur à zéro.")
                return

            # Créer une distribution de Poisson avec le paramètre lambda donné
            rv = poisson(lambda_value)

            # Générer les valeurs possibles (de 0 à n, où n est un nombre raisonnable)
            values = np.arange(0, int(lambda_value * 5) + 1)
            # Calculer la fonction de répartition pour ces valeurs
            cdf_values = rv.cdf(values)

            # Tracer la fonction de répartition
            self.ax[1].step(values, cdf_values, where='post')
            self.ax[1].set_xlabel("Nombre d'événements")
            self.ax[1].set_ylabel("Probabilité cumulée")
            self.ax[1].set_title("Fonction de répartition de la loi de Poisson")

            self.ax[1].grid(True)

            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un paramètre lambda valide.")

class SimulationLoiGeometrique:
    def __init__(self, master):
        self.master = master
        
        self.description_text = """
           La loi géométrique modélise le nombre d'essais indépendants 
           nécessaires avant d'observer le premier succès dans une 
           séquence de Bernoulli, où chaque essai a seulement deux 
           résultats possibles (succès ou échec) et la probabilité
           de succès est constante. Elle est souvent utilisée pour
           modéliser des phénomènes tels que le nombre de 
           tentatives avant de réussir à effectuer une tâche 
           spécifique pour la première fois.

            la formule de Loi Geometrique:
             P(X=k)=p ​× (1−p)^(1−k)
            où :
            p est la probabilité de succès
            k est la valeur que peut prendre la variable aléatoire (0 ou 1).

            Espérance E(X) :
            E(X)=1/p

            Variance V(X) :
            V(X)=(1−p)/p^2
            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.probability_label = tk.Label(master, text="Probabilité de succès:")
        self.probability_label.pack()

        self.probability_entry = tk.Entry(master)
        self.probability_entry.insert(tk.END, "0.5")  # Valeur par défaut
        self.probability_entry.pack()

        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.graph_cdf_button = tk.Button(master, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.graph_cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots(2, 1, figsize=(6, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            probability = float(self.probability_entry.get())
            if 0 <= probability <= 1:
                # Créer une distribution géométrique avec la probabilité donnée
                rv = geom(probability)
                # Calculer la probabilité d'observer le premier succès (k=1)
                success_probability = rv.pmf(1)
                # Calculer l'espérance et la variance
                expectation = 1 / probability
                variance = (1 - probability) / (probability ** 2)
                self.result_label.config(text=f"Probabilité du premier succès (k=1): {success_probability:.4f}\n"
                                              f"Espérance: {expectation:.4f}\n"
                                              f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une probabilité valide.")

    def plot_graph(self):
        try:
            self.ax[0].clear()  # Nettoie le graphique précédent

            probability = float(self.probability_entry.get())
            if 0 <= probability <= 1:
                # Créer une distribution géométrique avec la probabilité donnée
                rv = geom(probability)
                # Générer les valeurs possibles (1 jusqu'à un nombre raisonnable)
                values = np.arange(1, 11)  # Exemple : jusqu'à 10 tentatives
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax[0].bar(values, probabilities, align="center", alpha=0.75)
                self.ax[0].set_xticks(values)
                self.ax[0].set_xlabel("Nombre d'essais")
                self.ax[0].set_ylabel("Probabilité")
                self.ax[0].set_title("Distribution de la loi géométrique")

                self.ax[0].grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une probabilité valide.")

    def plot_cdf(self):
        try:
            self.ax[1].clear()  # Nettoie le graphique précédent

            probability = float(self.probability_entry.get())
            if 0 <= probability <= 1:
                # Créer une distribution géométrique avec la probabilité donnée
                rv = geom(probability)
                # Générer les valeurs possibles (1 jusqu'à un nombre raisonnable)
                values = np.arange(1, 11)  # Exemple : jusqu'à 10 tentatives
                # Calculer les fonctions de répartition associées
                cdf_values = rv.cdf(values)
                # Tracer la fonction de répartition
                self.ax[1].step(values, cdf_values, where='post')
                self.ax[1].set_xlabel("Nombre d'essais")
                self.ax[1].set_ylabel("Probabilité cumulée")
                self.ax[1].set_title("Fonction de répartition de la loi géométrique")

                self.ax[1].grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "La probabilité doit être comprise entre 0 et 1.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une probabilité valide.")


class SimulationLoiHypergeometrique:
    def __init__(self, master):
        self.master = master
        
        self.description_text = """
          La loi hypergéométrique modélise le nombre d'éléments d'une 
          population finie qui sont classés dans une catégorie particulière,
          après avoir effectué un échantillonnage sans remplacement. Contrairement
          à la loi binomiale qui modélise des échantillonnages avec remplacement, 
          la loi hypergéométrique prend en compte le fait que la taille de la population
          diminue à chaque tirage sans remplacement.

            la formule de Loi Hypergeometrique:
             P(X=i)=(K!×(n-K)!× n!×(N-n)!) /(N!×(K−i)!×(n-i)!×(N-K-n+i)!) ​× p^k × (1−p)^(n−k)
            où :
            N : Taille totale de la population.
            K : Nombre total d'éléments de la catégorie dans la population.
            n : Taille de l'échantillon (le nombre d'éléments que vous extrayez de la population sans remplacement).
            k : Nombre d'éléments de la catégorie que vous souhaitez observer dans l'échantillon

            Espérance (ou moyenne)
            E(X)=nK/N

            variance:
            V(X)=nK(N−K)(N−n)/(N^2(N−1))
            """
        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.total_population_label = tk.Label(master, text="Taille de la population:")
        self.total_population_label.pack()

        self.total_population_entry = tk.Entry(master)
        self.total_population_entry.insert(tk.END, "100")  # Valeur par défaut
        self.total_population_entry.pack()

        self.success_population_label = tk.Label(master, text="Nombre d'éléments de la catégorie:")
        self.success_population_label.pack()

        self.success_population_entry = tk.Entry(master)
        self.success_population_entry.insert(tk.END, "53")  # Valeur par défaut
        self.success_population_entry.pack()

        self.sample_size_label = tk.Label(master, text="Taille de l'échantillon:")
        self.sample_size_label.pack()

        self.sample_size_entry = tk.Entry(master)
        self.sample_size_entry.insert(tk.END, "10")  # Valeur par défaut
        self.sample_size_entry.pack()

        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            N = int(self.total_population_entry.get())  # Taille de la population
            K = int(self.success_population_entry.get())  # Nombre d'éléments de la catégorie dans la population
            n = int(self.sample_size_entry.get())  # Taille de l'échantillon

            if 0 <= K <= N and 0 <= n <= N:
                # Créer une distribution hypergéométrique avec les paramètres donnés
                rv = hypergeom(N, K, n)
                # Calculer la probabilité d'obtenir k éléments de la catégorie dans l'échantillon
                k_value = int(self.success_population_entry.get())
                probability = rv.pmf(k_value)
                
                # Calculer l'espérance et la variance
                expectation = n * K / N
                variance = n * K * (N - K) * (N - n) / (N ** 2 * (N - 1))
                
                self.result_label.config(text=f"Probabilité d'obtenir {k_value} éléments de la catégorie: {probability:.4f}\n"
                                            f"Espérance: {expectation:.4f}\n"
                                            f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la population, de la catégorie et de l'échantillon.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            N = int(self.total_population_entry.get())  # Taille de la population
            K = int(self.success_population_entry.get())  # Nombre d'éléments de la catégorie dans la population
            n = int(self.sample_size_entry.get())  # Taille de l'échantillon

            if 0 <= K <= N and 0 <= n <= N:
                # Créer une distribution hypergéométrique avec les paramètres donnés
                rv = hypergeom(N, K, n)
                # Générer les valeurs possibles
                values = np.arange(max(0, n - (N - K)), min(n, K) + 1)
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'éléments de la catégorie dans l'échantillon")
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title("Distribution de la loi hypergéométrique")

                self.ax.grid(True)
                self.canvas.draw()
                
                # Afficher la fonction de répartition cumulative
                cumulative_values, cumulative_probabilities = self.cumulative_distribution_function(values, probabilities)
                self.plot_cumulative_distribution(cumulative_values, cumulative_probabilities)
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la population, de la catégorie et de l'échantillon.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def cumulative_distribution_function(self, values, probabilities):
        # Fonction de répartition cumulée
        cumulative_probabilities = np.cumsum(probabilities)
        return values, cumulative_probabilities

    def plot_cumulative_distribution(self, values, cumulative_probabilities):
        # Créer un graphe pour la fonction de répartition cumulative
        self.ax.plot(values, cumulative_probabilities, marker='o', linestyle='-')
        self.ax.set_xlabel("Nombre d'éléments de la catégorie dans l'échantillon")
        self.ax.set_ylabel("Probabilité cumulée")
        self.ax.set_title("Fonction de répartition cumulative de la loi hypergéométrique")
        self.ax.grid(True)
        self.canvas.draw()




class SimulationLoiNegativeBinomiale:
    def __init__(self, master):
        self.master = master
        self.description_text = """
            La loi négative binomiale modélise le nombre d'essais indépendants nécessaires pour
            observer un nombre spécifié de succès dans une séquence de Bernoulli, où chaque essai
            a seulement deux résultats possibles (succès ou échec) et la probabilité de succès est
            constante. Elle diffère de la loi binomiale en ce sens qu'elle ne fixe pas un nombre 
            prédéterminé d'essais, mais plutôt un nombre prédéterminé de succès.
            la formule de Simulateur de Loi Negative Binomiale :
             P(X=k)=(p^k ⋅ (1−p)^k ⋅ p^r) / (k! ⋅ (r-1)!)
            où :
            p est la probabilité de succès à chaque essai,
            k est le nombre d'essais avant d'obtenir rr succès.
            r représente le nombre total de succès que l'on souhaite observer.
            p est la probabilité de succès à chaque essai.

            L'espérance E(X) :
            E(X)=r/p

            La variance V(X) :
            V(X)=r(1−p)/p^2

            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.r_label = tk.Label(master, text="Nombre total de succès souhaités:")
        self.r_label.pack()

        # Valeur par défaut pour "Nombre total de succès souhaités"
        self.total_success_entry = tk.Entry(master)
        self.total_success_entry.insert(tk.END, "11")
        self.total_success_entry.pack()

        self.probability_label = tk.Label(master, text="Probabilité de succès à chaque essai:")
        self.probability_label.pack()

        # Valeur par défaut pour "Probabilité de succès à chaque essai"
        self.probability_entry = tk.Entry(master)
        self.probability_entry.insert(tk.END, "0.04")
        self.probability_entry.pack()

        self.trials_label = tk.Label(master, text="Nombre d'essais:")
        self.trials_label.pack()

        # Valeur par défaut pour "Nombre d'essais"
        self.trials_entry = tk.Entry(master)
        self.trials_entry.insert(tk.END, "18")
        self.trials_entry.pack()

        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.cumulative_graph_button = tk.Button(master, text="Afficher le graphe de la fonction de répartition cumulative", command=self.plot_cumulative_graph)
        self.cumulative_graph_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

        # Calcul initial
        self.calculate_probability()

    def calculate_probability(self):
        try:
            r = int(self.total_success_entry.get())  # Nombre total de succès souhaités
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and r > 0:
                # Créer une distribution négative binomiale avec les paramètres donnés
                rv = nbinom(r, probability)
                # Calculer la probabilité d'observer k essais avant d'obtenir r succès
                k_value = int(self.trials_entry.get())
                probability = rv.pmf(k_value)
                
                # Calculer l'espérance et la variance
                expectation = r * (1 - probability) / probability
                variance = r * (1 - probability) / (probability ** 2)
                
                self.result_label.config(text=f"Probabilité d'observer {k_value} essais avant d'obtenir {r} succès : {probability:.4f}\n"
                                              f"Espérance: {expectation:.4f}\n"
                                              f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1), le nombre de succès souhaités (supérieur à 0), et le nombre d'essais.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            r = int(self.total_success_entry.get())  # Nombre total de succès souhaités
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and r > 0:
                # Créer une distribution négative binomiale avec les paramètres donnés
                rv = nbinom(r, probability)
                # Générer les valeurs possibles (de 0 à un nombre raisonnable)
                values = np.arange(0, 20)  # Exemple : jusqu'à 20 essais
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'essais avant d'obtenir le nombre de succès souhaités")
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title(f"Distribution de la loi négative binomiale (r={r}, p={probability})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1), le nombre de succès souhaités (supérieur à 0), et le nombre d'essais.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_cumulative_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            r = int(self.total_success_entry.get())  # Nombre total de succès souhaités
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and r > 0:
                # Créer une distribution négative binomiale avec les paramètres donnés
                rv = nbinom(r, probability)
                # Générer les valeurs possibles (de 0 à un nombre raisonnable)
                values = np.arange(0, 20)  # Exemple : jusqu'à 20 essais
                # Calculer les probabilités associées
                probabilities = rv.cdf(values)
                # Créer un graphe à barres
                self.ax.plot(values, probabilities, marker='o', linestyle='-')
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'essais avant d'obtenir le nombre de succès souhaités")
                self.ax.set_ylabel("Probabilité cumulative")
                self.ax.set_title(f"Fonction de répartition cumulative de la loi négative binomiale (r={r}, p={probability})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1), le nombre de succès souhaités (supérieur à 0), et le nombre d'essais.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

class SimulationLoiPoissonBinomiale:
    def __init__(self, master):
        self.master = master
        self.description_text = """
            La loi Poisson binomiale est une distribution de probabilité qui 
            combine les caractéristiques de la loi binomiale et de la loi de 
            Poisson. Elle modélise le nombre d'occurrences d'un événement rare
            dans un nombre fixe d'essais indépendants, où chaque essai a seulement 
            deux résultats possibles (succès ou échec).

            la formule de Simulateur de Loi Negative Binomiale :
             P(X=k)=(n! ⋅ (1-p)^(n-k) ⋅ e^(-λ) ⋅ λ^k) / (k!⋅ k! ⋅(n-k)!)
            où :
            n est le nombre total d'essais,
            k est le nombre d'occurrences de succès,
            p est la probabilité de succès dans chaque essai,
            λ est le taux d'occurrence moyen dans la distribution de Poisson associée.

            Espérance (E[X]) :
            E[X]=n×p

            Variance (V[X]) :
            V[X]=n×p×(1−p)

            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        # Ajouter un champ d'entrée et une étiquette pour le nombre d'essais (n)
        self.n_label = tk.Label(master, text="Nombre d'essais:")
        self.n_label.pack()

        self.trials_entry = tk.Entry(master)
        self.trials_entry.pack()
        self.trials_entry.insert(tk.END, "100")  # Valeur par défaut

        # Garder le champ d'entrée pour la probabilité (probability)
        self.probability_label = tk.Label(master, text="Probabilité de succès:")
        self.probability_label.pack()

        self.probability_entry = tk.Entry(master)
        self.probability_entry.pack()
        self.probability_entry.insert(tk.END, "0.4")  # Valeur par défaut

        # Ajouter un bouton de calcul et garder les widgets du résultat et du graphe
        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.cdf_button = tk.Button(master, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            n = int(self.trials_entry.get())  # Nombre total d'essais
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and n > 0:
                # Créer une distribution binomiale avec les paramètres donnés
                rv = binom(n, probability)
                # Calculer la probabilité d'observer k succès dans n essais
                k_value = int(self.trials_entry.get())
                probability = rv.pmf(k_value)
                # Calculer l'espérance et la variance
                expectation = n * probability
                variance = n * probability * (1 - probability)
                self.result_label.config(text=f"Probabilité d'observer {k_value} succès dans {n} essais : {probability:.4f}\n"
                                              f"Espérance: {expectation:.4f}\n"
                                              f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1) et le nombre d'essais (supérieur à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            n = int(self.trials_entry.get())  # Nombre total d'essais
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and n > 0:
                # Créer une distribution binomiale avec les paramètres donnés
                rv = binom(n, probability)
                # Générer les valeurs possibles (de 0 à n)
                values = np.arange(0, n + 1)
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre de succès dans les essais")
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title(f"Distribution de la loi binomiale (n={n}, p={probability})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1) et le nombre d'essais (supérieur à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            n = int(self.trials_entry.get())  # Nombre total d'essais
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and n > 0:
                # Créer une distribution binomiale avec les paramètres donnés
                rv = binom(n, probability)
                # Générer les valeurs possibles (de 0 à n)
                values = np.arange(0, n + 1)
                # Calculer les probabilités cumulatives associées
                cdf = rv.cdf(values)
                # Créer un graphe de la fonction de répartition cumulative (CDF)
                self.ax.plot(values, cdf, marker='o')
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre de succès dans les essais")
                self.ax.set_ylabel("Probabilité cumulée")
                self.ax.set_title(f"Fonction de répartition cumulative de la loi binomiale (n={n}, p={probability})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1) et le nombre d'essais (supérieur à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")


class SimulationYulesimon:
    def __init__(self, master):
        self.master = master
        
        self.description_text = """
           La loi de Yule-Simon, également appelée distribution de Yule, est une distribution de 
           probabilité discrète qui modélise la distribution du nombre d'événements dans une séquence 
           de processus de naissance. Elle est souvent utilisée pour décrire des phénomènes où de nouveaux 
           éléments apparaissent au fil du temps, et la probabilité d'apparition d'un nouvel élément dépend 
           du nombre d'éléments déjà présents.
            la formule de Loi de Yule_Simon:
             P(X=k)=α​ / k(k+α)^2
            où :
            α est le paramètre de forme de la distribution, et α>0,
            k est un entier positif représentant le nombre d'événements.

            Espérance (E(X)) :
            E(X)=α/(α−1)

            Variance (V(X)) :
            V(X)=α/((α−1)^2(α−2))

            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.alpha_label = tk.Label(master, text="Paramètre alpha:")
        self.alpha_label.pack()

        self.alpha_entry = tk.Entry(master)
        self.alpha_entry.pack()
        self.alpha_entry.insert(tk.END, "4")  # Valeur par défaut

        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.cdf_button = tk.Button(master, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            alpha = float(self.alpha_entry.get())  # Paramètre de forme de la distribution de Yule-Simon
            if alpha > 0:
                # Calculer la probabilité pour k=1 dans la distribution de Yule-Simon
                probability_k1 = alpha / (1 + alpha)**2
                expectation = alpha / (1 - alpha)
                variance = alpha / ((1 - alpha)**2 * (1 - alpha - 1))
                self.result_label.config(text=f"Probabilité pour k=1 : {probability_k1:.4f}\n"
                                               f"Espérance: {expectation:.4f}\n"
                                               f"Variance: {variance:.4f}")
            else:
                messagebox.showerror("Erreur", "Le paramètre alpha doit être supérieur à 0.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une valeur valide pour le paramètre alpha.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            alpha = float(self.alpha_entry.get())  # Paramètre de forme de la distribution de Yule-Simon

            if alpha > 0:
                # Générer les valeurs possibles (de 1 à un nombre raisonnable)
                values = np.arange(1, 11)  # Exemple : jusqu'à 10 événements
                # Calculer les probabilités associées pour la distribution de Yule-Simon
                probabilities = alpha / (values * (1 + alpha)**2)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'événements")
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title(f"Distribution de Yule-Simon (alpha={alpha})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Le paramètre alpha doit être supérieur à 0.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une valeur valide pour le paramètre alpha.")

    def plot_cdf(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            alpha = float(self.alpha_entry.get())  # Paramètre de forme de la distribution de Yule-Simon

            if alpha > 0:
                # Générer les valeurs possibles (de 1 à un nombre raisonnable)
                values = np.arange(1, 11)  # Exemple : jusqu'à 10 événements
                # Calculer les probabilités associées pour la distribution de Yule-Simon
                probabilities = alpha / (values * (1 + alpha)**2)
                # Calculer la fonction de répartition (CDF) en prenant la somme cumulée des probabilités
                cdf = np.cumsum(probabilities)
                # Créer un graphe pour la fonction de répartition
                self.ax.plot(values, cdf, marker='o', linestyle='-')
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'événements")
                self.ax.set_ylabel("Probabilité cumulée")
                self.ax.set_title(f"Fonction de répartition de Yule-Simon (alpha={alpha})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Le paramètre alpha doit être supérieur à 0.")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une valeur valide pour le paramètre alpha.")




class SimulationPascal:
    def __init__(self, master):
        self.master = master

        self.description_text = """
           La loi de Pascal, également connue sous le nom de distribution binomiale négative, 
           est une distribution de probabilité discrète qui modélise le nombre d'essais nécessaires
            avant d'obtenir un certain nombre prédéfini de succès dans une séquence d'essais indépendants.
            Elle est souvent utilisée pour modéliser des situations où l'on compte le nombre d'échecs avant
            d'atteindre un nombre fixe de succès dans un processus de Bernoulli.
            la formule de Loi Pascal:
             P(X=k)=(k-1)!​×p^r​×(1-p)^(k-r) /((r-1)!​×(k-r)!)
            où :
            p est la probabilité de succès à chaque essai,
            r est le nombre total de succès souhaités,
            k est le nombre d'essais avant d'obtenir rr succès.

            L'espérance (E[X]):
            E[X]=r/p

            La variance (Var[X]):
            Var[X]=r(1−p)/p^2
            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.trials_label = tk.Label(master, text="Nombre d'essais:")
        self.trials_label.pack()

        self.trials_entry = tk.Entry(master)
        self.trials_entry.pack()
        self.trials_entry.insert(tk.END, "1000")  # Valeur par défaut

        self.total_success_label = tk.Label(master, text="Nombre total de succès souhaités:")
        self.total_success_label.pack()

        self.total_success_entry = tk.Entry(master)
        self.total_success_entry.pack()
        self.total_success_entry.insert(tk.END, "100")  # Valeur par défaut

        self.probability_label = tk.Label(master, text="Probabilité de succès à chaque essai:")
        self.probability_label.pack()

        self.probability_entry = tk.Entry(master)
        self.probability_entry.pack()
        self.probability_entry.insert(tk.END, "0.7")  # Valeur par défaut

        self.calculate_button = tk.Button(master, text="Calculer", command=self.calculate_probability)
        self.calculate_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.expectation_label = tk.Label(master, text="")
        self.expectation_label.pack()

        self.variance_label = tk.Label(master, text="")
        self.variance_label.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        self.cdf_button = tk.Button(master, text="Afficher la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def calculate_probability(self):
        try:
            # Récupérer les valeurs nécessaires depuis l'interface utilisateur
            trials = int(self.trials_entry.get())  # Nombre d'essais
            r = int(self.total_success_entry.get())  # Nombre total de succès souhaités
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and r > 0 and trials > 0:
                # Créer une distribution négative binomiale (Pascal) avec les paramètres donnés
                rv = nbinom(r, probability)
                # Calculer l'espérance et la variance
                expectation = rv.mean()
                variance = rv.var()
                self.expectation_label.config(text=f"Espérance : {expectation:.4f}")
                self.variance_label.config(text=f"Variance : {variance:.4f}")

                # Calculer la probabilité d'observer k essais avant d'obtenir r succès
                k_value = trials
                probability = rv.pmf(k_value)
                self.result_label.config(text=f"Probabilité d'observer {k_value} essais avant d'obtenir {r} succès : {probability:.4f}")
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1), le nombre de succès souhaités (supérieur à 0) et le nombre d'essais (supérieur à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            trials = int(self.trials_entry.get())  # Nombre d'essais
            r = int(self.total_success_entry.get())  # Nombre total de succès souhaités
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and r > 0 and trials > 0:
                # Créer une distribution négative binomiale (Pascal) avec les paramètres donnés
                rv = nbinom(r, probability)
                # Générer les valeurs possibles (de 0 à un nombre raisonnable)
                values = np.arange(0, 20)  # Exemple : jusqu'à 20 essais
                # Calculer les probabilités associées
                probabilities = rv.pmf(values)
                # Créer un graphe à barres
                self.ax.bar(values, probabilities, align="center", alpha=0.75)
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'essais avant d'obtenir le nombre de succès souhaités")
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title(f"Distribution de la loi de Pascal (r={r}, p={probability})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1), le nombre de succès souhaités (supérieur à 0) et le nombre d'essais (supérieur à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    def plot_cdf(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            trials = int(self.trials_entry.get())  # Nombre d'essais
            r = int(self.total_success_entry.get())  # Nombre total de succès souhaités
            probability = float(self.probability_entry.get())  # Probabilité de succès à chaque essai

            if 0 <= probability <= 1 and r > 0 and trials > 0:
                # Créer une distribution négative binomiale (Pascal) avec les paramètres donnés
                rv = nbinom(r, probability)
                # Générer les valeurs possibles (de 0 à un nombre raisonnable)
                values = np.arange(0, 20)  # Exemple : jusqu'à 20 essais
                # Calculer la fonction de répartition (CDF)
                cdf = rv.cdf(values)
                # Créer un graphe pour la fonction de répartition
                self.ax.plot(values, cdf, marker='o', linestyle='-')
                self.ax.set_xticks(values)
                self.ax.set_xlabel("Nombre d'essais avant d'obtenir le nombre de succès souhaités")
                self.ax.set_ylabel("Probabilité cumulée")
                self.ax.set_title(f"Fonction de répartition de la loi de Pascal (r={r}, p={probability})")

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de la probabilité (entre 0 et 1), le nombre de succès souhaités (supérieur à 0) et le nombre d'essais (supérieur à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

class SimulationLoiSkellam:
    def __init__(self, master):
        self.master = master

        self.description_text = """
           La loi de Skellam est une distribution de probabilité discrète qui modélise la 
           différence entre deux variables aléatoires indépendantes et identiquement distribuées 
           qui suivent la distribution de Poisson. Elle est souvent utilisée pour modéliser la 
           différence entre les nombres d'événements qui se produisent dans deux régions ou à 
           deux moments différents, lorsque ces événements sont rares et indépendants.
            la formule de Loi Skellam:
             P(Z=k)=e^(-λ1​-λ2​) ​× (λ2/​λ1​​)^(k/2) ​× Ik
            où :
            Z=X-Y
            λ1​ et λ2​ sont les paramètres de Poisson des variables aléatoires X et Y,
            k est la différence entre les réalisations de X et Y,
            Ik​ est la fonction de Bessel modifiée d'ordre k.

            Eperance E(X):
            E(X−Y)=λ1​−λ2​
            où λ1 et λ2  sont les paramètres de la distribution de Poisson associée aux variables aléatoires X et Y respectivement.

            Variance V(X):
            Var(X−Y)=λ1​+λ2​


            """

        self.description_label = tk.Label(master, text=self.description_text, justify=tk.LEFT)
        self.description_label.pack()

        self.lambda1_label = tk.Label(master, text="Paramètre lambda 1:")
        self.lambda1_label.pack()

        self.lambda1_entry = tk.Entry(master)
        self.lambda1_entry.pack()
        self.lambda1_entry.insert(tk.END, "4")  # Valeur par défaut

        self.lambda2_label = tk.Label(master, text="Paramètre lambda 2:")
        self.lambda2_label.pack()

        self.lambda2_entry = tk.Entry(master)
        self.lambda2_entry.pack()
        self.lambda2_entry.insert(tk.END, "6")  # Valeur par défaut

        self.difference_label = tk.Label(master, text="Différence spécifique (k):")
        self.difference_label.pack()

        self.difference_entry = tk.Entry(master)
        self.difference_entry.pack()
        self.difference_entry.insert(tk.END, "4")  # Valeur par défaut

        self.plot_type = tk.StringVar()
        self.plot_type.set("pmf")  # Par défaut, afficher la PMF

        self.plot_type_label = tk.Label(master, text="Type de graphique:")
        self.plot_type_label.pack()

        self.pmf_radio = tk.Radiobutton(master, text="PMF", variable=self.plot_type, value="pmf")
        self.pmf_radio.pack()

        self.cdf_radio = tk.Radiobutton(master, text="CDF", variable=self.plot_type, value="cdf")
        self.cdf_radio.pack()

        self.graph_button = tk.Button(master, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.pack()

        # Création d'une figure pour le graphique
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def plot_graph(self):
        try:
            self.ax.clear()  # Nettoie le graphique précédent

            lambda_1 = float(self.lambda1_entry.get())  # Paramètre de Poisson pour la première variable aléatoire
            lambda_2 = float(self.lambda2_entry.get())  # Paramètre de Poisson pour la deuxième variable aléatoire

            if lambda_1 >= 0 and lambda_2 >= 0:
                # Créer une distribution de Skellam avec les paramètres donnés
                rv = skellam(lambda_1, lambda_2)

                # Générer les valeurs possibles
                values = np.arange(-20, 21)
                if self.plot_type.get() == "pmf":
                    probabilities = rv.pmf(values)
                    xlabel = "Différence entre les réalisations de X et Y"
                    title = f"Distribution de la loi de Skellam (λ1={lambda_1}, λ2={lambda_2})"
                elif self.plot_type.get() == "cdf":
                    probabilities = rv.cdf(values)
                    xlabel = "Valeur de la variable"
                    title = f"Fonction de répartition cumulative de la loi de Skellam (λ1={lambda_1}, λ2={lambda_2})"

                # Afficher le graphe
                self.ax.plot(values, probabilities, marker='o', linestyle='-')
                self.ax.set_xlabel(xlabel)
                self.ax.set_ylabel("Probabilité")
                self.ax.set_title(title)

                self.ax.grid(True)
                self.canvas.draw()
            else:
                messagebox.showerror("Erreur", "Vérifiez les paramètres de Poisson (doivent être supérieurs ou égaux à 0).")
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")


if __name__ == "__main__":
    root = tk.Tk()
    root.option_add('*Font', 'Arial 14') 
    app = SimulateurApp(root)
    root.mainloop()

