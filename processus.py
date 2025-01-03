import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import poisson

class PoissonSimulator:
    def __init__(self, master):
        self.master = master
        master.title("Simulateur de Processus de Poisson et markov ")

        # Onglets pour homogène et non homogène
        self.tabControl = ttk.Notebook(master)

        self.homogeneous_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.homogeneous_tab, text="Poisson Homogène")
        self.non_homogeneous_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.non_homogeneous_tab, text="Poisson Non Homogène")
        self.markov_tab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.markov_tab, text="Markov")

        self.tabControl.pack(expand=1, fill="both")

        # Initialisation des onglets
        self.init_homogeneous_tab()
        self.init_non_homogeneous_tab()
        self.init_markov()

    def init_homogeneous_tab(self):
        # Cadre pour la description
        self.description_frame = tk.Frame(self.homogeneous_tab)
        self.description_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.description_label = tk.Label(
            self.description_frame,
            text=(
                "Un processus de Poisson homogène modélise des événements à un taux constant (\u03bb).\n"
                "La probabilité de k événements dans un intervalle de temps t est donnée par :\n"
                "P(k; \u03bbt) = (\u03bbt)^k * e^(-\u03bbt) / k!"
            ),
            justify=tk.LEFT,
            wraplength=600,
            font=("Time New Roman", 12)
        )
        self.description_label.grid(row=0, column=0, padx=10, pady=10)

        # Cadre pour les paramètres
        self.parameters_frame = tk.Frame(self.homogeneous_tab)
        self.parameters_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.lambda_label = tk.Label(self.parameters_frame, text="Taux moyen d'événements (\u03bb):")
        self.lambda_label.grid(row=0, column=0, sticky="e")

        self.lambda_entry = tk.Entry(self.parameters_frame)
        self.lambda_entry.grid(row=0, column=1, sticky="w")

        self.time_label = tk.Label(self.parameters_frame, text="Durée de l'intervalle (t):")
        self.time_label.grid(row=1, column=0, sticky="e")

        self.time_entry = tk.Entry(self.parameters_frame)
        self.time_entry.grid(row=1, column=1, sticky="w")

        self.k_label = tk.Label(self.parameters_frame, text="Nombre d'événements (k):")
        self.k_label.grid(row=2, column=0, sticky="e")

        self.k_entry = tk.Entry(self.parameters_frame)
        self.k_entry.grid(row=2, column=1, sticky="w")

        # Cadre pour les actions
        self.actions_frame = tk.Frame(self.homogeneous_tab)
        self.actions_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        self.calculate_button = tk.Button(self.actions_frame, text="Calculer", command=self.calculate_probability)
        self.calculate_button.grid(row=0, column=0)

        self.cdf_button = tk.Button(self.actions_frame, text="Tracer la fonction de répartition", command=self.plot_cdf)
        self.cdf_button.grid(row=0, column=1)

        self.graph_button = tk.Button(self.actions_frame, text="Afficher le graphe", command=self.plot_graph)
        self.graph_button.grid(row=0, column=2)

        # Cadre pour le résultat
        self.result_frame = tk.Frame(self.homogeneous_tab)
        self.result_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")

        self.result_label = tk.Label(self.result_frame, text="")
        self.result_label.pack()

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(self.homogeneous_tab)
        self.graph_frame.grid(row=0, column=2, rowspan=4, sticky="nsew")

        self.figure, self.ax_homog = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
    def calculate_probability(self):
        try:
            lam = float(self.lambda_entry.get())
            t = float(self.time_entry.get())
            k = int(self.k_entry.get())

            mean_rate = lam * t
            probability = poisson.pmf(k, mean_rate)

            self.result_label.config(text=f"P(k={k}; \u03bbt={mean_rate:.2f}) = {probability:.4f}")
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs valides.")
            

    def plot_cdf(self):
        try:
            self.ax_homog.clear()

            lam = float(self.lambda_entry.get())
            t = float(self.time_entry.get())

            mean_rate = lam * t
            k_values = np.arange(0, int(mean_rate) + 15)
            cdf_values = poisson.cdf(k_values, mean_rate)

            self.ax_homog.plot(k_values, cdf_values, marker='o')
            self.ax_homog.set_title("Fonction de répartition cumulative (CDF)")
            self.ax_homog.set_xlabel("Nombre d'événements (k)")
            self.ax_homog.set_ylabel("Probabilité cumulée")
            self.ax_homog.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs valides.")

    def plot_graph(self):
        try:
            self.ax_homog.clear()

            lam = float(self.lambda_entry.get())
            t = float(self.time_entry.get())

            mean_rate = lam * t
            k_values = np.arange(0, int(mean_rate) + 15)
            pmf_values = poisson.pmf(k_values, mean_rate)

            self.ax_homog.bar(k_values, pmf_values, color="skyblue", edgecolor="black")
            self.ax_homog.set_title("Distribution de Poisson")
            self.ax_homog.set_xlabel("Nombre d'événements (k)")
            self.ax_homog.set_ylabel("Probabilité")
            self.ax_homog.grid(True)

            self.canvas.draw()
        except ValueError:
            self.result_label.config(text="Veuillez entrer des valeurs valides.")


    def init_non_homogeneous_tab(self):
        # Cadre pour la description
        self.description_frame_non_homogeneous = tk.Frame(self.non_homogeneous_tab)
        self.description_frame_non_homogeneous.grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.description_label_non_homogeneous = tk.Label(
            self.description_frame_non_homogeneous,
            text=(
                "Un processus de Poisson non homogène a un taux d'intensité \u03bb(t) variable dans le temps.\n"
                "La probabilité de k événements dépend de l'intégrale de \u03bb(t) sur l'intervalle considéré."
            ),
            justify=tk.LEFT,
            wraplength=600,
            font=("Time New Roman", 12)
        )
        self.description_label_non_homogeneous.grid(row=0, column=0, padx=10, pady=10)

        # Cadre pour les paramètres
        self.parameters_frame_non_homogeneous = tk.Frame(self.non_homogeneous_tab)
        self.parameters_frame_non_homogeneous.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Valeurs par défaut des paramètres
        self.default_params = {"m": 2, "n": 3, "c": 4, "k": 10}
        self.param_vars = {param: tk.DoubleVar(value=value) for param, value in self.default_params.items()}

        # Paramètres pour la fonction d'intensité paramétrable
        self.param_frame = tk.Frame(self.parameters_frame_non_homogeneous)
        self.param_frame.grid(row=0, column=0, columnspan=2, pady=10)

        self.param_labels = {}
        self.param_entries = {}
        for i, param in enumerate(["m", "n", "c"]):
            self.param_labels[param] = tk.Label(self.param_frame, text=f"{param}:")
            self.param_labels[param].grid(row=i, column=0, sticky="e")

            self.param_entries[param] = tk.Entry(self.param_frame, textvariable=self.param_vars[param])
            self.param_entries[param].grid(row=i, column=1, sticky="w")
            self.param_entries[param].bind("<KeyRelease>", self.update_lambda_function)

        # Champ pour la fonction \lambda(t)
        self.lambda_function_label = tk.Label(self.parameters_frame_non_homogeneous, text="Fonction \u03bb(t):")
        self.lambda_function_label.grid(row=1, column=0, sticky="e")

        self.lambda_function_entry = tk.Entry(self.parameters_frame_non_homogeneous, state="readonly")
        self.lambda_function_entry.grid(row=1, column=1, sticky="w")
        self.update_lambda_function()

        # Intervalle de temps
        self.time_interval_label = tk.Label(self.parameters_frame_non_homogeneous, text="Intervalle de temps (a, b):")
        self.time_interval_label.grid(row=2, column=0, sticky="e")

        self.time_interval_entry = tk.Entry(self.parameters_frame_non_homogeneous)
        self.time_interval_entry.grid(row=2, column=1, sticky="w")
        self.time_interval_entry.insert(0, "0, 10")

        # Nombre d'événements
        self.k_label_non_homogeneous = tk.Label(self.parameters_frame_non_homogeneous, text="Nombre d'événements (k):")
        self.k_label_non_homogeneous.grid(row=3, column=0, sticky="e")

        self.k_entry_non_homogeneous = tk.Entry(self.parameters_frame_non_homogeneous)
        self.k_entry_non_homogeneous.grid(row=3, column=1, sticky="w")
        self.k_entry_non_homogeneous.insert(0, str(self.default_params["k"]))

        # Cadre pour les actions
        self.actions_frame_non_homogeneous = tk.Frame(self.non_homogeneous_tab)
        self.actions_frame_non_homogeneous.grid(row=4, column=0, columnspan=2, sticky="nsew")

        self.calculate_button_non_homogeneous = tk.Button(
            self.actions_frame_non_homogeneous, text="Calculer", command=self.calculate_probability_non_homogeneous
        )
        self.calculate_button_non_homogeneous.grid(row=0, column=0)

        self.graph_button_non_homogeneous = tk.Button(
            self.actions_frame_non_homogeneous, text="Afficher le graphe", command=self.plot_graph_non_homogeneous
        )
        self.graph_button_non_homogeneous.grid(row=0, column=1)

        # Cadre pour le résultat
        self.result_frame_non_homogeneous = tk.Frame(self.non_homogeneous_tab)
        self.result_frame_non_homogeneous.grid(row=5, column=0, columnspan=2, sticky="nsew")

        self.result_label_non_homogeneous = tk.Label(self.result_frame_non_homogeneous, text="")
        self.result_label_non_homogeneous.pack()

        # Cadre pour le graphique
        self.graph_frame_non_homogeneous = tk.Frame(self.non_homogeneous_tab)
        self.graph_frame_non_homogeneous.grid(row=0, column=2, rowspan=6, sticky="nsew")

        self.figure_non_homogeneous, self.ax_non_homogeneous = plt.subplots()
        self.canvas_non_homogeneous = FigureCanvasTkAgg(self.figure_non_homogeneous, master=self.graph_frame_non_homogeneous)
        self.canvas_non_homogeneous.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def calculate_probability_non_homogeneous(self):
        try:
            lambda_func_str = self.lambda_function_entry.get()
            # Create a lambda function with 't' explicitly in scope
            lambda_func = eval(f"lambda t: {lambda_func_str}")
            interval = self.time_interval_entry.get()
            a, b = map(float, interval.split(','))
            k = int(self.k_entry_non_homogeneous.get())

            # Calculate the integral of λ(t) over [a, b]
            t_values = np.linspace(a, b, 1000)
            lambda_values = np.array([lambda_func(t) for t in t_values])
            integral_lambda = np.trapz(lambda_values, t_values)

            # Calculate the probability
            probability = poisson.pmf(k, integral_lambda)
            self.result_label_non_homogeneous.config(
                text=f"P(k={k}; ∫λ(t)dt={integral_lambda:.2f}) = {probability:.4f}"
            )
        except Exception as e:
            self.result_label_non_homogeneous.config(text=f"Erreur : {str(e)}")

    def update_lambda_function(self, event=None):
        m = self.param_vars["m"].get()
        n = self.param_vars["n"].get()
        c = self.param_vars["c"].get()
        lambda_function = f"{m} + {n} * np.sin({c} * t)"
        self.lambda_function_entry.config(state="normal")
        self.lambda_function_entry.delete(0, tk.END)
        self.lambda_function_entry.insert(0, lambda_function)
        self.lambda_function_entry.config(state="readonly")

    def plot_graph_non_homogeneous(self):
        try:
            lambda_func_str = self.lambda_function_entry.get()
            # Create a lambda function with 't' explicitly in scope
            lambda_func = eval(f"lambda t: {lambda_func_str}")
            interval = self.time_interval_entry.get()
            a, b = map(float, interval.split(','))

            t_values = np.linspace(a, b, 1000)
            lambda_values = np.array([lambda_func(t) for t in t_values])

            self.ax_non_homogeneous.clear()
            self.ax_non_homogeneous.plot(t_values, lambda_values, label="λ(t)", color="blue")
            self.ax_non_homogeneous.set_title("Fonction d'intensité λ(t)")
            self.ax_non_homogeneous.set_xlabel("Temps (t)")
            self.ax_non_homogeneous.set_ylabel("Intensité λ(t)")
            self.ax_non_homogeneous.grid(True)
            self.ax_non_homogeneous.legend()

            self.canvas_non_homogeneous.draw()
        except Exception as e:
            self.result_label_non_homogeneous.config(text=f"Erreur : {str(e)}")

    def init_markov(self):
       
        # Cadre de description
        self.description_frame = tk.Frame(self.markov_tab)
        self.description_frame.pack(pady=10)

        self.description_label = tk.Label(
            self.description_frame,
            text=(
                "Un processus de Markov modélise des transitions entre états en fonction d'une matrice de probabilités.\n"
                "Exemple : Pij représente la probabilité de passer de l'état i à l'état j."
            ),
            justify=tk.LEFT,
            wraplength=600,
            font=("Time New Roman", 12)
        )
        self.description_label.pack()

        # Cadre pour les paramètres
        self.parameters_frame = tk.Frame(self.markov_tab)
        self.parameters_frame.pack(pady=10)

        self.num_states_label = tk.Label(self.parameters_frame, text="Nombre d'états:")
        self.num_states_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.num_states_entry = tk.Entry(self.parameters_frame)
        self.num_states_entry.grid(row=0, column=1, padx=5, pady=5)

        self.steps_label = tk.Label(self.parameters_frame, text="Nombre d'étapes:")
        self.steps_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")

        self.steps_entry = tk.Entry(self.parameters_frame)
        self.steps_entry.grid(row=1, column=1, padx=5, pady=5)

        self.simulate_button = tk.Button(self.parameters_frame, text="Simuler", command=self.simulate_markov_chain)
        self.simulate_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Cadre pour le graphique
        self.graph_frame = tk.Frame(self.markov_tab)
        self.graph_frame.pack(expand=True, fill="both")

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def simulate_markov_chain(self):
        try:
            num_states = int(self.num_states_entry.get())
            steps = int(self.steps_entry.get())

            # Générer une matrice de transition aléatoire
            transition_matrix = np.random.dirichlet(np.ones(num_states), size=num_states)

            # État initial
            current_state = 0
            states = [current_state]

            # Simuler les transitions
            for _ in range(steps):
                current_state = np.random.choice(num_states, p=transition_matrix[current_state])
                states.append(current_state)

            # Afficher le graphique
            self.ax.clear()
            self.ax.plot(range(len(states)), states, marker='o', linestyle='-', label="Trajectoire")
            self.ax.set_title("Simulation d'une chaîne de Markov")
            self.ax.set_xlabel("Étapes")
            self.ax.set_ylabel("État")
            self.ax.grid(True)
            self.ax.legend()

            self.canvas.draw()

        except ValueError:
            tk.messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")




if __name__ == "__main__":
    root = tk.Tk()
    app = PoissonSimulator(root)
    root.mainloop()
