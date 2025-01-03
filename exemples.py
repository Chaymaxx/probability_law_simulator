import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

class ExerciceInterface(tk.Tk):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Exercices Corrigés")
        self.geometry("1000x800")

        # Définition des couleurs
        self.background_color = "#f0f0f0"
        self.foreground_color = "#333333"

        # Définition de la police
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=13, family="Helvetica")

        # Création d'un style personnalisé pour les widgets ttk
        self.style = ttk.Style(self)
        self.style.theme_use("clam")  # Utilisation d'un thème ttk par défaut
        self.style.configure("Custom.TFrame", background=self.background_color)

        # Exercices
        self.exercices = [
            {
                "title": "Loi Binomiale - Exercice",
                "annonce": """
                    Un fabricant de jouets affirme que la probabilité qu'un robot jouet fonctionne correctement est de 0.9. Si 10 de 
                    ces robots sont choisis au hasard,quelle est la probabilité exacte que exactement 8 d'entre eux fonctionnent
                    correctement ?
                """,
                "solution": """
                     La probabilité exacte que 8 sur 10 robots fonctionnent correctement est donnée par la distribution binomiale,
                     avec les paramètres suivants :
                    n = 10 (nombre total de robots)
                    k = 8 (nombre de robots qui fonctionnent correctement)
                    p = 0.9 (probabilité qu'un robot fonctionne correctement)

                    La formule de la distribution binomiale pour la probabilité d'exactement k succès dans n  essais est :
                    P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
                    En substituant les valeurs données :
                    P(X = 8) = C(10, 8) * 0.9^8 * (1-0.9)^(10-8)
                    P(X = 8) = C(10, 8) * 0.9^8 * 0.1²
                    
                    En utilisant la formule du coefficient binomial (n! / (k! * (n-k)!)), 
                    nous obtenons : P(X = 8) = (10! / (8! * 2!)) * 0.9^8 * 0.1²
                                    P(X = 8) = (10 * 9 / 2) * 0.9^8 * 0.1²
                                    P(X = 8) = 45 * 0.9^8 * 0.1²
                """
            },
            {
                "title": "Loi de Poisson - Exercice",
                "annonce": """
                    La moyenne des accidents de la route sur une autoroute est de 2 par jour.Quelle est la probabilité 
                    qu'il y ait exactement 3 accidents de la route en un jour donné ?
                """,
                "solution": """
                    Pour résoudre ce problème, nous pouvons utiliser la distribution de Poisson, qui est souvent utilisée pour 
                    modéliser le nombre d'événements rares dans un intervalle de  temps donné, lorsque la moyenne des événements
                    est connue. La formule de la distribution de Poisson pour calculer la probabilité d'observer exactement k 
                    événements, lorsque la moyenne est λ, est donnée par :
                                                  P(X = k) = (e^(-λ) * λ^k) / k! 
                    Dans ce cas, la moyenne (λ) des accidents de la route est de 2 par jour et nous voulons trouver la probabilité
                    d'avoir exactement 3 accidents (k=3) en un jour donné. Nous pouvons utiliser λ=2 et k=3 dans la formule de la 
                    distribution de Poisson.
                     Calculons cela :
                             P(X = 3) = (e^(-2) * 2^3) / 3!
                """
            },
            {
                "title": "Loi Normale - Exercice",
                "annonce": """
                    Les scores d'un test standardisé sont distribués de manière normale avec une moyenne  de 100 et un écart-type
                    de 15. Quelle est la probabilité qu'un étudiant choisi au hasard ait un score supérieur à 120 ?
                """,
                "solution": """
                     Pour calculer la probabilité qu'un étudiant choisi au hasard ait un score supérieur à 120 dans un test standardisé
                     distribué de manière normale, nous devons utiliser la fonction de répartition cumulative (CDF) de la distribution 
                     normale.
                    La formule de la CDF de la distribution normale est donnée par :
                                             P(X > x) = 1 - Φ((x - μ) / σ)
                    Où :
                                -x est la valeur seuil (score dans ce cas),
                                -μ est la moyenne de la distribution,
                                -σ est l'écart-type de la distribution,
                                -Φ est la fonction de distribution cumulative normale.
                    Dans ce cas, la moyenne μ est de 100 et l'écart-type σ est de 15. Nous voulons trouver la  probabilité que X > 120
                    Nous utiliserons ces valeurs pour calculer la probabilité.
                    Calculons cela :
                                 P(X > 120) = 1 - Φ(20 / 15)
                    nous devons chercher la valeur de Φ(20 / 15), qui est la probabilité que la variable aléatoire soit inférieure à
                    20 / 15.
                    En utilisant une table de la distribution normale standard ou une fonction de calcul, nous trouvons 
                                Φ(20 / 15) ≈ Φ(1.33) ≈ 0.9082.
                    Par conséquent :
                                P(X > 120) = 1 - 0.9082
                                P(X > 120) ≈ 0.0918
                    La probabilité qu'un étudiant choisi au hasard ait un score supérieur à 120 est d'environ 0.0918, soit environ 9.18%.
                """
            },
            {
                "title": "Loi de Bernoulli - Exercice",
                "annonce": """
                    Un joueur lance un dé équilibré à 6 faces. Quelle est la probabilité d'obtenir un 6 en un seul lancer ?
                """,
                "solution": """
                 Nous considérons un seul essai de succès/échec.Dans ce cas, le succès est d'obtenir un 6, et il y a un seul succès 
                 possible (obtenir un 6) parmi les six résultatspossibles (obtenir un nombre entre 1 et 6).
                 La probabilité d'obtenir un succès (un 6) est donc p=1/6, et la probabilité d'échec est q=1-p=1-1/6=5/6.
                 La formule de la loi de Bernoulli est : P(X=k) = p^k * q^(1-k)
                 Où :
                                P(X=k) est la probabilité d'obtenir k succès,
                                p est la probabilité de succès,
                                q est la probabilité d'échec,
                                k est le nombre de succès.
                    
                  Dans notre cas, nous voulons trouver la probabilité d'obtenir un 6 en un seul lancer, donc k=1. En substituant les
                  valeurs, nous avons :
                                                P(X=1) = (1/6)^1 * (5/6)^(1-1)
                                                P(X=1) = (1/6) * 1
                                                P(X=1) = 1/6
                  Donc, la probabilité d'obtenir un 6 en un seul lancer est 1/6.
                """
            },
            {
                "title": "Inégalité de Markov - Exercice",
                "annonce": """
                    Une variable aléatoire X suit une distribution avec une moyenne de 50 et une variance  de 100.
                    Appliquez l'inégalité de Markov pour estimer la probabilité P(X ≥ 75).
                """,
                "solution": """
                    L'inégalité de Markov stipule que pour toute variable aléatoire X ≥ 0 et tout a > 0,la probabilité P(X ≥ a)
                    est bornée par la moyenne de X divisée par a. Mathématiquement, cela peut être exprimé  comme suit :
                                 P(X ≥ a) ≤ E(X) / a
                    Où :
                               - P(X ≥ a) est la probabilité que X soit supérieur ou égal à a,
                               - E(X) est la moyenne de X,
                               - a est un nombre réel positif.
                    
                  Dans notre cas, E(X) = 50 et nous voulons estimer la probabilité P(X ≥ 75).En utilisant l'inégalité de Markov, 
                  nous obtenons :         P(X ≥ 75) ≤ 50 / 75
                                          P(X ≥ 75) ≤ 2/3
                  Donc, selon l'inégalité de Markov, la probabilité P(X ≥ 75) est bornée par 2/3.
                """
            },
            {
                "title": "Inégalité de Bienaymé-Tchebychev - Exercice",
                "annonce": """
                    Les scores d'un examen ont une moyenne de 70 et un écart-type de 5. Utilisez l'inégalitéde Bienaymé-Tchebychev 
                    pour estimer la probabilité que le score soit compris entre 60 et 80.
                """,
                "solution": """
                    L'inégalité de Bienaymé-Tchebychev peut être exprimée comme suit :
                                        P(|X - μ| ≥ kσ) ≤ 1/k^2
                    
                    Où :
                    P(|X - μ| ≥ kσ) est la probabilité que X s'écarte de sa moyenne de plus de k fois l'écart-type, μ est la moyenne
                    de X, σ est l'écart-type de X, k est un nombre réel positif.
                    Dans ce cas, nous voulons estimer la probabilité que le score soit compris entre 60 et 80,c'est-à-dire que
                                                    |X - 70| ≤ 10.       
                    Étant donné que l'écart-type est de 5, nous prenons k = 2.
                    En utilisant l'inégalité de Bienaymé-Tchebychev, nous avons :
                                                            P(|X - 70| ≤ 10) = P(-10 ≤ X - 70 ≤ 10)
                                                                            = P(-2σ ≤ X - μ ≤ 2σ) ≤ 1/2^2 = 1/4
                    Donc, la probabilité que le score soit compris entre 60 et 80 est bornée par 1/4, soit 25%.
                """
            }
            # Ajoutez d'autres exercices ici avec le même format
        ]

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        for exercice in self.exercices:
            tab = ttk.Frame(self.notebook, style="Custom.TFrame")
            self.notebook.add(tab, text=exercice["title"])

            annonce_label = tk.Label(tab, text=exercice["annonce"], padx=20, pady=20, wraplength=950, justify="left",
                                      bg=self.background_color, fg=self.foreground_color, font=("Helvetica", 13))
            annonce_label.pack(anchor="w")

            bouton_solution = tk.Button(tab, text=" Solution ",
                                         command=lambda tab=tab, content=exercice["solution"]: self.toggle_solution(tab, content),
                                         bg="DARKVIOLET", fg="white")
            bouton_solution.pack(pady=0, anchor="center")
            

            # Label pour afficher la solution
            solution_label = tk.Label(tab, text=exercice["solution"], padx=20, pady=20, wraplength=950, justify="left",
                                      bg=self.background_color, fg=self.foreground_color, font=("Helvetica", 13))

            # Sauvegarder le label dans un attribut pour y accéder plus tard
            tab.solution_label = solution_label

    def toggle_solution(self, tab, content):
        # Si le label est actuellement affiché, masquez-le. Sinon, affichez-le.
        if tab.solution_label.winfo_ismapped():
            tab.solution_label.pack_forget()
        else:
            tab.solution_label.pack(anchor="w")

if __name__ == "__main__":
    app = ExerciceInterface()
    app.mainloop()