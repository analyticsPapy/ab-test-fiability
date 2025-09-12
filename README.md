📊 Calculateur de fiabilité d’un A/B test

Application Streamlit permettant d’évaluer la significativité et la puissance d’un test A/B, que ce soit sur :
	•	des taux de conversion (binomial)
	•	des métriques continues (ex : panier moyen, durée, …)

✨ Fonctionnalités
	•	Calcul de la p-valeur et décision de significativité
	•	Affichage des intervalles de confiance
	•	Calcul du lift relatif
	•	Estimation de la puissance post hoc (≈)
	•	Estimation de la taille d’échantillon a priori (pour un MDE donné)
	•	Export CSV des résultats
	•	Interface lisible pour profils non-statisticiens (badges, KPIs, explications)

🚀 Installation

1. Cloner le projet

git clone https://github.com/<ton-user>/<ton-repo>.git
cd <ton-repo>

2. Créer un environnement virtuel (recommandé)

Avec Python 3.12 (⚠️ Streamlit pas encore compatible 3.13) :

python3.12 -m venv .venv
source .venv/bin/activate

3. Installer les dépendances

pip install -r requirements.txt

Fichier requirements.txt minimal :

streamlit
numpy
scipy
pandas

▶️ Utilisation

Lancer l’application :

streamlit run app.py

Puis ouvrir dans votre navigateur à l’adresse http://localhost:8501.

🖼️ Interface
	•	Sidebar : paramètres globaux (α, hypothèse alternative, type de métrique)
	•	Section données : saisie des effectifs ou statistiques
	•	Résultats : KPIs, graphiques, interprétation
	•	Export : bouton pour télécharger un résumé CSV

📝 Notes méthodologiques
	•	Binomiale : test Z de différence de proportions (Wald)
	•	Continue : test t de Welch
	•	Puissance post hoc : approximation normale sous l’effet observé (indicatif)
	•	Taille d’échantillon : formules classiques (approx normale)
	•	Attention aux biais : durée d’exposition, randomisation, saisonnalité, multiples comparaisons…

