# FAQ

**Q: Le nom du projet parle de DQN, mais l'appli que je lance a l'air d'utiliser des règles fixes. Pourquoi ?**
Parce que c'est effectivement le cas. Le DQN est entraîné dans `notebooks/3_dqn_training.ipynb` et sauvegardé dans `models/dqn_schedule_model.h5`, mais l'application Streamlit livrée (`ui/app.py`) ne le charge pas — elle utilise un optimiseur à base de règles (`SimpleScheduleOptimizer`) qui imite le comportement attendu. Voir [Architecture](Architecture.md) pour le détail.

**Q: Où sont les données ATUS ?**
Elles ne sont pas incluses dans le dépôt (taille et conditions de licence de l'ATUS). Le dossier `Data/` est listé dans `.gitignore`. Pour reproduire les notebooks, télécharge les données depuis [bls.gov/tus](https://www.bls.gov/tus/) et place-les dans `Data/raw/`.

**Q: `pip install -r requirements.txt` ne suffit pas pour lancer les notebooks, pourquoi ?**
`requirements.txt` est volontairement limité aux dépendances de l'app Streamlit déployée (léger, sans TensorFlow/Gym). Pour les notebooks, installe en plus `tensorflow-cpu`, `gym`, `scikit-learn`, `seaborn`, `tqdm` — voir [Getting-Started](Getting-Started.md).

**Q: Pourquoi le titre du README ressemble à celui d'un autre dépôt (`Bosaj/Assistant-Personnel-pour-la-Gestion-du-Temps`) ?**
Les deux dépôts viennent d'un même projet académique à deux personnes sur le même sujet (planification via DQN entraîné sur ATUS) ; le pipeline de notebooks a une origine commune. Chaque personne a ensuite construit sa propre interface indépendamment. Ce ne sont pas des forks l'un de l'autre (aucun historique Git partagé), simplement deux implémentations parallèles d'un même sujet.

**Q: La CI GitHub Actions est-elle verte ?**
Le workflow `python-package.yml` s'exécute (lint + `pytest`), mais comme il n'y a pas encore de fichiers de tests dans le dépôt, l'étape `pytest` ne trouve aucun test à lancer. C'est documenté comme limitation connue plutôt que corrigé silencieusement.

**Q: Le `Jenkinsfile` est-il utilisé ?**
Non, pas dans l'état actuel : il référence un serveur Jenkins et des identifiants Docker Hub externes qui ne sont pas configurés pour ce dépôt. Il est conservé comme référence/template.
