# Machine_Learning_Project_SALAS_KOUSSAIER
Application permettant de différencier des lettres manuscrites.
Pour la première version: seulement 3 lettres ('a', 'c', et 'j')


## Process et avancement avec étapes : 
### 1 - Collection et préparation des données :

* **Ensemble de données :**
  Un ensemble de données de lettres manuscrites « a », « j » et « c ». On a créer notre propre ensemble de données en dessinant les lettres 'a', 'j' et 'c' en utilisant "paint". Total de données : 45 images par lettre. 

* **Prétraitement :** Pour s'assurer que les images sont dans un format uniforme. On a : 
    * Converti les images en grayscale.
    * Normaliser les valeurs des pixels.
    * Aplatit l'image en un array 1D.

### 2 - Développement du modèle: 

* **PMC :**
 - *Propriétés :*
    ```
    learning rate = 0.005
    input size = 400x400 (vecteur d'entrée = nombre de pixels)
    hidden size = 128 (nombre de neurones par couche)
    output size = 3 (a, c et j)
    epochs = 50
    hidden layers = 1
    weights, biases = random
    ```

### 3 - Entraînement du modèle :

* **PMC :**
  - On a :
    * diviser notre ensemble de données en ensembles d'apprentissage et de test.
    * Entraîner notre modèle sur l’ensemble d’entraînement : 
      * *Forward propagation :*
        - En utilisant *sigmoid* pour la couche cachée (hidden layer)
        - En utilisant *softmax* pour la couche de sortie (output layer)
      * *Back propagation (train) :* 
        Calculer les erreurs et mettre à jour les 'weights' de chaque couche.
    
    * Calculer la 'loss' et 'accuracy' de chaque epoch en utilisant cross-entropy pour la loss.

### 4 - Évaluation du modèle :

* **PMC :**
  - Après la formation, on a évaluer la performance de notre modèle sur l'ensemble de test pour voir dans quelle mesure il se généralise à de nouvelles données invisibles.
  On a utilisé **"accuracy"** comme indicateur de performance.

### 5 - Itération et amélioration :

* **PMC :**
  - On a créer d'avantage des données.
  - On a ajuster le processus de formation (diminuer le learning rate, changer la loss function et la fonction d'activation pour la couche de sorite)

### 6 - Outils de mise en œuvre :

  On a utiliser **C++** comme langage de programmation et la lib **OpenCV** pour charger et manipuler les images.

## Résultats obtenus et commentaires : 

* **PMC:**
  * Au début, dans la première iteration, on a eu une accuracy de **100%**, ce qui nous a surpris. On a constaté que c'etait parce que les données de test faisait partie de celle de l'entrainement.
  * Dans une deuxieme iteration, on a eu une accuracy de **46%**, on a donc ajouter plus de données et changer la fonction de loss et activation pour la couche de sortie.
  * Dans une troisième version, on a eu une accuracy de **66%**, on a donc diminuer le "learning rate".
    → Resultat : **Accuracy = 80%**
  
  *Ce résultat n'est pas final, on a encore des modifications et ajustement à ajouter dans les prochaines version*

  


