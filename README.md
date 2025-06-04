CNN for triassic cephalopods fetched from publications

Installation:

Originally trained on nvidia tensorflow docker container version 25.02 with rtx 5070 12 Go

pip install -r requirements.txt

Training:

 usage: training.py [-h] [-sz SIZE_IMG] [-b BATCH] [-cm COLOR_MODE] [-d DROPOUT] [-bl BASE_LEARNING_RATE] [-ie INITIAL_EPOCHS] [-fe FINE_EPOCHS] [-fl FREEZE_LAYER] [-wa WEIGHT_ARCHI] [-vs VALID_SPLIT]
                        [-ts TEST_SPLIT] [-dp DATA_PATH] [-sp SAVE_PATH] [-sn SAVE_FILENAME] [-sh]

options:
  -h, --help            show this help message and exit
  -sz SIZE_IMG, --size_img SIZE_IMG
                        Tailles des images passées au modèle
  -b BATCH, --batch BATCH
                        Taille du batch (min:10)
  -cm COLOR_MODE, --color_mode COLOR_MODE
                        Définie le mode de représentation des images (rgb/grayscale)
  -d DROPOUT, --dropout DROPOUT
                        Valeur du Dropout (entre 0 et 1)
  -bl BASE_LEARNING_RATE, --base_learning_rate BASE_LEARNING_RATE
                        Taux apprentissage de base
  -ie INITIAL_EPOCHS, --initial_epochs INITIAL_EPOCHS
                        Nombres inital itérations
  -fe FINE_EPOCHS, --fine_epochs FINE_EPOCHS
                        Nombres finition itérations
  -fl FREEZE_LAYER, --freeze_layer FREEZE_LAYER
                        Nombre de couches à geler
  -wa WEIGHT_ARCHI, --weight_archi WEIGHT_ARCHI
                        Poid à utiliser avec architecture (imagenet, None ou un chemin de fichier)
  -vs VALID_SPLIT, --valid_split VALID_SPLIT
                        Pourcentage de la base de données entrainement à utiliser pour la validation (entre 0 et 1)
  -ts TEST_SPLIT, --test_split TEST_SPLIT
                        Pourcentage de la base de données validation à utiliser pour la base de données test (entre 0 et 100), pointe vers un dossier test dans --data_path si égal à 0
  -dp DATA_PATH, --data_path DATA_PATH
                        Chemin pour la base de donnée (le chemin doit contenir /train/fichier_séparé_dans_des_dossiers_de_chaque_classes)
  -sp SAVE_PATH, --save_path SAVE_PATH
                        Chemin pour sauvegarder
  -sn SAVE_FILENAME, --save_filename SAVE_FILENAME
                        Nom du fichier de sauvegarde
  -sh, --show_plot      Montre les figures à la fin

Prediction:

predict.py can output Matrix confusion, ROC-AUC (OvR), t-SNE, LIME, Shap and basic stats

List of used publications for datasets in publis_list folder

Datasets available at : [https://www.kaggle.com/datasets/pyraro/triassic-cephalopods](https://kaggle.com/datasets/d4654299a78b2e69d97e4bede802604fa1e098cee3d948e4a807a2e4a87d564e)
