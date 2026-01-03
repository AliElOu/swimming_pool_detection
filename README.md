# Swimming Pool Detection CLI

Détecte automatiquement les piscines dans des images aériennes en utilisant YOLO pour la détection et SAM pour la segmentation précise.

## Installation

### 1. Installer Python 3.11.13

Télécharger et installer Python 3.11.13 depuis [python.org](https://www.python.org/downloads/release/python-31113/)

### 2. Installer les dépendances avec versions exactes

```bash
# Installer les dépendances principales
pip install ultralytics==8.3.171 opencv-python==4.10.0.84 pillow==10.4.0 numpy==1.26.4

# Dépendances supplémentaires (installées automatiquement)
# torch==2.5.1+cu121
# torchvision==0.20.1+cu121
# PyYAML==6.0.2
# scipy==1.16.1
# matplotlib==3.10.5
# pandas==2.2.2
# seaborn==0.13.2
```

## Fichiers requis

- `detect_pools.py` : script principal
- `last.pt` : modèle YOLO entraîné (détection) - doit être dans le même dossier
- `sam_b.pt` : modèle SAM (segmentation) - doit être dans le même dossier

## Usage

### Commande basique

```bash
python detect_pools.py --image aerial_image.jpg
```

### Spécifier un dossier de sortie

```bash
python detect_pools.py --image aerial_image.jpg --output-dir results/
```

## Sorties

Le script génère 2 fichiers :

### 1. `coordinates.txt`
Coordonnées des contours des piscines détectées (format x y par ligne) :
```
# Pool 1 (45 points)
123 456
125 458
...

# Pool 2 (38 points)
300 200
302 201
...
```

### 2. `output_image.jpg`
Image originale avec contours rouges autour des piscines détectées.

## Options disponibles

| Argument | Court | Description | Défaut |
|----------|-------|-------------|--------|
| `--image` | `-i` | Chemin de l'image d'entrée (requis) | - |
| `--output-dir` | `-o` | Dossier de sortie | `.` (dossier courant) |

## Configuration

Le script utilise les paramètres suivants (codés en dur dans le script) :
- **Modèle YOLO** : `last.pt` (doit être présent dans le dossier)
- **Modèle SAM** : `sam_b.pt` (doit être présent dans le dossier)
- **Seuil de confiance** : 0.5
- **Post-traitement morphologique** : Activé
- **Épaisseur du contour** : 2 pixels

Pour modifier ces paramètres, éditez directement le fichier `detect_pools.py` (section "Configuration" dans la fonction `main()`).

### Batch processing (PowerShell)
```powershell
# Traiter toutes les images PNG d'un dossier
foreach ($img in Get-ChildItem *.png) {
    python detect_pools.py --image $img.Name --output-dir "results/$($img.BaseName)/"
}
```

## Format de sortie `coordinates.txt`

```
# Pool 1 (N points)
x1 y1
x2 y2
...
xN yN

# Pool 2 (M points)
x1 y1
...
```

- Coordonnées en **pixels** (entiers)
- Un pool par section
- Contour **fermé** (le dernier point rejoint automatiquement le premier lors du dessin)

## Troubleshooting

### Aucune piscine détectée
- Vérifier que le modèle YOLO est bien entraîné sur des images aériennes
- Si besoin, modifier le seuil de confiance dans le script (variable `CONFIDENCE_THRESHOLD`)

### Contours imprécis / trous
- Le post-traitement morphologique est activé par défaut et devrait corriger ces problèmes
- Si nécessaire, ajuster les paramètres dans le script

### Erreur "model not found"
- Vérifier que `last.pt` et `sam_b.pt` sont présents dans le même dossier que `detect_pools.py`

## Pipeline technique

1. **Chargement** : YOLO + SAM
2. **Détection YOLO** : extraction des bounding boxes (pools)
3. **Segmentation SAM** : masque précis pour chaque bbox
4. **Post-traitement morphologique** :
   - Fermeture (combler gaps)
   - Remplissage des trous
   - Ouverture (nettoyer artefacts)
5. **Extraction contour** : findContours + simplification Douglas-Peucker
6. **Export** : coordinates.txt + output_image.jpg

## Performance

- **Images 512×512** : ~2-5 secondes par image (GPU)
- **Images 2048×2048** : ~10-20 secondes par image (GPU)
- CPU : 5-10× plus lent
