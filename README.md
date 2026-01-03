# Swimming Pool Detection CLI

Détecte automatiquement les piscines dans des images aériennes en utilisant YOLO pour la détection et SAM pour la segmentation précise.

## Prérequis

- Python 3.11.13
- Git
- Git LFS (Large File Storage)

## Installation complète

### 1. Installer Git LFS

Git LFS est nécessaire pour télécharger les fichiers de modèles volumineux (`.pt`).

**Windows :**
```bash
# Télécharger depuis https://git-lfs.github.com/
# Ou avec Chocolatey :
choco install git-lfs

# Initialiser Git LFS
git lfs install
```

**Linux/macOS :**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Initialiser Git LFS
git lfs install
```

### 2. Cloner le projet

```bash
git clone <URL_DU_REPO>
cd swimming_pool_detection

# Vérifier que les fichiers LFS ont été téléchargés
git lfs ls-files
# Devrait afficher : sam2.1_l.pt
```

### 3. Installer Python 3.11.13

Télécharger et installer Python 3.11.13 depuis [python.org](https://www.python.org/downloads/release/python-31113/)

### 4. Installer les dépendances avec versions exactes

```bash
pip install -r requirements.txt
```

## Fichiers requis

Les fichiers suivants doivent être présents dans le dossier du projet :

- `detect_pools.py` : script principal
- `last.pt` : modèle YOLO entraîné (détection)
- `sam2.1_l.pt` : modèle SAM (segmentation) - **téléchargé automatiquement via Git LFS**

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
- **Modèle SAM** : `sam2.1_l.pt` (téléchargé via Git LFS)
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
