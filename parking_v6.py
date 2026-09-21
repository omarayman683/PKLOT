"""parking_v6 : PKLot - occupation de places de parking.

Contient tout le pipeline de parking_v5 (RGB -> LBP global, 7 distances,
validation croisee, chronometrage) auquel s'ajoute la caracterisation
LBP multi-echelle / par blocs (FEATURE_MODE = "multiscale", par defaut).
Passer FEATURE_MODE a "mosaic" (ou "global") retrouve exactement le
comportement de la v5.

Extension a la couleur : trois strategies sont comparees sur plusieurs tirages
(gris de reference, mosaique R|G|B, un LBP par plan) - voir
compare_color_strategies(). Lancement : python parking_v6.py [--color-only]
"""
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1) PARAMETRES GENERAUX
# ==============================================================================

# Dataset deja separe en train/ et test/, chacun avec un dossier "empty" et
# un dossier "occupied" (ou variantes : free/busy, vacant/taken...)
DATA_DIR = r"D:\uha\2d 3d\projet\projet\database"
SUB_FOLDERS = ["A", "B"]  # database/A/... et database/B/... sont combines

# Choix de la caracterisation :
#   "multiscale" : LBP uniforme par blocs, multi-echelle (v6, 1239 valeurs)
#   "mosaic"     : RGB -> R|G|B cote a cote -> un seul LBP (v5, 256 valeurs)
#                  ("global" est un alias de "mosaic")
#   "gray"       : niveaux de gris -> LBP global (reference, 256 valeurs)
#   "per_plane"  : un LBP par plan R, G, B, histogrammes concatenes (768 valeurs)
FEATURE_MODE = "multiscale"

# Extension couleur : strategies comparees (voir compare_color_strategies)
COLOR_STRATEGIES = ["gray", "mosaic", "per_plane"]
RUN_COLOR_COMPARISON = True  # lance apres la validation croisee

# Pyramide de grilles du mode "multiscale" : 1x1 = image entiere (vue globale),
# 2x2 et 4x4 = blocs de plus en plus fins (vue locale).
GRID_SCALES = [1, 2, 4]

# Resultats ranges par etape du pipeline :
#   resultats_v6/
#     1_modele/              model.txt, test.txt
#     2_caracterisation/     multi_echelle_exemple.png, histogramme_moyen.png
#     3_distances/           comparaison_distances.txt / .png
#     4_pyramide_tirages/    pyramide_tirages.csv / .png (accuracy par tirage
#                            + matrice de confusion), validation_croisee.png
#     5_couleur/             comparaison_couleur.txt / .png, mosaique_couleur.png
#     chronometre.txt        duree de chaque etape
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats_v6")
MODEL_DIR = os.path.join(RESULT_DIR, "1_modele")
FEATURE_DIR = os.path.join(RESULT_DIR, "2_caracterisation")
DISTANCE_DIR = os.path.join(RESULT_DIR, "3_distances")
PYRAMID_DIR = os.path.join(RESULT_DIR, "4_pyramide_tirages")
COLOR_DIR = os.path.join(RESULT_DIR, "5_couleur")
MODEL_FILE = os.path.join(MODEL_DIR, "model.txt")
TEST_RESULT_FILE = os.path.join(MODEL_DIR, "test.txt")
TIMING_FILE = os.path.join(RESULT_DIR, "chronometre.txt")

N_TRAIN_PER_CLASS = 3682   # limite par la classe la plus petite ("empty" = 4182 images)
N_TEST_PER_CLASS = 500     # -> chaque classe garde vraiment 500 images de test

LABELS = {"empty": 0, "occupied": 1}
EMPTY_NAMES = {"empty", "free", "vacant"}
OCCUPIED_NAMES = {"occupied", "busy", "taken"}

RANDOM_SEED = 42
# Le 1er tirage (RANDOM_SEED) est le tirage "officiel" : c'est le seul qui
# ecrit model.txt / test.txt / les graphes. Les autres ne servent qu'a
# mesurer la stabilite de l'accuracy (validation croisee).
# 5 tirages au maximum (temps de calcul) : la comparaison couleur reutilise
# les memes 5 tirages que la validation croisee.
CV_SEEDS = [RANDOM_SEED, 1, 2, 3, 4]
COLOR_SEEDS = CV_SEEDS

RESIZE_SIZE = (64, 64)
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Stocke la duree de chaque etape du run officiel, pour le rapport final
TIMINGS = {}

for _d in (MODEL_DIR, FEATURE_DIR, DISTANCE_DIR, PYRAMID_DIR, COLOR_DIR):
    os.makedirs(_d, exist_ok=True)


@contextmanager
def timer(label):
    """Chronometre un bloc de code et affiche/enregistre sa duree."""
    t0 = time.perf_counter()
    print(f"[timer] {label}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        TIMINGS[label] = TIMINGS.get(label, 0.0) + elapsed
        print(f"[timer] {label} termine en {elapsed:.2f} s")


# ==============================================================================
# 2) EXTRACTION DU VECTEUR DE CARACTERISTIQUES
# ==============================================================================

def compute_lbp(gray_img):
    """LBP de base (8 voisins, rayon 1, sans interpolation) : pour chaque
    pixel, on compare sa valeur a ses 8 voisins (>= => bit 1, sinon bit 0),
    ce qui donne un code de 0 a 255 par pixel. Vectorise en numpy (pas de
    boucle Python pixel par pixel)."""
    img = gray_img.astype(np.int16)
    padded = np.pad(img, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    # ordre des 8 voisins (sens horaire, en partant du haut-gauche)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    code = np.zeros_like(center, dtype=np.uint8)
    h, w = center.shape
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    return code


def extract_feature_vector_global(image_path):
    """Caracterisation v5 (couleur + LBP global) :
      1) image chargee en RGB (3 canaux, 8 bits -> equivalent CV_8UC3)
      2) separation en 3 images en niveaux de gris : R, G, B
      3) concatenation horizontale des 3 -> une "grosse image" (meme hauteur,
         3x la largeur)
      4) LBP calcule sur cette grosse image
      5) vecteur de caracteristiques = histogramme des codes LBP (256 valeurs),
         garde la meme taille que l'ancien histogramme de gris -> compatible
         avec tout le reste du pipeline (model.txt, SAD, les 6 autres distances)
    """
    img_rgb = np.array(Image.open(image_path).convert("RGB").resize(RESIZE_SIZE))
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    big_image = np.hstack([r, g, b])  # (H, 3*W) : R | G | B cote a cote

    lbp_codes = compute_lbp(big_image)
    histogram = np.bincount(lbp_codes.ravel(), minlength=256)[:256]
    return histogram.tolist()


# ------------------------------------------------------------------------
# 2bis) LBP MULTI-ECHELLE / PAR BLOCS (v6)
# ------------------------------------------------------------------------
# LBP "uniform" : un code est uniforme s'il a au plus 2 transitions 0<->1 (en
# lisant les 8 bits en cercle). Il y en a 58 ; tous les autres codes partagent
# un 59e bin. Avec 21 blocs (1 + 4 + 16), 59 bins garde le vecteur a 1239
# valeurs au lieu de 5376 : model.txt reste de taille raisonnable.

N_BINS = 59


def _build_uniform_lut():
    lut = np.zeros(256, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            lut[code] = next_bin
            next_bin += 1
        else:
            lut[code] = 58  # bin "non uniforme"
    assert next_bin == 58
    return lut


UNIFORM_LUT = _build_uniform_lut()


def load_gray(image_path):
    return np.array(Image.open(image_path).convert("L").resize(RESIZE_SIZE))


def block_histograms(lbp_codes, grid):
    """Decoupe l'image de codes en grid x grid blocs et renvoie la liste des
    histogrammes (N_BINS valeurs chacun), bloc par bloc, ligne par ligne."""
    h, w = lbp_codes.shape
    hists = []
    for by in range(grid):
        for bx in range(grid):
            block = lbp_codes[by * h // grid:(by + 1) * h // grid,
                              bx * w // grid:(bx + 1) * w // grid]
            hists.append(np.bincount(block.ravel(), minlength=N_BINS))
    return hists


def extract_feature_vector_multiscale(image_path):
    """1) image en niveaux de gris, redimensionnee
       2) LBP (8 voisins, rayon 1) -> mapping uniforme (59 bins)
       3) pour chaque echelle de GRID_SCALES : histogramme par bloc
       4) concatenation de tous les histogrammes -> vecteur de
          N_BINS * (1 + 4 + 16) = 1239 entiers
    Chaque echelle a la meme masse totale (nb de pixels de l'image), donc
    aucune echelle ne domine la distance SAD."""
    codes = UNIFORM_LUT[compute_lbp(load_gray(image_path))]
    parts = []
    for grid in GRID_SCALES:
        parts.extend(block_histograms(codes, grid))
    return np.concatenate(parts).astype(np.int32).tolist()


# ------------------------------------------------------------------------
# 2ter) EXTENSION A LA COULEUR : 3 strategies (LBP defini sur UN canal)
# ------------------------------------------------------------------------
# Le LBP est defini sur un canal unique. Pour une image couleur (CV_8UC3) :
#   - gray      : on convertit en niveaux de gris (reference)
#   - mosaic    : R, G, B juxtaposes en une image de largeur 3W, un seul LBP
#                 (= extract_feature_vector_global, la methode de la v5)
#   - per_plane : un LBP par plan, les 3 histogrammes sont concatenes
# Le LBP compare chaque voisin au pixel central : il est invariant aux
# changements monotones d'intensite, donc les trois plans portent
# (a peu pres) la meme texture locale meme s'ils different en luminosite.

def extract_feature_vector_gray(image_path):
    """Reference : gris -> LBP -> histogramme de 256 codes."""
    lbp_codes = compute_lbp(load_gray(image_path))
    return np.bincount(lbp_codes.ravel(), minlength=256)[:256].tolist()


def extract_feature_vector_per_plane(image_path):
    """Un LBP par plan R, G, B, puis concatenation des 3 histogrammes
    (3 x 256 = 768 valeurs)."""
    img_rgb = np.array(Image.open(image_path).convert("RGB").resize(RESIZE_SIZE))
    hists = [np.bincount(compute_lbp(img_rgb[:, :, c]).ravel(), minlength=256)[:256]
             for c in range(3)]
    return np.concatenate(hists).tolist()


FEATURE_EXTRACTORS = {
    "multiscale": extract_feature_vector_multiscale,
    "mosaic":     extract_feature_vector_global,
    "global":     extract_feature_vector_global,  # alias de "mosaic"
    "gray":       extract_feature_vector_gray,
    "per_plane":  extract_feature_vector_per_plane,
}

# Les tirages piochent dans le meme pool d'images : on extrait chaque image
# une seule fois par mode (cle = (mode, chemin)).
_FEATURE_CACHE = {}


def extract_feature_vector(image_path):
    """Point d'entree unique utilise par tout le pipeline : dispatche selon
    FEATURE_MODE et met le resultat en cache."""
    key = (FEATURE_MODE, image_path)
    vec = _FEATURE_CACHE.get(key)
    if vec is None:
        if FEATURE_MODE not in FEATURE_EXTRACTORS:
            raise ValueError(f"FEATURE_MODE inconnu : {FEATURE_MODE!r}")
        vec = _FEATURE_CACHE[key] = FEATURE_EXTRACTORS[FEATURE_MODE](image_path)
    return vec


def extract_feature_vector_gray_hist(image_path):
    """Ancienne caracterisation (histogramme de gris, 256 valeurs) - gardee
    pour comparaison / retour en arriere si besoin."""
    img = Image.open(image_path).convert("L")
    img = img.resize(RESIZE_SIZE)
    return img.histogram()


# ==============================================================================
# 3) LISTER LES IMAGES DE dataset2/train ET dataset2/test
# ==============================================================================

def list_images(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dossier introuvable : {folder}")
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VALID_EXT)
    )


def find_class_folder(base_dir, aliases):
    """Cherche dans base_dir un sous-dossier dont le nom (insensible a la
    casse) correspond a l'un des alias (ex: 'empty', 'free', 'vacant')."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Dossier introuvable : {base_dir}")
    for entry in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full) and entry.lower() in aliases:
            return full
    raise FileNotFoundError(
        f"Aucun dossier parmi {sorted(aliases)} trouve dans {base_dir}")


def list_dataset():
    """Charge une seule fois la liste des chemins d'images disponibles dans
    database/A/<classe> et database/B/<classe>, sans encore rien
    echantillonner (les deux sous-dossiers A et B sont combines par classe)."""
    with timer("Listage des images (A et B combines)"):
        pools = {"empty": [], "occupied": []}
        alias_by_class = {"empty": EMPTY_NAMES, "occupied": OCCUPIED_NAMES}

        for class_name, aliases in alias_by_class.items():
            for sub in SUB_FOLDERS:
                sub_dir = os.path.join(DATA_DIR, sub)
                folder = find_class_folder(sub_dir, aliases)
                imgs = list_images(folder)
                pools[class_name].extend(imgs)
                print(f"  {sub}/{os.path.basename(folder)} : {len(imgs)} images")

        print(f"  Total 'empty'    : {len(pools['empty'])} images")
        print(f"  Total 'occupied' : {len(pools['occupied'])} images")

    return pools


def sample_split(pools, seed=RANDOM_SEED):
    """Melange le pool complet de chaque classe (A+B combines) puis en tire
    N_TRAIN_PER_CLASS pour l'entrainement et N_TEST_PER_CLASS pour le test,
    sans recouvrement entre les deux (memes images jamais dans les 2)."""
    rng = random.Random(seed)
    train_images, test_images = {}, {}

    for class_name in LABELS:
        images = list(pools[class_name])
        rng.shuffle(images)

        needed = N_TRAIN_PER_CLASS + N_TEST_PER_CLASS
        if len(images) < needed:
            print(f"[!] {class_name} : seulement {len(images)} images "
                  f"(besoin de {needed} = {N_TRAIN_PER_CLASS} train + {N_TEST_PER_CLASS} test).")

        # Le test set est reserve EN PREMIER (jamais vide tant qu'il y a au
        # moins quelques images), le train prend ce qu'il reste.
        n_te = min(N_TEST_PER_CLASS, len(images))
        n_tr = min(N_TRAIN_PER_CLASS, max(len(images) - n_te, 0))
        test_images[class_name] = images[:n_te]
        train_images[class_name] = images[n_te:n_te + n_tr]

        print(f"[split] {class_name} : {n_tr} pour train, {n_te} pour test")

    return train_images, test_images


# ==============================================================================
# 4) CONSTRUCTION DU MODELE (model.txt)
# ==============================================================================

def build_model(train_images, write_file=True):
    label_name = "Construction de model.txt" if write_file else "Extraction training (tirage de controle)"
    with timer(label_name):
        model_entries = []
        for class_name, label in LABELS.items():
            images = train_images[class_name]
            print(f"[model] {class_name} ({label}) : {len(images)} images utilisees pour l'entrainement")
            for img_path in images:
                vector = extract_feature_vector(img_path)
                model_entries.append((label, vector))

        if write_file:
            with open(MODEL_FILE, "w") as f_out:
                for label, vector in model_entries:
                    vector_str = ",".join(str(v) for v in vector)
                    f_out.write(f"{label};{vector_str}\n")
            print(f"[model] {MODEL_FILE} ecrit avec {len(model_entries)} vecteurs "
                  f"(attendu : {N_TRAIN_PER_CLASS * len(LABELS)})")

    return model_entries


def load_model(path=MODEL_FILE):
    model_entries = []
    with open(path) as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            label_str, vector_str = line.split(";")
            label = int(label_str)
            vector = [int(v) for v in vector_str.split(",")]
            model_entries.append((label, vector))
    return model_entries


# ==============================================================================
# 4bis) GRAPHIQUES DES HISTOGRAMMES
# ==============================================================================

def plot_sample_histograms(train_images,
                           out_file=os.path.join(FEATURE_DIR, "histogramme_exemple.png")):
    """Construction de la mosaique, pour 1 exemple par classe :
    image couleur -> R, G, B separes -> grosse image concatenee -> LBP -> histogramme."""
    with timer("Graphique histogramme_exemple.png"):
        fig, axes = plt.subplots(len(LABELS), 4, figsize=(16, 4 * len(LABELS)))
        if len(LABELS) == 1:
            axes = axes.reshape(1, -1)

        for row, class_name in enumerate(LABELS):
            img_path = train_images[class_name][0]

            img_rgb = np.array(Image.open(img_path).convert("RGB").resize(RESIZE_SIZE))
            r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
            big_image = np.hstack([r, g, b])
            lbp_codes = compute_lbp(big_image)
            vector = extract_feature_vector_global(img_path)

            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title(f"'{class_name}' (label {LABELS[class_name]}) - RGB")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(big_image, cmap="gray")
            axes[row, 1].set_title("Grosse image = R | G | B")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(lbp_codes, cmap="gray")
            axes[row, 2].set_title("Codes LBP")
            axes[row, 2].axis("off")

            axes[row, 3].bar(range(256), vector, width=1.0, color="tab:blue")
            axes[row, 3].set_title("Histogramme LBP (256 valeurs)")
            axes[row, 3].set_xlabel("code LBP (0-255)")
            axes[row, 3].set_ylabel("nb de pixels")

        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
    print(f"[graph] {out_file} enregistre")


def plot_multiscale_example(train_images,
                            out_file=os.path.join(FEATURE_DIR, "multi_echelle_exemple.png")):
    """Pour 1 exemple par classe : image + grille 4x4, codes LBP uniformes,
    et vecteur multi-echelle complet (1x1 | 2x2 | 4x4)."""
    with timer("Graphique multi_echelle_exemple.png"):
        fig, axes = plt.subplots(len(LABELS), 3, figsize=(16, 4 * len(LABELS)),
                                 gridspec_kw={"width_ratios": [1, 1, 2.5]})

        fine = GRID_SCALES[-1]
        for row, class_name in enumerate(LABELS):
            img_path = train_images[class_name][0]
            gray = load_gray(img_path)
            codes = UNIFORM_LUT[compute_lbp(gray)]
            vector = extract_feature_vector_multiscale(img_path)
            h, w = gray.shape

            ax = axes[row, 0]
            ax.imshow(gray, cmap="gray")
            for k in range(1, fine):
                ax.axhline(k * h / fine - 0.5, color="yellow", lw=0.8)
                ax.axvline(k * w / fine - 0.5, color="yellow", lw=0.8)
            ax.set_title(f"'{class_name}' (label {LABELS[class_name]}) - grille {fine}x{fine}")
            ax.axis("off")

            axes[row, 1].imshow(codes, cmap="nipy_spectral")
            axes[row, 1].set_title("Codes LBP uniformes (59 bins)")
            axes[row, 1].axis("off")

            ax = axes[row, 2]
            ax.bar(range(len(vector)), vector, width=1.0, color="tab:blue")
            start = 0
            for grid in GRID_SCALES:
                size = grid * grid * N_BINS
                ax.axvline(start - 0.5, color="red", lw=1)
                ax.text(start + size / 2, ax.get_ylim()[1] * 0.95, f"{grid}x{grid}",
                        ha="center", va="top", color="red")
                start += size
            ax.set_title(f"Vecteur multi-echelle ({len(vector)} valeurs)")
            ax.set_xlabel("indice (bloc x bin LBP)")
            ax.set_ylabel("nb de pixels")

        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
    print(f"[graph] {out_file} enregistre")


def plot_pyramid_levels(train_images, out_dir=FEATURE_DIR):
    """Construction de la pyramide, niveau par niveau, pour 1 exemple par
    classe : 1 histogramme sur l'image entiere, puis 4 histogrammes sur des
    blocs 2 fois plus petits, puis 16 sur des blocs encore 2 fois plus
    petits... et a la fin tous les histogrammes mis bout a bout forment le
    vecteur final. Verifie au passage que les histogrammes d'un niveau
    additionnes redonnent l'histogramme de l'image entiere."""
    fine = GRID_SCALES[-1]
    for class_name in LABELS:
        out_file = os.path.join(out_dir, f"pyramide_histogrammes_{class_name}.png")
        with timer(f"Graphique pyramide_histogrammes_{class_name}.png"):
            gray = load_gray(train_images[class_name][0])
            codes = UNIFORM_LUT[compute_lbp(gray)]
            h, w = gray.shape
            whole = np.bincount(codes.ravel(), minlength=N_BINS)

            n_levels = len(GRID_SCALES)
            fig = plt.figure(figsize=(14, 3.6 * (n_levels + 1)))
            outer = fig.add_gridspec(n_levels + 1, 2, width_ratios=[1, 4],
                                     hspace=0.45, wspace=0.08, top=0.94)

            all_hists = []
            for lvl, grid in enumerate(GRID_SCALES):
                hists = block_histograms(codes, grid)
                all_hists.extend(hists)
                sums_ok = np.array_equal(np.sum(hists, axis=0), whole)

                ax_img = fig.add_subplot(outer[lvl, 0])
                ax_img.imshow(gray, cmap="gray")
                for k in range(1, grid):
                    ax_img.axhline(k * h / grid - 0.5, color="yellow", lw=1)
                    ax_img.axvline(k * w / grid - 0.5, color="yellow", lw=1)
                ax_img.set_title(f"niveau {lvl} : grille {grid}x{grid}\n"
                                 f"blocs de {h // grid}x{w // grid} pixels", fontsize=10)
                ax_img.axis("off")

                inner = outer[lvl, 1].subgridspec(grid, grid, hspace=0.15, wspace=0.08)
                ymax = max(hh.max() for hh in hists)
                for idx, hist in enumerate(hists):
                    ax = fig.add_subplot(inner[idx // grid, idx % grid])
                    ax.bar(range(N_BINS), hist, width=1.0, color=f"C{lvl}")
                    ax.set_ylim(0, ymax * 1.05)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if idx == 0:
                        ax.set_title(f"{grid * grid} histogramme(s) de {N_BINS} bins"
                                     f" - somme = histogramme 1x1 : "
                                     f"{'oui' if sums_ok else 'NON'}",
                                     loc="left", fontsize=10)

            vector = np.concatenate(all_hists)
            ax = fig.add_subplot(outer[n_levels, :])
            start = 0
            for lvl, grid in enumerate(GRID_SCALES):
                size = grid * grid * N_BINS
                ax.bar(range(start, start + size), vector[start:start + size],
                       width=1.0, color=f"C{lvl}", label=f"niveau {lvl} ({grid}x{grid})")
                start += size
            ax.set_xlim(-0.5, len(vector) - 0.5)
            ax.set_title(f"Vecteur final = les {len(all_hists)} histogrammes mis bout a bout "
                         f"({len(vector)} valeurs)", fontsize=11)
            ax.set_xlabel("indice (bloc x bin LBP)")
            ax.set_ylabel("nb de pixels")
            ax.legend(loc="upper right")

            fig.suptitle(f"Pyramide LBP {'-'.join(str(g) for g in GRID_SCALES)} - "
                         f"exemple '{class_name}' (grille la plus fine : {fine}x{fine})",
                         fontsize=13)
            fig.savefig(out_file, dpi=130, bbox_inches="tight")
            plt.close(fig)
        print(f"[graph] {out_file} enregistre")


def plot_average_histograms(model_entries, out_file=os.path.join(FEATURE_DIR, "histogramme_moyen.png")):
    with timer("Graphique histogramme_moyen.png"):
        vectors_by_label = {0: [], 1: []}
        for label, vector in model_entries:
            vectors_by_label[label].append(vector)

        fig, ax = plt.subplots(figsize=(12 if FEATURE_MODE == "multiscale" else 9, 5))
        names = {v: k for k, v in LABELS.items()}
        colors = {0: "tab:blue", 1: "tab:red"}

        for label, vectors in vectors_by_label.items():
            if not vectors:
                continue
            mean_vector = np.mean(vectors, axis=0)
            ax.plot(mean_vector, label=f"{names[label]} (label {label})", color=colors[label])

        if FEATURE_MODE == "multiscale":
            start = 0
            for grid in GRID_SCALES:
                ax.axvline(start - 0.5, color="k", lw=1, linestyle="--")
                ax.text(start + grid * grid * N_BINS / 2, ax.get_ylim()[1] * 0.95,
                        f"{grid}x{grid}", ha="center", va="top")
                start += grid * grid * N_BINS
            ax.set_xlabel("indice (bloc x bin LBP)")
        else:
            ax.set_xlabel("code LBP (0-255)")
        ax.set_ylabel("nb moyen de pixels")
        ax.set_title("Histogramme LBP moyen par classe (training set)")
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
    print(f"[graph] {out_file} enregistre")


# ==============================================================================
# 5) DISTANCE SAD (Sum of Absolute Differences) - VERSION VECTORISEE (numpy)
# ==============================================================================
# A 5000+5000 images d'entrainement, comparer une image de test aux 10000
# vecteurs un par un en pur Python (boucle + zip) est beaucoup trop lent
# (des millions de calculs de distance au total). On convertit donc le
# training set en une seule matrice numpy et on calcule TOUTES les distances
# d'un coup avec des operations vectorisees (meme resultat, juste rapide).

def sad_distance(vec_a, vec_b):
    """Version pure Python gardee pour reference / petits volumes."""
    return sum(abs(a - b) for a, b in zip(vec_a, vec_b))


def model_to_matrix(model_entries):
    """Convertit la liste [(label, vecteur), ...] en une matrice numpy
    (n_images x taille du vecteur) et un vecteur de labels, pour un calcul
    vectorise. float32 : 2x moins de memoire a parcourir que float64, et les
    distances SAD (entiers < 2^24) restent exactes."""
    labels = np.array([label for label, _ in model_entries], dtype=np.int32)
    matrix = np.array([vec for _, vec in model_entries], dtype=np.float32)
    return matrix, labels


def classify_vector_fast(test_vector, train_matrix, train_labels):
    """1-NN vectorise (distance SAD) : calcule la distance entre test_vector
    et TOUS les vecteurs d'entrainement en une seule operation numpy."""
    test_vec = np.asarray(test_vector, dtype=np.float32)
    distances = np.sum(np.abs(train_matrix - test_vec), axis=1)
    best_idx = int(np.argmin(distances))
    return int(train_labels[best_idx]), float(distances[best_idx])


# ------------------------------------------------------------------------
# 5bis) AUTRES METHODES DE COMPARAISON D'HISTOGRAMMES (equivalent a
# cv2.compareHist, mais vectorise en numpy pour comparer 1 image de test a
# TOUTES les images d'entrainement d'un coup, au lieu d'appeler
# cv2.compareHist des millions de fois en boucle Python).
# Formules verifiees numeriquement identiques a cv2.HISTCMP_* (a l'epsilon
# de precision float32 pres).
# ------------------------------------------------------------------------

def dist_correlation(query, cand):
    """cv2.HISTCMP_CORREL - similarite, plus GRAND = plus proche."""
    qm, cm = query.mean(), cand.mean(axis=1, keepdims=True)
    qc, cc = query - qm, cand - cm
    num = (cc * qc).sum(axis=1)
    den = np.sqrt((qc ** 2).sum() * (cc ** 2).sum(axis=1))
    return num / np.clip(den, 1e-10, None)


def dist_chisquare(query, cand):
    """cv2.HISTCMP_CHISQR - distance, plus PETIT = plus proche."""
    mask = query > 1e-10
    diff2 = (cand - query) ** 2
    denom = np.where(mask, query, 1.0)
    terms = np.where(mask, diff2 / denom, 0.0)
    return terms.sum(axis=1)


def dist_intersection(query, cand):
    """cv2.HISTCMP_INTERSECT - similarite, plus GRAND = plus proche."""
    return np.minimum(cand, query).sum(axis=1)


def dist_bhattacharyya(query, cand):
    """cv2.HISTCMP_BHATTACHARYYA - distance, plus PETIT = plus proche."""
    q = query / query.sum()
    p = cand / cand.sum(axis=1, keepdims=True)
    bc = np.sum(np.sqrt(p * q), axis=1)
    return np.sqrt(np.clip(1 - bc, 0, None))


def dist_chisquare_alt(query, cand):
    """cv2.HISTCMP_CHISQR_ALT - distance, plus PETIT = plus proche."""
    diff2 = (cand - query) ** 2
    denom = cand + query
    terms = np.where(denom > 1e-10, diff2 / np.where(denom > 1e-10, denom, 1), 0.0)
    return 2 * terms.sum(axis=1)


def dist_kl_divergence(query, cand):
    """cv2.HISTCMP_KL_DIV - distance, plus PETIT = plus proche."""
    mask = query > 1e-10
    ratio = np.where(mask, query[None, :] / np.clip(cand, 1e-10, None), 1.0)
    terms = np.where(mask, query[None, :] * np.log(ratio), 0.0)
    return terms.sum(axis=1)


def dist_sad(query, cand):
    """SAD (methode initiale) - distance, plus PETIT = plus proche."""
    return np.sum(np.abs(cand - query), axis=1)


# Chaque methode : (fonction, higher_is_better)
DISTANCE_METHODS = {
    "SAD":           (dist_sad, False),
    "Correlation":   (dist_correlation, True),
    "ChiSquare":     (dist_chisquare, False),
    "Intersection":  (dist_intersection, True),
    "Bhattacharyya": (dist_bhattacharyya, False),
    "ChiSquareAlt":  (dist_chisquare_alt, False),
    "KLDivergence":  (dist_kl_divergence, False),
}


def compare_distance_methods(model_entries, test_images,
                             out_txt=os.path.join(DISTANCE_DIR, "comparaison_distances.txt"),
                             out_png=os.path.join(DISTANCE_DIR, "comparaison_distances.png")):
    """Classe tout le test set avec CHAQUE methode de distance/similarite,
    et compare leurs accuracies (SAD + les 6 methodes cv2.compareHist)."""
    with timer("Comparaison des methodes de distance"):
        train_matrix, train_labels = model_to_matrix(model_entries)
        # float32 : 2x moins de memoire a parcourir (vecteurs de 1239 valeurs)
        train_matrix = train_matrix.astype(np.float32)

        per_method_correct = {name: 0 for name in DISTANCE_METHODS}
        per_method_total = 0
        rows = []  # pour le fichier texte

        def predict_all_methods(img_path):
            test_vector = np.asarray(extract_feature_vector(img_path), dtype=np.float32)
            preds = []
            for name, (fn, higher_is_better) in DISTANCE_METHODS.items():
                scores = fn(test_vector, train_matrix)
                best_idx = int(np.argmax(scores) if higher_is_better else np.argmin(scores))
                preds.append(int(train_labels[best_idx]))
            return preds

        # numpy libere le GIL sur les gros tableaux : on traite plusieurs images
        # de test en parallele (threads), et on affiche l'avancement.
        jobs = [(p, lbl) for cn, lbl in LABELS.items() for p in test_images[cn]]
        n_workers = min(4, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i, ((img_path, true_label), preds) in enumerate(
                    zip(jobs, pool.map(lambda job: predict_all_methods(job[0]), jobs)), 1):
                per_method_total += 1
                for name, pred_label in zip(DISTANCE_METHODS, preds):
                    per_method_correct[name] += int(pred_label == true_label)
                rows.append([img_path, str(true_label)] + [str(x) for x in preds])
                if i % 100 == 0 or i == len(jobs):
                    print(f"[compare] {i}/{len(jobs)} images classees")

        accuracies = {name: per_method_correct[name] / per_method_total
                      for name in DISTANCE_METHODS}

        with open(out_txt, "w") as f_out:
            f_out.write("path;true;" + ";".join(DISTANCE_METHODS.keys()) + "\n")
            for row in rows:
                f_out.write(";".join(row) + "\n")
        print(f"[compare] {out_txt} ecrit ({per_method_total} images x {len(DISTANCE_METHODS)} methodes)")

        print("\n=== Accuracy par methode de distance ===")
        for name, acc in sorted(accuracies.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<15} {acc * 100:5.1f}%")

        fig, ax = plt.subplots(figsize=(9, 5))
        names = list(accuracies.keys())
        values = [accuracies[n] * 100 for n in names]
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        ax.bar(names, values, color=colors)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"Comparaison des methodes de distance ({per_method_total} images de test)")
        ax.set_ylim(0, 110)  # marge au-dessus de 100 % pour les etiquettes
        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center")
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[graph] {out_png} enregistre")

    return accuracies


# ==============================================================================
# 6) CLASSIFICATION DU TEST SET (test.txt)
# ==============================================================================

def classify_test_set(model_entries, test_images, write_file=True, return_confusion=False):
    """1-NN SAD sur tout le test set. Renvoie l'accuracy, et en plus la
    matrice de confusion (lignes = reel, colonnes = predit) si
    return_confusion=True."""
    label_name = "Classification / construction de test.txt" if write_file else "Classification (tirage de controle)"
    with timer(label_name):
        # Une seule conversion en matrice numpy, reutilisee pour toutes les
        # images de test (au lieu de reparcourir la liste Python a chaque fois)
        train_matrix, train_labels = model_to_matrix(model_entries)

        total, correct = 0, 0
        confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        total_distance_calculations = 0
        rows = []

        for class_name, true_label in LABELS.items():
            images = test_images[class_name]
            print(f"[test] {class_name} ({true_label}) : {len(images)} images utilisees pour le test")
            for img_path in images:
                test_vector = extract_feature_vector(img_path)
                predicted_label, min_distance = classify_vector_fast(test_vector, train_matrix, train_labels)
                total_distance_calculations += len(model_entries)
                total += 1
                correct += int(predicted_label == true_label)
                confusion[true_label, predicted_label] += 1
                rows.append((img_path, true_label, predicted_label, min_distance))

        if write_file:
            with open(TEST_RESULT_FILE, "w") as f_out:
                f_out.write("chemin;label_reel;label_predit;distance_min\n")
                for img_path, true_label, predicted_label, min_distance in rows:
                    f_out.write(f"{img_path};{true_label};{predicted_label};{min_distance}\n")
            print(f"[test] {TEST_RESULT_FILE} ecrit ({total} images)")

        accuracy = correct / total if total else 0.0
        print(f"[test] Nombre total de calculs de distance : {total_distance_calculations}")
        print(f"[test] Accuracy : {accuracy * 100:.1f}% ({correct}/{total} bonnes predictions)")

    return (accuracy, confusion) if return_confusion else accuracy


# ==============================================================================
# 7) VALIDATION CROISEE
# ==============================================================================

def run_cross_validation(pools):
    summary = []  # (seed, accuracy)
    official_confusion = None

    for i, seed in enumerate(CV_SEEDS):
        is_official = (i == 0)
        print(f"\n=== Tirage {i + 1}/{len(CV_SEEDS)} (seed={seed}"
              f"{', OFFICIEL' if is_official else ''}) ===")

        with timer(f"Tirage complet seed={seed}"):
            train_images, test_images = sample_split(pools, seed=seed)
            model_entries = build_model(train_images, write_file=is_official)
            if is_official:
                if FEATURE_MODE == "multiscale":
                    plot_multiscale_example(train_images)
                    plot_pyramid_levels(train_images)
                else:
                    plot_sample_histograms(train_images)
                plot_average_histograms(model_entries)
            accuracy, confusion = classify_test_set(model_entries, test_images,
                                                    write_file=is_official, return_confusion=True)
            if is_official:
                official_confusion = confusion
                compare_distance_methods(model_entries, test_images)

        summary.append((seed, accuracy))

    return summary, official_confusion


# ==============================================================================
# 7bis) EXTENSION COULEUR : COMPARAISON DES 3 STRATEGIES
# ==============================================================================

def compare_color_strategies(pools, seeds=COLOR_SEEDS,
                             out_txt=os.path.join(COLOR_DIR, "comparaison_couleur.txt"),
                             out_png=os.path.join(COLOR_DIR, "comparaison_couleur.png"),
                             out_mosaic=os.path.join(COLOR_DIR, "mosaique_couleur.png")):
    """Compare gris / mosaique / un LBP par plan (1-NN SAD) sur plusieurs
    tirages aleatoires. Les 3 strategies voient exactement les memes
    train/test a chaque tirage : la difference mesuree ne vient que de la
    caracterisation."""
    global FEATURE_MODE
    original_mode = FEATURE_MODE
    accuracies = {name: [] for name in COLOR_STRATEGIES}

    try:
        with timer("Comparaison des strategies couleur"):
            for i, seed in enumerate(seeds, 1):
                print(f"\n=== Couleur : tirage {i}/{len(seeds)} (seed={seed}) ===")
                train_images, test_images = sample_split(pools, seed=seed)
                if i == 1:
                    plot_sample_histograms(train_images, out_file=out_mosaic)
                for name in COLOR_STRATEGIES:
                    FEATURE_MODE = name
                    print(f"--- strategie : {name} ---")
                    model_entries = build_model(train_images, write_file=False)
                    accuracies[name].append(
                        classify_test_set(model_entries, test_images, write_file=False))
    finally:
        FEATURE_MODE = original_mode

    means = {n: float(np.mean(a)) for n, a in accuracies.items()}
    stds = {n: float(np.std(a)) for n, a in accuracies.items()}

    with open(out_txt, "w") as f_out:
        f_out.write("seed;" + ";".join(COLOR_STRATEGIES) + "\n")
        for i, seed in enumerate(seeds):
            f_out.write(f"{seed};" + ";".join(f"{accuracies[n][i]:.4f}" for n in COLOR_STRATEGIES) + "\n")
    print(f"[couleur] {out_txt} ecrit")

    print(f"\n=== Strategies couleur ({len(seeds)} tirages) ===")
    for name in COLOR_STRATEGIES:
        print(f"  {name:<10} moyenne = {means[name] * 100:.2f}%  "
              f"ecart-type = {stds[name] * 100:.2f} pts")
    gap = (max(means.values()) - min(means.values())) * 100
    print(f"Ecart entre la meilleure et la moins bonne strategie : {gap:.2f} points")

    with timer("Graphique comparaison_couleur.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        names = COLOR_STRATEGIES
        ax.bar(names, [means[n] * 100 for n in names], yerr=[stds[n] * 100 for n in names],
               capsize=6, color=["tab:gray", "tab:orange", "tab:green"])
        for i, n in enumerate(names):
            ax.text(i, means[n] * 100 + stds[n] * 100 + 0.15, f"{means[n] * 100:.2f}%", ha="center")
        lo = min(means.values()) * 100
        ax.set_ylim(max(0, lo - 3), 100.8)
        ax.set_ylabel("accuracy moyenne (%)")
        ax.set_title(f"Strategies couleur : {len(seeds)} tirages "
                     f"(ecart max = {gap:.2f} pt)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    print(f"[graph] {out_png} enregistre")

    return accuracies


def print_and_plot_cv_summary(summary, out_file=os.path.join(PYRAMID_DIR, "validation_croisee.png")):
    accuracies = [a for _, a in summary]
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    print("\n=== Resume validation croisee ===")
    for seed, accuracy in summary:
        print(f"  seed={seed:<3} accuracy={accuracy * 100:.1f}%")
    print(f"Accuracy moyenne : {mean_acc * 100:.1f}% (ecart-type : {std_acc * 100:.2f} points)")

    with timer("Graphique validation_croisee.png"):
        fig, ax = plt.subplots(figsize=(7, 5))
        seeds = [str(s) for s, _ in summary]
        ax.bar(seeds, [a * 100 for a in accuracies], color="tab:blue")
        ax.axhline(mean_acc * 100, color="k", linestyle="--",
                    label=f"moyenne = {mean_acc * 100:.1f}%")
        ax.set_xlabel("seed du tirage")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"Accuracy sur {len(summary)} tirages (ecart-type = {std_acc * 100:.2f} pts)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
    print(f"[graph] {out_file} enregistre")


def plot_pyramid_draws(summary, confusion,
                       out_csv=os.path.join(PYRAMID_DIR, "pyramide_tirages.csv"),
                       out_png=os.path.join(PYRAMID_DIR, "pyramide_tirages.png")):
    """Resultat de la pyramide LBP tirage par tirage (meme presentation que
    cnn_vs_pyramide.png de la branche deep-learning-cnn) : accuracy par
    tirage + moyenne en pointilles, et matrice de confusion du tirage
    officiel."""
    accuracies = [a * 100 for _, a in summary]
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))
    pyramid = "-".join(str(g) for g in GRID_SCALES)

    with open(out_csv, "w") as f_out:
        f_out.write(f"tirage,seed,LBP pyramide {pyramid} (%)\n")
        for i, (seed, _) in enumerate(summary, 1):
            f_out.write(f"{i},{seed},{accuracies[i - 1]:.1f}\n")
        f_out.write(f"moyenne,,{mean_acc:.2f}\n")
        f_out.write(f"ecart-type,,{std_acc:.2f}\n")
    print(f"[pyramide] {out_csv} ecrit")

    with timer("Graphique pyramide_tirages.png"):
        fig, (ax, ax_cm) = plt.subplots(1, 2, figsize=(14, 5),
                                        gridspec_kw={"width_ratios": [2, 1]})
        x = list(range(len(summary)))
        ax.plot(x, accuracies, "-o", color="tab:blue", lw=2,
                label=f"Pyramide LBP {pyramid}  (moy. {mean_acc:.2f}%)")
        ax.axhline(mean_acc, color="tab:blue", linestyle="--", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"tirage {i}\n(seed {s})" for i, (s, _) in enumerate(summary, 1)])
        ax.set_ylabel("accuracy test (%)")
        ax.set_title(f"Pyramide LBP {pyramid} - {N_TRAIN_PER_CLASS}/classe en train\n"
                     f"(pointilles = moyenne, ecart-type = {std_acc:.2f} pt)")
        ax.grid(alpha=0.3)
        ax.legend()

        names = list(LABELS)
        official_acc = np.trace(confusion) / confusion.sum() * 100
        ax_cm.imshow(confusion, cmap="viridis")
        for r in range(len(names)):
            for c in range(len(names)):
                v = confusion[r, c]
                ax_cm.text(c, r, str(v), ha="center", va="center", fontsize=14,
                           color="black" if v > confusion.max() / 2 else "white")
        ax_cm.set_xticks(range(len(names)))
        ax_cm.set_xticklabels(names)
        ax_cm.set_yticks(range(len(names)))
        ax_cm.set_yticklabels(names)
        ax_cm.set_xlabel("Predit")
        ax_cm.set_ylabel("Reel")
        ax_cm.set_title(f"Matrice de confusion (seed {summary[0][0]})\n"
                        f"Pyramide {pyramid} : {official_acc:.1f}%")

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    print(f"[graph] {out_png} enregistre")


def print_timing_report(out_file=TIMING_FILE):
    lines = ["=== Chronometre par etape ==="]
    total = 0.0
    for label, elapsed in TIMINGS.items():
        lines.append(f"  {label:<55} {elapsed:>7.2f} s")
        total += elapsed
    lines.append(f"  {'TOTAL':<55} {total:>7.2f} s")
    print("\n" + "\n".join(lines))
    with open(out_file, "w") as f_out:
        f_out.write("\n".join(lines) + "\n")
    print(f"[timer] {out_file} ecrit")


def main():
    print(f"[config] FEATURE_MODE = {FEATURE_MODE}")
    color_only = "--color-only" in sys.argv
    pools = list_dataset()
    if not color_only:
        summary, confusion = run_cross_validation(pools)
        print_and_plot_cv_summary(summary)
        plot_pyramid_draws(summary, confusion)
    if color_only or RUN_COLOR_COMPARISON:
        compare_color_strategies(pools)
    print_timing_report()


if __name__ == "__main__":
    main()