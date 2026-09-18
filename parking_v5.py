import os
import random
import time
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

MODEL_FILE = "model.txt"
TEST_RESULT_FILE = "test.txt"

N_TRAIN_PER_CLASS = 3682   # limite par la classe la plus petite ("empty" = 4182 images)
N_TEST_PER_CLASS = 500     # -> chaque classe garde vraiment 500 images de test

LABELS = {"empty": 0, "occupied": 1}
EMPTY_NAMES = {"empty", "free", "vacant"}
OCCUPIED_NAMES = {"occupied", "busy", "taken"}

RANDOM_SEED = 42
# Le 1er tirage (RANDOM_SEED) est le tirage "officiel" : c'est le seul qui
# ecrit model.txt / test.txt / les graphes. Les autres ne servent qu'a
# mesurer la stabilite de l'accuracy (validation croisee).
CV_SEEDS = [RANDOM_SEED, 1, 2, 3, 4]

RESIZE_SIZE = (64, 64)
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Stocke la duree de chaque etape du run officiel, pour le rapport final
TIMINGS = {}


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
# 2) EXTRACTION DU VECTEUR DE CARACTERISTIQUES (256 valeurs)
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


def extract_feature_vector(image_path):
    """Nouvelle caracterisation (couleur + LBP) :
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

def plot_sample_histograms(train_images, out_file="histogramme_exemple.png"):
    """Illustre le pipeline complet pour 1 exemple par classe :
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
            vector = extract_feature_vector(img_path)

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


def plot_average_histograms(model_entries, out_file="histogramme_moyen.png"):
    with timer("Graphique histogramme_moyen.png"):
        vectors_by_label = {0: [], 1: []}
        for label, vector in model_entries:
            vectors_by_label[label].append(vector)

        fig, ax = plt.subplots(figsize=(9, 5))
        names = {v: k for k, v in LABELS.items()}
        colors = {0: "tab:blue", 1: "tab:red"}

        for label, vectors in vectors_by_label.items():
            if not vectors:
                continue
            mean_vector = np.mean(vectors, axis=0)
            ax.plot(mean_vector, label=f"{names[label]} (label {label})", color=colors[label])

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
    (n_images x 256) et un vecteur de labels, pour un calcul vectorise."""
    labels = np.array([label for label, _ in model_entries], dtype=np.int32)
    matrix = np.array([vec for _, vec in model_entries], dtype=np.float64)
    return matrix, labels


def classify_vector_fast(test_vector, train_matrix, train_labels):
    """1-NN vectorise (distance SAD) : calcule la distance entre test_vector
    et TOUS les vecteurs d'entrainement en une seule operation numpy."""
    test_vec = np.asarray(test_vector, dtype=np.float64)
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
                              out_txt="comparaison_distances.txt",
                              out_png="comparaison_distances.png"):
    """Classe tout le test set avec CHAQUE methode de distance/similarite,
    et compare leurs accuracies (SAD + les 6 methodes cv2.compareHist)."""
    with timer("Comparaison des methodes de distance"):
        train_matrix, train_labels = model_to_matrix(model_entries)

        per_method_correct = {name: 0 for name in DISTANCE_METHODS}
        per_method_total = 0
        rows = []  # pour le fichier texte

        for class_name, true_label in LABELS.items():
            for img_path in test_images[class_name]:
                test_vector = np.asarray(extract_feature_vector(img_path), dtype=np.float64)
                per_method_total += 1
                row = [img_path, str(true_label)]
                for name, (fn, higher_is_better) in DISTANCE_METHODS.items():
                    scores = fn(test_vector, train_matrix)
                    best_idx = int(np.argmax(scores) if higher_is_better else np.argmin(scores))
                    pred_label = int(train_labels[best_idx])
                    per_method_correct[name] += int(pred_label == true_label)
                    row.append(str(pred_label))
                rows.append(row)

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
        ax.set_ylim(0, 100)
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

def classify_test_set(model_entries, test_images, write_file=True):
    label_name = "Classification / construction de test.txt" if write_file else "Classification (tirage de controle)"
    with timer(label_name):
        # Une seule conversion en matrice numpy, reutilisee pour toutes les
        # images de test (au lieu de reparcourir la liste Python a chaque fois)
        train_matrix, train_labels = model_to_matrix(model_entries)

        total, correct = 0, 0
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

    return accuracy


# ==============================================================================
# 7) VALIDATION CROISEE
# ==============================================================================

def run_cross_validation(pools):
    summary = []  # (seed, accuracy)

    for i, seed in enumerate(CV_SEEDS):
        is_official = (i == 0)
        print(f"\n=== Tirage {i + 1}/{len(CV_SEEDS)} (seed={seed}"
              f"{', OFFICIEL' if is_official else ''}) ===")

        with timer(f"Tirage complet seed={seed}"):
            train_images, test_images = sample_split(pools, seed=seed)
            model_entries = build_model(train_images, write_file=is_official)
            if is_official:
                plot_sample_histograms(train_images)
                plot_average_histograms(model_entries)
            accuracy = classify_test_set(model_entries, test_images, write_file=is_official)
            if is_official:
                compare_distance_methods(model_entries, test_images)

        summary.append((seed, accuracy))

    return summary


def print_and_plot_cv_summary(summary, out_file="validation_croisee.png"):
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


def print_timing_report():
    print("\n=== Chronometre par etape (tirage officiel) ===")
    total = 0.0
    for label, elapsed in TIMINGS.items():
        print(f"  {label:<55} {elapsed:>7.2f} s")
        total += elapsed
    print(f"  {'TOTAL':<55} {total:>7.2f} s")


def main():
    pools = list_dataset()
    summary = run_cross_validation(pools)
    print_and_plot_cv_summary(summary)
    print_timing_report()


if __name__ == "__main__":
    main()