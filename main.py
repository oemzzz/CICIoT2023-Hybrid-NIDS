
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from scipy.stats import mode as scipy_mode

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# Tekrarlanabilirlik
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────
# 1. GLOBAL AYARLAR
# ─────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\ataka\OneDrive\Desktop\CICIOT23")
OUTPUT_DIR = Path("outputs")
MODEL_DIR  = Path("saved_models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE  = 42
SAMPLE_FRAC   = 0.20
TEST_SIZE     = 0.20          # test = %20
VAL_SIZE      = 0.10          # kalan %80'in %10'u -> doğrulama %8, eğitim %72

CNN_EPOCHS    = 60
CNN_BATCH     = 512
CNN_PATIENCE  = 10
CNN_FEATURES  = 128
WEIGHT_CAP_RATIO = 10.0       # [FIX-1] ağırlık üst sınırı = 10 × medyan

RF_N_TREES    = 200
XGB_N_ROUNDS  = 500
LGB_N_ROUNDS  = 500

MACRO_MODE    = False         # False = 34 sınıf (rapor bu kurguya dayanıyor)

MACRO_MAP = {
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    "Mirai-greeth_flood": "Botnet", "Mirai-greip_flood": "Botnet",
    "Mirai-udpplain": "Botnet", "MQTT-Publish": "Botnet",
    "Recon-PingSweep": "Reconnaissance", "Recon-OsScan": "Reconnaissance",
    "Recon-PortScan": "Reconnaissance", "Recon-HostDiscovery": "Reconnaissance",
    "VulnerabilityScan": "Reconnaissance", "DNS_Spoofing": "Reconnaissance",
    "BruteForce-Web": "Brute Force", "BruteForce-XSS": "Brute Force",
    "DictionaryBruteForce": "Brute Force",
    "BenignTraffic": "Normal",
    "Backdoor_Malware": "Backdoor", "Uploading_Attack": "Backdoor",
    "SqlInjection": "Injection", "CommandInjection": "Injection", "XSS": "Injection",
}


# ─────────────────────────────────────────────────────────────────────────
# 2. VERİ YÜKLEME  ([FIX-6] bellek-güvenli, dosya-içi stratified örnekleme)
# ─────────────────────────────────────────────────────────────────────────
def _detect_label_col(columns):
    for col in columns:
        if col.strip().lower() in ("label", "labels", "class", "attack_type"):
            return col.strip()
    return columns[-1].strip()


def load_and_sample(data_dir: Path):
    """
    Her CSV tek tek okunur, float32'ye indirgenir, Inf/NaN temizlenir ve
    DOSYA İÇİNDE stratified örneklenir (SAMPLE_FRAC). Yalnızca örneklenen
    parçalar birleştirilir -> tepe bellek kullanımı düşük kalır.
    """
    csv_files = sorted(list(data_dir.rglob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"CSV bulunamadı: {data_dir}")
    print(f"  {len(csv_files)} CSV bulundu. Bellek-güvenli yükleme başlıyor...")

    parcalar = []
    label_col = None
    toplam_ham = 0
    for i, f in enumerate(csv_files, 1):
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if label_col is None:
            label_col = _detect_label_col(list(df.columns))

        # sayısal sütunları float32'ye indir
        num_cols = [c for c in df.columns
                    if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        df = df.dropna(subset=num_cols)
        toplam_ham += len(df)

        # dosya-içi stratified örnekleme
        if SAMPLE_FRAC < 1.0 and len(df) > 50:
            try:
                df, _ = train_test_split(
                    df, train_size=SAMPLE_FRAC,
                    stratify=df[label_col], random_state=RANDOM_STATE)
            except ValueError:
                # bir sınıfın o dosyada <2 örneği varsa stratify çöker -> rastgele
                df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

        parcalar.append(df)
        if i % 10 == 0 or i == len(csv_files):
            print(f"    [{i}/{len(csv_files)}] işlendi...")

    ornek = pd.concat(parcalar, ignore_index=True)
    del parcalar
    print(f"  Toplam temiz ham kayıt (yaklaşık): {toplam_ham:>12,}")
    print(f"  Örneklenen alt küme            : {len(ornek):>12,}")
    return ornek, label_col


def preprocess(data_dir: Path):
    print("\n" + "=" * 64)
    print("ADIM 1: Veri Yükleme ve Ön İşleme")
    print(f"  Mod: {'MAKRO (8 sınıf)' if MACRO_MODE else 'HAM (34 sınıf)'}")
    print("=" * 64)

    df, label_col = load_and_sample(data_dir)

    feature_cols = [c for c in df.columns
                    if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Sayısal öznitelik sayısı: {len(feature_cols)}")

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[label_col].astype(str).values
    if MACRO_MODE:
        y_raw = np.array([MACRO_MAP.get(lbl, lbl) for lbl in y_raw])

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"  Sınıf sayısı: {n_classes}")
    for i, cls in enumerate(le.classes_):
        print(f"    [{i:2d}] {cls:<32s} -> {(y == i).sum():>9,}")

    # Bölme: test %20 -> kalanın %10'u doğrulama (=> %72/%8/%20)
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=VAL_SIZE,
        stratify=y_tr_full, random_state=RANDOM_STATE)
    print(f"\n  Eğitim: {len(X_train):>9,} | Doğrulama: {len(X_val):>9,} "
          f"| Test: {len(X_test):>9,}")

    # Normalleştirme: parametreler YALNIZCA eğitimden öğrenilir
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(le,     MODEL_DIR / "label_encoder.joblib")

    # [FIX-1] Sınıf ağırlıkları + ÜST SINIR (capping)
    classes = np.unique(y_train)
    cw_arr = compute_class_weight("balanced", classes=classes, y=y_train)
    cap = WEIGHT_CAP_RATIO * np.median(cw_arr)
    cw_arr_capped = np.minimum(cw_arr, cap)
    print(f"\n  Ağırlık (capping öncesi) aralığı: "
          f"[{cw_arr.min():.2f}, {cw_arr.max():.2f}]")
    print(f"  Ağırlık ÜST SINIRI (10×medyan)  : {cap:.2f}")
    print(f"  Ağırlık (capping sonrası) aralığı: "
          f"[{cw_arr_capped.min():.2f}, {cw_arr_capped.max():.2f}]")

    cw_dict = dict(zip(classes, cw_arr_capped))
    sw_train = np.array([cw_dict[c] for c in y_train], dtype=np.float32)

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            le, n_classes, cw_dict, sw_train)


# ─────────────────────────────────────────────────────────────────────────
# 3. METRİK
# ─────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, label="") -> dict:
    r = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if label:
        print(f"  {label:32s} | Acc={r['accuracy']:.4f}  Prec={r['precision']:.4f}  "
              f"Rec={r['recall']:.4f}  F1={r['f1']:.4f}")
    return r


# ─────────────────────────────────────────────────────────────────────────
# 4. BASE MODELLER
# ─────────────────────────────────────────────────────────────────────────
def train_base_models(X_train, X_val, X_test, y_train, y_val, y_test,
                      n_classes, cw_dict, sw_train):
    print("\n" + "=" * 64)
    print("ADIM 2: Base (Yalın) Modellerin Eğitimi  [46 ham öznitelik]")
    print("=" * 64)
    results, preds = {}, {}
    sw_val = np.array([cw_dict.get(c, 1.0) for c in y_val], dtype=np.float32)

    # XGBoost
    print("\n  [XGBoost] eğitiliyor...")
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=XGB_N_ROUNDS, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        objective="multi:softmax", num_class=n_classes,
        eval_metric="mlogloss", early_stopping_rounds=20,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    m.fit(X_train, y_train, sample_weight=sw_train,
          eval_set=[(X_val, y_val)], sample_weight_eval_set=[sw_val], verbose=False)
    preds["XGBoost"] = m.predict(X_test)
    results["XGBoost_Base"] = compute_metrics(y_test, preds["XGBoost"], "XGBoost (Base)")
    results["XGBoost_Base"]["train_time"] = time.time() - t0
    joblib.dump(m, MODEL_DIR / "xgb_base.joblib")

    # Random Forest
    print("\n  [Random Forest] eğitiliyor...")
    t0 = time.time()
    m = RandomForestClassifier(
        n_estimators=RF_N_TREES, max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train, y_train)
    preds["Random Forest"] = m.predict(X_test)
    results["RF_Base"] = compute_metrics(y_test, preds["Random Forest"], "Random Forest (Base)")
    results["RF_Base"]["train_time"] = time.time() - t0
    joblib.dump(m, MODEL_DIR / "rf_base.joblib")

    # LightGBM
    print("\n  [LightGBM] eğitiliyor...")
    t0 = time.time()
    m = lgb.LGBMClassifier(
        n_estimators=LGB_N_ROUNDS, learning_rate=0.1, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    preds["LightGBM"] = m.predict(X_test)
    results["LightGBM_Base"] = compute_metrics(y_test, preds["LightGBM"], "LightGBM (Base)")
    results["LightGBM_Base"]["train_time"] = time.time() - t0
    joblib.dump(m, MODEL_DIR / "lgb_base.joblib")

    return results, preds


# ─────────────────────────────────────────────────────────────────────────
# 5. 1D-CNN ÖZNİTELİK ÇIKARICI  (Multi-Scale, KARARLI eğitim)
# ─────────────────────────────────────────────────────────────────────────
def build_cnn(input_dim: int, n_classes: int) -> keras.Model:
    """
    Çok ölçekli (multi-scale) 1D-CNN: kernel 3 / 5 / 7 paralel dalları.
    Kararlılık önlemleri:
      - Adam(clipnorm=1.0)  -> gradient clipping (gradyan patlamasını engeller)
      - BatchNormalization  -> aktivasyon dağılımını sabitler
      - Dropout(0.4)        -> aşırı öğrenmeyi azaltır
    """
    inp = keras.Input(shape=(input_dim, 1), name="input")

    # BatchNorm momentum=0.9: hareketli ortalama/varyans istatistikleri daha
    # hızlı oturur; eğitim ile çıkarım (inference) arasındaki uyumsuzluğu ve
    # buna bağlı erken-dönem doğrulama kaybı dalgalanmalarını önler.
    def conv_branch(x, k):
        c = layers.Conv1D(128, k, padding="same")(x)
        c = layers.BatchNormalization(momentum=0.9)(c)
        c = layers.Activation("relu")(c)
        return c

    x = layers.Concatenate(axis=-1)(
        [conv_branch(inp, 3), conv_branch(inp, 5), conv_branch(inp, 7)])  # (D, 384)
    x = layers.Conv1D(256, 3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    feat = layers.Dense(CNN_FEATURES, activation="relu", name="feature_vec")(x)
    out  = layers.Dense(n_classes, activation="softmax", name="output")(feat)

    model = keras.Model(inp, out, name="1D_CNN_MultiScale")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),  # [FIX-1]
        loss=keras.losses.SparseCategoricalCrossentropy(),                  # [FIX-3]
        metrics=["accuracy"])
    return model


def train_cnn(X_train, X_val, X_test, y_train, y_val, y_test,
              n_classes, sw_train):
    print("\n" + "=" * 64)
    print("ADIM 3: 1D-CNN Öznitelik Çıkarıcı Eğitimi  (kararlı)")
    print("=" * 64)

    X_tr = X_train[:, :, np.newaxis]
    X_vl = X_val[:,   :, np.newaxis]
    X_te = X_test[:,  :, np.newaxis]

    model = build_cnn(X_train.shape[1], n_classes)
    model.summary()

    cb = [
        # [FIX-2] ReduceLROnPlateau -> rapordaki anlatımla birebir aynı
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=CNN_PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(str(MODEL_DIR / "cnn_best.keras"),
                                  monitor="val_loss", save_best_only=True),
    ]

    t0 = time.time()
    history = model.fit(
        X_tr, y_train, validation_data=(X_vl, y_val),
        epochs=CNN_EPOCHS, batch_size=CNN_BATCH, callbacks=cb,
        sample_weight=sw_train, verbose=1)
    print(f"\n  1D-CNN eğitim süresi: {time.time() - t0:.1f}s")

    cnn_pred = np.argmax(model.predict(X_te, batch_size=1024), axis=1)
    compute_metrics(y_test, cnn_pred, "1D-CNN (doğrudan)")

    extractor = keras.Model(model.input, model.get_layer("feature_vec").output)
    extractor.save(str(MODEL_DIR / "cnn_feature_extractor.keras"))

    Xtr_cnn = extractor.predict(X_tr, batch_size=1024)
    Xvl_cnn = extractor.predict(X_vl, batch_size=1024)
    Xte_cnn = extractor.predict(X_te, batch_size=1024)

    cnn_scaler = StandardScaler()
    Xtr_cnn = cnn_scaler.fit_transform(Xtr_cnn).astype(np.float32)
    Xvl_cnn = cnn_scaler.transform(Xvl_cnn).astype(np.float32)
    Xte_cnn = cnn_scaler.transform(Xte_cnn).astype(np.float32)
    joblib.dump(cnn_scaler, MODEL_DIR / "cnn_feature_scaler.joblib")

    # Hibrit = [orijinal 46 || normalize CNN 128]
    Xtr_h = np.concatenate([X_train, Xtr_cnn], axis=1)
    Xvl_h = np.concatenate([X_val,   Xvl_cnn], axis=1)
    Xte_h = np.concatenate([X_test,  Xte_cnn], axis=1)
    print(f"  Hibrit öznitelik boyutu: {Xtr_h.shape[1]} "
          f"(orijinal {X_train.shape[1]} + CNN {Xtr_cnn.shape[1]})")

    return history, Xtr_h, Xvl_h, Xte_h


# ─────────────────────────────────────────────────────────────────────────
# 6. HİBRİT MODELLER
# ─────────────────────────────────────────────────────────────────────────
def train_hybrid_models(Xtr_h, Xvl_h, Xte_h, y_train, y_val, y_test,
                        n_classes, cw_dict, sw_train):
    print("\n" + "=" * 64)
    print("ADIM 4: Hibrit Modeller  [orijinal + CNN öznitelikleri]")
    print("=" * 64)
    results, preds = {}, {}
    sw_val = np.array([cw_dict.get(c, 1.0) for c in y_val], dtype=np.float32)

    # XGBoost Hibrit
    print("\n  [XGBoost Hibrit] eğitiliyor...")
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=XGB_N_ROUNDS, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, colsample_bylevel=0.8,
        tree_method="hist", objective="multi:softmax", num_class=n_classes,
        eval_metric="mlogloss", early_stopping_rounds=30,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    m.fit(Xtr_h, y_train, sample_weight=sw_train,
          eval_set=[(Xvl_h, y_val)], sample_weight_eval_set=[sw_val], verbose=False)
    preds["XGBoost"] = m.predict(Xte_h)
    results["XGBoost_Hybrid"] = compute_metrics(y_test, preds["XGBoost"], "XGBoost (Hibrit)")
    results["XGBoost_Hybrid"]["train_time"] = time.time() - t0
    joblib.dump(m, MODEL_DIR / "xgb_hybrid.joblib")

    # Random Forest Hibrit
    print("\n  [Random Forest Hibrit] eğitiliyor...")
    t0 = time.time()
    m = RandomForestClassifier(
        n_estimators=RF_N_TREES, max_features="sqrt", class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(Xtr_h, y_train)
    preds["Random Forest"] = m.predict(Xte_h)
    results["RF_Hybrid"] = compute_metrics(y_test, preds["Random Forest"], "Random Forest (Hibrit)")
    results["RF_Hybrid"]["train_time"] = time.time() - t0
    joblib.dump(m, MODEL_DIR / "rf_hybrid.joblib")

    # LightGBM Hibrit
    print("\n  [LightGBM Hibrit] eğitiliyor...")
    t0 = time.time()
    m = lgb.LGBMClassifier(
        n_estimators=LGB_N_ROUNDS, learning_rate=0.05, num_leaves=127,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        min_child_samples=30, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    m.fit(Xtr_h, y_train, eval_set=[(Xvl_h, y_val)],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    preds["LightGBM"] = m.predict(Xte_h)
    results["LightGBM_Hybrid"] = compute_metrics(y_test, preds["LightGBM"], "LightGBM (Hibrit)")
    results["LightGBM_Hybrid"]["train_time"] = time.time() - t0
    joblib.dump(m, MODEL_DIR / "lgb_hybrid.joblib")

    return results, preds


def majority_vote(pred_dict, y_test):
    """
    [FIX-4] Çoğunluk oylaması. NOT: bu yöntem en iyi tekil modeli geçmeyi
    GARANTİ ETMEZ; amacı üç farklı hata örüntüsünü dengeleyerek daha
    dengeli (özellikle Yanlış Pozitif açısından) bir karar üretmektir.
    """
    print("\n  [Ensemble] Majority Vote hesaplanıyor...")
    preds = np.stack(list(pred_dict.values()), axis=1)
    ens, _ = scipy_mode(preds, axis=1, keepdims=False)
    ens = ens.flatten().astype(int)
    return compute_metrics(y_test, ens, "Ensemble (Majority Vote)"), ens


# ─────────────────────────────────────────────────────────────────────────
# 7. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────────────────────────────────
PALETTE = {"XGBoost": "#2196F3", "Random Forest": "#4CAF50", "LightGBM": "#FF9800"}
MK = ["accuracy", "precision", "recall", "f1"]
ML = ["Accuracy", "Precision", "Recall", "F1-Score"]


def _labels(ax, bars, fs=8):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=fs, fontweight="bold")


def plot_individual(base_res, hyb_res):
    pairs = [("XGBoost", "XGBoost_Base", "XGBoost_Hybrid"),
             ("Random Forest", "RF_Base", "RF_Hybrid"),
             ("LightGBM", "LightGBM_Base", "LightGBM_Hybrid")]
    for name, bk, hk in pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(4); w = 0.35
        bv = [base_res[bk][k] for k in MK]; hv = [hyb_res[hk][k] for k in MK]
        b1 = ax.bar(x - w/2, bv, w, label="Yalın (Base)",
                    color=PALETTE[name], alpha=0.5, edgecolor="black", linewidth=0.7)
        b2 = ax.bar(x + w/2, hv, w, label="Hibrit (1D-CNN)",
                    color=PALETTE[name], alpha=0.95, edgecolor="black", linewidth=0.7)
        _labels(ax, b1); _labels(ax, b2)
        ax.set_xticks(x); ax.set_xticklabels(ML, fontsize=10)
        ax.set_ylim(0, 1.13); ax.set_ylabel("Skor")
        ax.set_title(f"{name}: Yalın vs Hibrit", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"compare_{name.replace(' ', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ compare_{name.replace(' ', '_')}.png")


def plot_combined_base(base_res):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (name, key) in zip(axes, [("XGBoost", "XGBoost_Base"),
                                       ("Random Forest", "RF_Base"),
                                       ("LightGBM", "LightGBM_Base")]):
        bars = ax.bar(ML, [base_res[key][k] for k in MK], color=PALETTE[name],
                      edgecolor="black", linewidth=0.8, alpha=0.85)
        _labels(ax, bars, 9)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.13); ax.set_ylabel("Skor")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle("Yalın (Base) Modeller Karşılaştırması\n"
                 "(CICIoT2023 %20 Alt Küme — Makro-Ort. Metrikler)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "compare_all_base.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ compare_all_base.png")


def plot_combined_hybrid(hyb_res, ens):
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
    trio = [("XGBoost", "XGBoost_Hybrid"), ("Random Forest", "RF_Hybrid"),
            ("LightGBM", "LightGBM_Hybrid")]
    for ax, (name, key) in zip(axes[:3], trio):
        bars = ax.bar(ML, [hyb_res[key][k] for k in MK], color=PALETTE[name],
                      edgecolor="black", linewidth=0.8, alpha=0.95)
        _labels(ax, bars, 9)
        ax.set_title(f"1D-CNN + {name}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.13); ax.set_ylabel("Skor")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    bars = axes[3].bar(ML, [ens[k] for k in MK], color="#9C27B0",
                       edgecolor="black", linewidth=0.8, alpha=0.95)
    _labels(axes[3], bars, 9)
    axes[3].set_title("Ensemble\n(Majority Vote)", fontsize=11, fontweight="bold")
    axes[3].set_ylim(0, 1.13)
    axes[3].yaxis.grid(True, linestyle="--", alpha=0.5); axes[3].set_axisbelow(True)
    fig.suptitle("Hibrit Modeller + Ensemble Karşılaştırması\n"
                 "(CICIoT2023 %20 Alt Küme — Makro-Ort. Metrikler)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "compare_all_hybrid.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ compare_all_hybrid.png")


def plot_summary(base_res, hyb_res):
    models = ["XGBoost", "Random Forest", "LightGBM"]
    bk = ["XGBoost_Base", "RF_Base", "LightGBM_Base"]
    hk = ["XGBoost_Hybrid", "RF_Hybrid", "LightGBM_Hybrid"]
    light = ["#90CAF9", "#A5D6A7", "#FFCC80"]; dark = ["#1565C0", "#2E7D32", "#E65100"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10)); axes = axes.flatten()
    for idx, (mk, ml) in enumerate(zip(MK, ML)):
        ax = axes[idx]; x = np.arange(3); w = 0.35
        b1 = ax.bar(x - w/2, [base_res[k][mk] for k in bk], w, label="Yalın",
                    color=light, edgecolor="black", linewidth=0.7)
        b2 = ax.bar(x + w/2, [hyb_res[k][mk] for k in hk], w, label="Hibrit",
                    color=dark, edgecolor="black", linewidth=0.7)
        _labels(ax, b1); _labels(ax, b2)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0, 1.13); ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    fig.suptitle("Yalın vs Hibrit — Tüm Metrikler\n(CICIoT2023 %20 Alt Küme | Makro-Ort.)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "base_vs_hybrid_summary.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ base_vs_hybrid_summary.png")


def plot_cnn_curves(history):
    h = history.history
    ep = range(1, len(h["loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(ep, h["loss"], "b-o", ms=3, label="Eğitim Kaybı")
    a1.plot(ep, h["val_loss"], "r--s", ms=3, label="Doğrulama Kaybı")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Çapraz Entropi")
    a1.set_title("1D-CNN: Kayıp Eğrileri", fontsize=11, fontweight="bold")
    a1.legend(); a1.yaxis.grid(True, linestyle="--", alpha=0.5)
    a2.plot(ep, h["accuracy"], "b-o", ms=3, label="Eğitim Doğruluğu")
    a2.plot(ep, h["val_accuracy"], "r--s", ms=3, label="Doğrulama Doğruluğu")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy")
    a2.set_title("1D-CNN: Doğruluk Eğrileri", fontsize=11, fontweight="bold")
    a2.legend(); a2.yaxis.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("1D-CNN Eğitim Süreci — Overfitting Analizi\n"
                 "(gradient clipping + ReduceLROnPlateau ile kararlı yakınsama)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cnn_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ cnn_training_curves.png")


def plot_confusion(y_true, y_pred, le, title, fname, top_n=15):
    uniq, cnt = np.unique(y_true, return_counts=True)
    top = uniq[np.argsort(cnt)[-top_n:]]
    mask = np.isin(y_true, top)
    yt, yp = y_true[mask], y_pred[mask]
    idx = np.sort(top)
    names = [n[:20] for n in le.inverse_transform(idx)]
    cm = confusion_matrix(yt, yp, labels=idx, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linewidths=0.3, ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Tahmin Edilen", fontsize=11); ax.set_ylabel("Gerçek", fontsize=11)
    ax.set_title(f"Normalize Karmaşıklık Matrisi — {title}\n(İlk {top_n} Sınıf)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(); print(f"  ✓ {fname}")


def plot_literature(base_res, hyb_res, ens):
    """
    Literatür karşılaştırması (CICIoT2023, 34 sınıf görevinde makro-F1).
    Dış değerler orijinal benchmark makalesinden (Neto vd., 2023) alınmıştır;
    o çalışmada en güçlü modeller (RF/DNN) zor görevlerde ~0,70 makro-F1
    seviyesinde kalmaktadır. Bu çalışmanın sonuçları aynı eksene konur.
    """
    labels = ["Neto vd. 2023\nRF (34 sınıf)*",
              "Bu çalışma\nXGBoost (Base)",
              "Bu çalışma\nEnsemble"]
    f1s = [0.70, base_res["XGBoost_Base"]["f1"], ens["f1"]]
    colors = ["#9E9E9E", "#2196F3", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, f1s, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    _labels(ax, bars, 10)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Makro-Ortalama F1-Skoru", fontsize=11)
    ax.set_title("Literatürle Karşılaştırma — CICIoT2023, 34 Sınıf Görevi\n"
                 "(*orijinal benchmark makalesinde raporlanan yaklaşık değer)",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "literatur_karsilastirma.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ literatur_karsilastirma.png")


# ─────────────────────────────────────────────────────────────────────────
# 8. ÖZET TABLO  (CSV + LaTeX otomatik üretimi -> kod/rapor uyumu)
# ─────────────────────────────────────────────────────────────────────────
def write_outputs(base_res, hyb_res, ens):
    rows = [
        ("XGBoost (Base)",          base_res["XGBoost_Base"]),
        ("XGBoost (Hibrit)",        hyb_res["XGBoost_Hybrid"]),
        ("Random Forest (Base)",    base_res["RF_Base"]),
        ("Random Forest (Hibrit)",  hyb_res["RF_Hybrid"]),
        ("LightGBM (Base)",         base_res["LightGBM_Base"]),
        ("LightGBM (Hibrit)",       hyb_res["LightGBM_Hybrid"]),
        ("Ensemble (Majority Vote)", ens),
    ]

    # konsol
    print("\n" + "=" * 78)
    print(f"  {'MODEL':30s} {'Acc':>9} {'Prec':>9} {'Rec':>9} {'F1':>9}")
    print("-" * 78)
    for name, m in rows:
        print(f"  {name:30s} {m['accuracy']:>9.4f} {m['precision']:>9.4f} "
              f"{m['recall']:>9.4f} {m['f1']:>9.4f}")
    print("=" * 78)

    # CSV
    pd.DataFrame([{**{"model": n}, **{k: m[k] for k in MK}} for n, m in rows]).to_csv(
        OUTPUT_DIR / "results_summary.csv", index=False)

    # LaTeX tablosu (rapora doğrudan kopyalanır -> sayılar %100 uyumlu)
    def tr(x): return f"{x:.3f}".replace(".", ",")
    lat = []
    lat.append(r"\begin{table}[!ht]")
    lat.append(r"    \centering")
    lat.append(r"    \caption{Tüm Modellerin Test Seti Üzerindeki Performans Özeti (Makro-Ortalama)}")
    lat.append(r"    \label{tab:tum_metrikler}")
    lat.append(r"    \renewcommand{\arraystretch}{1.35}")
    lat.append(r"    \begin{tabular}{|l|c|c|c|c|}")
    lat.append(r"        \hline")
    lat.append(r"        \textbf{Model} & \textbf{Doğruluk} & \textbf{Hassasiyet} & \textbf{Geri Çağırma} & \textbf{F1-Skoru} \\")
    lat.append(r"        \hline")
    for i, (name, m) in enumerate(rows):
        bold = name.startswith("Ensemble")
        nm = (r"\textbf{" + name + "}") if bold else name
        a, p, r_, f = (tr(m["accuracy"]), tr(m["precision"]), tr(m["recall"]), tr(m["f1"]))
        if bold:
            a, p, r_, f = (r"\textbf{"+a+"}", r"\textbf{"+p+"}", r"\textbf{"+r_+"}", r"\textbf{"+f+"}")
        lat.append(f"        {nm} & {a} & {p} & {r_} & {f} \\\\")
        if name.endswith("(Hibrit)") or bold:
            lat.append(r"        \hline")
    lat.append(r"    \end{tabular}")
    lat.append(r"\end{table}")
    (OUTPUT_DIR / "sonuc_tablosu.tex").write_text("\n".join(lat), encoding="utf-8")
    print("  ✓ results_summary.csv ve sonuc_tablosu.tex yazıldı.")


# ─────────────────────────────────────────────────────────────────────────
# 9. ANA AKIŞ
# ─────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("\n" + "#" * 64)
    print("  CICIoT2023 HİBRİT NIDS — v4 (kararlı eğitim, tutarlı)")
    print("  Hibrit = [Orijinal 46 öznitelik || Normalize CNN 128 öznitelik]")
    print("#" * 64)

    (X_train, X_val, X_test, y_train, y_val, y_test,
     le, n_classes, cw_dict, sw_train) = preprocess(DATA_DIR)

    base_res, _ = train_base_models(
        X_train, X_val, X_test, y_train, y_val, y_test, n_classes, cw_dict, sw_train)

    history, Xtr_h, Xvl_h, Xte_h = train_cnn(
        X_train, X_val, X_test, y_train, y_val, y_test, n_classes, sw_train)

    hyb_res, hyb_preds = train_hybrid_models(
        Xtr_h, Xvl_h, Xte_h, y_train, y_val, y_test, n_classes, cw_dict, sw_train)

    print("\n" + "=" * 64 + "\nADIM 5: Ensemble (Majority Vote)\n" + "=" * 64)
    ens, ens_pred = majority_vote(hyb_preds, y_test)

    print("\n" + "=" * 64 + "\nADIM 6: Görseller\n" + "=" * 64)
    plot_individual(base_res, hyb_res)
    plot_combined_base(base_res)
    plot_cnn_curves(history)
    plot_combined_hybrid(hyb_res, ens)
    plot_summary(base_res, hyb_res)
    plot_literature(base_res, hyb_res, ens)

    # Confusion matrisleri (rapordaki dosya adlarıyla birebir)
    # Base ve hibrit tahminlerini görsel için yeniden üret
    xgb_base = joblib.load(MODEL_DIR / "xgb_base.joblib")
    plot_confusion(y_test, xgb_base.predict(X_test), le,
                   "XGBoost Yalın", "confmat_xgb_base.png")
    plot_confusion(y_test, hyb_preds["XGBoost"], le,
                   "1D-CNN + XGBoost Hibrit", "confmat_hybrid_xgb.png")
    plot_confusion(y_test, ens_pred, le,
                   "Ensemble Majority Vote", "confmat_ensemble.png")

    write_outputs(base_res, hyb_res, ens)

    print(f"\n  Toplam süre: {(time.time() - t0)/60:.1f} dakika")
    print(f"  Görseller : {OUTPUT_DIR.resolve()}")
    print("  ✅ Tamamlandı.\n")


if __name__ == "__main__":
    main()