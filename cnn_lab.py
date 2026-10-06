# CNN on a Dataset of Your Choice (Fashion-MNIST)
# รันได้ทั้งใน Colab (วางทีละ block ที่คั่นด้วย # %%) หรือ VS Code / Jupyter แบบ cell

# %% [markdown]
# # 🧠 LAB: CNN on a Dataset of Your Choice
# **Dataset:** Fashion-MNIST (ภาพเสื้อผ้า/รองเท้า 10 ประเภท, ขนาด 28×28 สีเทา)
# 
# **วัตถุประสงค์:** ใช้ CNN จำแนกภาพ แล้วเปรียบเทียบผลเมื่อเปลี่ยนจำนวน epochs และโครงสร้างของ CNN
# 
# | ขั้นตอน | สิ่งที่ทำ |
# |---|---|
# | 1 | โหลดข้อมูล + split train/test |
# | 2 | Standardize ข้อมูล |
# | 3 | สร้าง CNN 3 แบบ (จำนวน conv layer และ neurons ต่างกัน) |
# | 4 | Train ด้วยจำนวน epochs ต่างกัน |
# | 5 | เปรียบเทียบ accuracy และดู training/validation curves |
# | 6 | ทำนายผลบน dataset |

# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

PALETTE = ["#C0392B", "#2E6F8E", "#E67E3C"]
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})
print("TensorFlow", tf.__version__)

# --- บันทึกกราฟเป็นไฟล์ PNG ลงโฟลเดอร์ figures (และแสดงหน้าต่างถ้าเครื่องรองรับ) ---
import os, matplotlib
os.makedirs("figures", exist_ok=True)
_fig_no = [0]
def show_fig(name):
    _fig_no[0] += 1
    path = f"figures/{_fig_no[0]:02d}_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[saved] {path}")
    if matplotlib.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% [markdown]
# ## 1️⃣ Select and load a dataset + Split train / test

# %%
try:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    DATASET = "Fashion-MNIST"
except Exception:                       # สำรอง กรณีโหลดข้อมูลไม่ได้ (ออฟไลน์)
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        d.images, d.target, test_size=0.2, stratify=d.target, random_state=SEED)
    class_names = [str(i) for i in range(10)]
    DATASET = "Digits (fallback)"

print(f"Dataset : {DATASET}")
print(f"Train   : {x_train.shape[0]:,} images | Test: {x_test.shape[0]:,} images | Image size: {x_train.shape[1:]}")

# ตัวอย่างภาพ 1 ภาพต่อ 1 class
fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for c, ax in enumerate(axes.ravel()):
    ax.imshow(x_train[np.where(y_train == c)[0][0]], cmap="gray")
    ax.set_title(class_names[c], fontsize=10); ax.axis("off")
fig.suptitle(f"{DATASET}: sample images", fontsize=15, fontweight="bold")
plt.tight_layout(); show_fig("samples")

# %% [markdown]
# ## 2️⃣ Standardize the input features

# %%
x_test_raw = x_test.astype("float32")          # เก็บภาพต้นฉบับไว้แสดงผลทำนาย

x_train = x_train.astype("float32")[..., None]  # เพิ่มมิติ channel -> (N, H, W, 1)
x_test  = x_test.astype("float32")[..., None]

mean, std = x_train.mean(), x_train.std()      # ใช้ค่าจาก train เท่านั้น (ป้องกัน data leakage)
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std
print(f"After standardization -> mean={x_train.mean():.3f}, std={x_train.std():.3f}")

# %% [markdown]
# ## 3️⃣ Build CNN models
# เปรียบเทียบ 3 โครงสร้าง: เพิ่มจำนวน **convolutional layers** และ **neurons** ใน Dense layer

# %%
CONFIGS = {
    "A: 1 Conv (32) + Dense 64":          dict(filters=[32],         dense=64),
    "B: 2 Conv (32,64) + Dense 128":      dict(filters=[32, 64],     dense=128),
    "C: 3 Conv (32,64,128) + Dense 256":  dict(filters=[32, 64, 128], dense=256),
}

def build_cnn(filters, dense, input_shape, n_classes=10):
    m = keras.Sequential([layers.Input(shape=input_shape)])
    for f in filters:
        m.add(layers.Conv2D(f, 3, padding="same", activation="relu"))
        m.add(layers.MaxPooling2D(2))
    m.add(layers.Flatten())
    m.add(layers.Dense(dense, activation="relu"))
    m.add(layers.Dropout(0.3))
    m.add(layers.Dense(n_classes, activation="softmax"))
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

build_cnn(**CONFIGS["B: 2 Conv (32,64) + Dense 128"], input_shape=x_train.shape[1:]).summary()

# %% [markdown]
# ## 4️⃣ Train with different numbers of epochs
# เทรนแต่ละ configuration 15 epochs และวัด **test accuracy ทุก epoch** เพื่อเปรียบเทียบผลที่ 5, 10 และ 15 epochs
# (ใช้ 10% ของ train เป็น validation set)

# %%
EPOCHS = 15
EPOCH_MARKS = [5, 10, 15]

class TestAccuracy(keras.callbacks.Callback):
    def __init__(self, x, y):
        super().__init__(); self.x, self.y, self.acc = x, y, []
    def on_epoch_end(self, epoch, logs=None):
        self.acc.append(self.model.evaluate(self.x, self.y, verbose=0)[1])

results = {}
for name, cfg in CONFIGS.items():
    print(f"▶ Training {name} ...")
    model = build_cnn(**cfg, input_shape=x_train.shape[1:])
    cb = TestAccuracy(x_test, y_test)
    h = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128,
                  validation_split=0.1, callbacks=[cb], verbose=0)
    results[name] = dict(model=model, history=h.history, test_acc=cb.acc, params=model.count_params())
    print(f"  ✓ params = {model.count_params():,} | final test accuracy = {cb.acc[-1]:.2%}")

# %% [markdown]
# ## 5️⃣ Evaluate and compare — Accuracy per configuration

# %%
rows = []
for name, r in results.items():
    row = {"Configuration": name, "Params": r["params"]}
    for e in EPOCH_MARKS:
        row[f"Test acc @ {e} ep"] = r["test_acc"][e - 1]
    row["Best val acc"] = max(r["history"]["val_accuracy"])
    rows.append(row)

summary = pd.DataFrame(rows).set_index("Configuration")
acc_cols = [c for c in summary.columns if "acc" in c]
display(summary.style
        .format({"Params": "{:,}", **{c: "{:.2%}" for c in acc_cols}})
        .background_gradient(cmap="YlOrRd", subset=acc_cols)
        .set_caption("Accuracy summary (higher = better)"))

# %% [markdown]
# ### 📈 Training and validation accuracy / loss

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
ep = range(1, EPOCHS + 1)
for j, (name, r) in enumerate(results.items()):
    h = r["history"]
    axes[0, j].plot(ep, h["accuracy"],     color=PALETTE[0], lw=2, label="Train")
    axes[0, j].plot(ep, h["val_accuracy"], color=PALETTE[1], lw=2, label="Validation")
    axes[1, j].plot(ep, h["loss"],         color=PALETTE[0], lw=2)
    axes[1, j].plot(ep, h["val_loss"],     color=PALETTE[1], lw=2)
    axes[0, j].set_title(name.replace(" + ", "\n+ "), fontsize=11)
    axes[1, j].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Accuracy"); axes[1, 0].set_ylabel("Loss")
axes[0, 0].legend()
fig.suptitle("Training vs Validation: Accuracy & Loss", fontsize=15, fontweight="bold")
plt.tight_layout(); show_fig("training_curves")

# %% [markdown]
# ### 📊 Comparison across epochs and configurations

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for (name, r), col in zip(results.items(), PALETTE):
    ax1.plot(ep, r["test_acc"], color=col, lw=2.2, label=name)
    ax1.scatter(EPOCH_MARKS, [r["test_acc"][e - 1] for e in EPOCH_MARKS], color=col, s=60, zorder=3)
for e in EPOCH_MARKS:
    ax1.axvline(e, color="gray", ls=":", lw=1)
ax1.set_title("Test accuracy vs number of epochs"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Test accuracy")
ax1.legend(fontsize=9, loc="lower right")

finals = [r["test_acc"][-1] for r in results.values()]
labels = [n.split(":")[0] for n in results]
bars = ax2.bar(labels, finals, color=PALETTE, width=0.55)
ax2.set_ylim(min(finals) - 0.03, min(1.0, max(finals) + 0.02))
for b, v in zip(bars, finals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.2%}", ha="center", fontweight="bold")
ax2.set_title(f"Final test accuracy ({EPOCHS} epochs)"); ax2.set_xlabel("CNN configuration")
plt.tight_layout(); show_fig("epoch_comparison")

# %% [markdown]
# ## 6️⃣ Predictions on the selected dataset (best model)

# %%
best = max(results, key=lambda n: results[n]["test_acc"][-1])
model = results[best]["model"]
probs = model.predict(x_test, verbose=0)
y_pred = probs.argmax(axis=1)
print(f"🏆 Best configuration: {best}  ->  test accuracy = {(y_pred == y_test).mean():.2%}\n")
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize="true")
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds", xticklabels=class_names,
            yticklabels=class_names, cbar=False, linewidths=0.5)
plt.title("Confusion matrix (normalized)"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(rotation=45, ha="right"); plt.tight_layout(); show_fig("confusion_matrix")

# %%
# ตัวอย่างผลทำนาย: เขียว = ถูก, แดง = ผิด
rng = np.random.default_rng(SEED)
idx = rng.choice(len(x_test), 12, replace=False)

fig, axes = plt.subplots(2, 6, figsize=(14, 8))
for ax, i in zip(axes.ravel(), idx):
    ok = y_pred[i] == y_test[i]
    ax.imshow(x_test_raw[i], cmap="gray"); ax.axis("off")
    ax.set_title(f"Pred: {class_names[y_pred[i]]}\nTrue: {class_names[y_test[i]]}\n({probs[i].max():.0%})",
                 fontsize=9, color="#1E8449" if ok else "#C0392B")
fig.suptitle(f"Predictions by model {best.split(':')[0]}", fontsize=15, fontweight="bold")
plt.tight_layout(); show_fig("predictions")

# %% [markdown]
# ## ✅ Summary

# %%
worst = min(results, key=lambda n: results[n]["test_acc"][-1])
print(f"• Best configuration  : {best} ({results[best]['test_acc'][-1]:.2%})")
print(f"• Weakest configuration: {worst} ({results[worst]['test_acc'][-1]:.2%})")
for name, r in results.items():
    gap = r['history']['accuracy'][-1] - r['history']['val_accuracy'][-1]
    print(f"• {name.split(':')[0]}: train-val accuracy gap = {gap:+.2%}  (gap กว้าง = เริ่ม overfit)")

# %% [markdown]
# ### ✍️ วิเคราะห์ผล (เติมเองหลังรัน)
# - Epochs มาก/น้อยส่งผลต่อ accuracy อย่างไร? เริ่มเห็น overfitting ที่ epoch ใด?
# - การเพิ่ม convolutional layer / neurons ช่วยเพิ่ม accuracy หรือไม่ และแลกกับจำนวน parameters อย่างไร?
# - Class ไหนที่โมเดลทำนายผิดบ่อยที่สุด (ดูจาก confusion matrix) เพราะอะไร?