# LAB: CNN on a Dataset of Your Choice

โปรเจกต์นี้ใช้ **Convolutional Neural Network (CNN)** จำแนกภาพจากชุดข้อมูล **Fashion-MNIST** (ภาพเสื้อผ้าและรองเท้า 10 ประเภท ขนาด 28×28 พิกเซล สีเทา) แล้วเปรียบเทียบผลเมื่อเปลี่ยน **จำนวน epochs** และ **โครงสร้างของ CNN**

---

## 1. ไฟล์ในโปรเจกต์

```
ML 7/
├── cnn_lab.py        # โค้ดหลัก (รันแล้วได้ผลลัพธ์ทั้งหมด)
├── CNN_Lab.ipynb     # โค้ดเดียวกันในรูปแบบโน้ตบุ๊ก (เหมาะกับ Colab / Jupyter)
├── README.md         # ไฟล์นี้
├── venv/             # virtual environment (สร้างตอนติดตั้ง)
└── figures/          # กราฟ PNG ที่โค้ดสร้างให้หลังรันเสร็จ
```

## 2. สิ่งที่ต้องมี

| รายการ | รายละเอียด |
|---|---|
| Python | **3.10 – 3.13** (ในงานนี้ใช้ 3.12.10) ยังใช้ Python 3.14 ไม่ได้ เพราะ TensorFlow ยังไม่มีไฟล์ติดตั้งสำหรับเวอร์ชันนั้น |
| ไลบรารี | `tensorflow`, `scikit-learn`, `seaborn`, `pandas`, `matplotlib`, `jinja2`, `ipython` |
| อินเทอร์เน็ต | ใช้ตอนรันครั้งแรกเพื่อดาวน์โหลด Fashion-MNIST |

## 3. วิธีติดตั้งและรัน (Windows + VS Code)

```powershell
# สร้าง virtual environment ด้วย Python 3.12
& "C:\Users\<ชื่อผู้ใช้>\AppData\Local\Python\bin\python3.12.exe" -m venv venv

# เปิดใช้ environment
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# ติดตั้งไลบรารี
python -m pip install --upgrade pip
pip install tensorflow scikit-learn seaborn pandas matplotlib jinja2 ipython

# รัน
python cnn_lab.py
```

- เวลารันประมาณ 6–20 นาที (ขึ้นกับ CPU)
- กราฟทั้งหมดจะถูกบันทึกในโฟลเดอร์ `figures/`
- ถ้าต้องการตารางไล่สีและกราฟแสดงใต้โค้ด ให้เปิด `CNN_Lab.ipynb` ใน Google Colab แล้วกด **Runtime → Run all**

---

## 4. โค้ดทำงานอย่างไร (อธิบายทีละส่วน)

โค้ดแบ่งเป็นบล็อกที่คั่นด้วย `# %%` เรียงตามหัวข้อในโจทย์ ดังนี้

### ส่วนที่ 0: ตั้งค่าเริ่มต้น

```python
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)
```
กำหนดค่า seed เพื่อให้ผลการสุ่มใกล้เคียงเดิมทุกครั้งที่รัน (แต่ TensorFlow บน CPU อาจให้ผลต่างกันเล็กน้อยได้)

```python
PALETTE = ["#C0392B", "#2E6F8E", "#E67E3C"]
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({...})
```
กำหนดสีและธีมของกราฟให้สวยและสม่ำเสมอ โดยใช้ 3 สีแทนโมเดล A, B, C ตามลำดับ

```python
def show_fig(name): ...
```
ฟังก์ชันช่วยบันทึกกราฟเป็นไฟล์ `figures/NN_ชื่อ.png` แล้วค่อยแสดงหน้าต่างกราฟ (ถ้าเครื่องรองรับ) ถ้าเครื่องไม่รองรับหน้าต่างกราฟ จะบันทึกไฟล์อย่างเดียว

### ส่วนที่ 1: โหลดข้อมูลและแบ่ง train / test

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
```
Fashion-MNIST ถูกแบ่งมาให้แล้ว: **train 60,000 ภาพ** และ **test 10,000 ภาพ** โดย `x` คือภาพ (ขนาด 28×28) และ `y` คือเลข class 0–9

```python
except Exception:   # สำรอง
    ... load_digits() ... train_test_split(...)
```
ถ้าดาวน์โหลดไม่ได้ (ออฟไลน์) โค้ดจะสลับไปใช้ชุดข้อมูลตัวเลขเล็กๆ จาก scikit-learn แทน เพื่อให้โปรแกรมไม่ล้ม ถ้าบรรทัดแรกที่พิมพ์ออกมาเป็น `Dataset : Fashion-MNIST` แปลว่าใช้ข้อมูลหลักตามปกติ

จากนั้นแสดงภาพตัวอย่าง 1 ภาพต่อ 1 class (บันทึกเป็น `01_samples.png`)

### ส่วนที่ 2: Standardize ข้อมูล

```python
x_train = x_train.astype("float32")[..., None]   # (N, 28, 28) -> (N, 28, 28, 1)
mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std
```
- `[..., None]` เพิ่มมิติ channel (ภาพสีเทามี 1 channel) เพราะ Conv2D ต้องการรูปแบบ `(สูง, กว้าง, channel)`
- Standardize คือปรับค่าพิกเซลให้มี **ค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1** ช่วยให้โมเดลเรียนรู้เร็วและเสถียรขึ้น
- ใช้ mean/std จาก **train เท่านั้น** แล้วนำไปใช้กับ test ด้วย เพื่อไม่ให้ข้อมูล test รั่วเข้ามาในการเทรน (data leakage)
- เก็บภาพต้นฉบับไว้ใน `x_test_raw` เพื่อใช้แสดงภาพตอนทำนาย

### ส่วนที่ 3: สร้างโมเดล CNN

```python
CONFIGS = {
    "A: 1 Conv (32) + Dense 64":         dict(filters=[32],          dense=64),
    "B: 2 Conv (32,64) + Dense 128":     dict(filters=[32, 64],      dense=128),
    "C: 3 Conv (32,64,128) + Dense 256": dict(filters=[32, 64, 128], dense=256),
}
```
กำหนด 3 โครงสร้างที่ต่างกันที่ **จำนวนชั้น Conv** และ **จำนวน neurons ในชั้น Dense** (ตามที่โจทย์ให้เปรียบเทียบ)

ฟังก์ชัน `build_cnn` สร้างโมเดลตามลำดับนี้

| ชั้น | หน้าที่ |
|---|---|
| `Conv2D(f, 3, padding="same", activation="relu")` | ใช้ฟิลเตอร์ขนาด 3×3 จำนวน `f` ตัว สแกนหาลวดลายในภาพ (ขอบ, รูปทรง) ค่า `same` ทำให้ขนาดภาพไม่ลดลง |
| `MaxPooling2D(2)` | ย่อภาพลงครึ่งหนึ่งโดยเลือกค่าสูงสุดในแต่ละช่อง 2×2 ลดจำนวนข้อมูลและเก็บเฉพาะลวดลายเด่น |
| `Flatten()` | แปลงข้อมูลหลายมิติเป็นเวกเตอร์ยาวเส้นเดียว เพื่อส่งต่อให้ Dense |
| `Dense(dense, activation="relu")` | ชั้น neurons ปกติ รวมลวดลายที่ได้เพื่อตัดสินใจ |
| `Dropout(0.3)` | สุ่มปิด neurons 30% ระหว่างเทรน ช่วยลด overfitting |
| `Dense(10, activation="softmax")` | ชั้นผลลัพธ์ 10 ตัว (หนึ่งตัวต่อหนึ่ง class) ให้ค่าเป็นความน่าจะเป็นที่รวมกันได้ 1 |

(ชุด Conv + Pooling จะถูกวนซ้ำตามจำนวนใน `filters` เช่น B จะมี 2 ชุด)

```python
m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
- `adam`: วิธีปรับน้ำหนักโมเดลที่ใช้กันแพร่หลาย ปรับ learning rate ให้อัตโนมัติ
- `sparse_categorical_crossentropy`: ฟังก์ชัน loss สำหรับจำแนกหลาย class เมื่อ label เป็นเลขจำนวนเต็ม (ไม่ต้องแปลงเป็น one-hot)
- `accuracy`: สัดส่วนที่ทายถูก

บรรทัดสุดท้ายของส่วนนี้พิมพ์ `summary()` ของโมเดล B ให้ดูโครงสร้างและจำนวน parameters (โมเดล B มี 421,642 parameters)

### ส่วนที่ 4: เทรนด้วยจำนวน epochs ต่างกัน

```python
EPOCHS = 15
EPOCH_MARKS = [5, 10, 15]
```
เทรนแต่ละโมเดล 15 epochs (1 epoch = ให้โมเดลเห็นข้อมูล train ครบทุกภาพ 1 รอบ) แล้วเปรียบเทียบผลที่ epoch ที่ 5, 10 และ 15

```python
class TestAccuracy(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.acc.append(self.model.evaluate(self.x, self.y, verbose=0)[1])
```
Callback นี้วัด **test accuracy ทุกจบ epoch** จึงได้ผลที่หลาย epochs จากการเทรนรอบเดียวต่อโมเดล (เร็วกว่าเทรนแยกทีละจำนวน epochs)

```python
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128,
          validation_split=0.1, callbacks=[cb], verbose=0)
```
- `batch_size=128`: ปรับน้ำหนักทุกๆ 128 ภาพ
- `validation_split=0.1`: แบ่ง train 10% ไว้เป็น **validation set** ใช้ดูว่าโมเดลทำงานกับข้อมูลที่ไม่เคยเห็นเป็นอย่างไรระหว่างเทรน
- ผลทุกอย่างเก็บใน dictionary `results` ใช้ต่อในส่วนถัดไป

### ส่วนที่ 5: ประเมินและเปรียบเทียบผล

**5.1 ตารางสรุป accuracy** — สร้าง DataFrame ที่มีจำนวน parameters, test accuracy ที่ 5/10/15 epochs และ validation accuracy ที่ดีที่สุดของแต่ละโมเดล แล้วไล่สีให้ค่าสูงเด่นขึ้น (ไล่สีแสดงได้เฉพาะในโน้ตบุ๊ก ถ้ารันเป็น .py ตารางจะแสดงเป็นข้อความ `Styler object`)

**5.2 กราฟ train vs validation** (`02_training_curves.png`) — แผงกราฟ 2×3: แถวบนคือ accuracy แถวล่างคือ loss ส่วนแต่ละคอลัมน์คือโมเดล A, B, C

**5.3 กราฟเปรียบเทียบ** (`03_epoch_comparison.png`)
- ซ้าย: test accuracy ตามจำนวน epochs ของทั้ง 3 โมเดล มีจุดกำกับที่ epoch 5, 10, 15
- ขวา: แท่งกราฟ accuracy สุดท้ายของแต่ละโมเดล

### ส่วนที่ 6: ทำนายผลบน dataset

```python
best = max(results, key=lambda n: results[n]["test_acc"][-1])
probs = model.predict(x_test, verbose=0)
y_pred = probs.argmax(axis=1)
```
เลือกโมเดลที่ test accuracy สุดท้ายสูงสุด แล้วให้ทำนายภาพ test ทั้ง 10,000 ภาพ โดย `argmax` เลือก class ที่ความน่าจะเป็นสูงสุด

- **classification report**: แสดง precision, recall, f1-score ของแต่ละ class
- **confusion matrix** (`04_confusion_matrix.png`): ตารางว่า class จริงแต่ละอันถูกทายเป็น class ไหนบ้าง (ปรับเป็นสัดส่วนต่อแถว) แนวทแยงคือทายถูก
- **ภาพทำนาย** (`05_predictions.png`): สุ่ม 12 ภาพจาก test แสดงคำทำนาย คำตอบจริง และความมั่นใจ สีเขียวคือถูก สีแดงคือผิด

ส่วนสุดท้ายพิมพ์สรุป: โมเดลที่ดีที่สุด/แย่ที่สุด และ **ช่องว่าง train–validation accuracy** ของแต่ละโมเดล (ยิ่งกว้างยิ่งเสี่ยง overfitting)

---

## 5. ตรงกับข้อกำหนดของโจทย์อย่างไร

| ข้อกำหนด | อยู่ที่ไหน |
|---|---|
| Select and load a dataset | ส่วนที่ 1 |
| Split into training and testing sets | ส่วนที่ 1 |
| Standardize the input features | ส่วนที่ 2 |
| Build a CNN model | ส่วนที่ 3 |
| Train with different numbers of epochs | ส่วนที่ 4 (ดูผลที่ 5/10/15 epochs) |
| Compare different CNN configurations | ส่วนที่ 3–5 (โมเดล A, B, C) |
| Evaluate each model using accuracy | ส่วนที่ 5 |
| **Output:** accuracy per configuration | ตารางสรุปและกราฟแท่ง |
| **Output:** comparison across epochs | กราฟ `03_epoch_comparison.png` |
| **Output:** training/validation accuracy and loss | กราฟ `02_training_curves.png` |
| **Output:** predictions on the dataset | classification report, confusion matrix และภาพทำนาย |

---

## 6. ตัวอย่างผลที่ได้จากการรันจริง

(ตัวเลขอาจต่างกันเล็กน้อยในแต่ละครั้ง เพราะ TensorFlow ใช้การสุ่มและการคำนวณทศนิยมบน CPU)

| รายการ | ผล |
|---|---|
| โมเดลที่ดีที่สุด | B (2 Conv + Dense 128) ได้ test accuracy ประมาณ **92.19%** |
| โมเดลที่แย่ที่สุด | A (1 Conv + Dense 64) ได้ประมาณ **91.19%** |
| ช่องว่าง train−validation | A +1.60%, B +4.01%, C +5.89% (ยิ่งโมเดลใหญ่ ยิ่งเริ่ม overfit) |
| Class ที่ทายยากที่สุด | Shirt (f1 ≈ 0.77) มักสับสนกับเสื้อยืดและเสื้อโค้ต |
| Class ที่ทายง่ายที่สุด | Trouser, Bag, Sandal, Sneaker (f1 ≈ 0.97–0.99) |

**แนวทางเขียนวิเคราะห์ผล:** ชั้น Conv ที่เพิ่มขึ้นช่วยให้ accuracy สูงขึ้นจาก A ไป B แต่การเพิ่มต่อเป็น C ไม่ได้ช่วยเพิ่มอีก ในขณะที่ช่องว่าง train–validation กว้างขึ้น แสดงว่าโมเดลที่ซับซ้อนเกินไปเริ่มจำข้อมูลเทรนมากกว่าเรียนรู้ลวดลายทั่วไป

---

## 7. ข้อจำกัดที่ควรรู้

- การเปรียบเทียบ epochs ใช้การเทรนรอบเดียวแล้ววัดผลทุก epoch ไม่ได้เทรนแยกใหม่ทีละจำนวน epochs
- การเลือก "โมเดลที่ดีที่สุด" ใช้ test accuracy ตัดสิน ซึ่งถ้าจะให้เข้มงวดควรเลือกด้วย validation accuracy แล้วใช้ test วัดผลสุดท้ายครั้งเดียว
- รันแต่ละโมเดลครั้งเดียว ความต่างเล็กๆ (เช่น 1%) อาจเกิดจากการสุ่มก็ได้ ถ้าต้องการข้อสรุปที่แน่นขึ้น ควรรันซ้ำหลาย seed แล้วเฉลี่ย

## 8. แก้ปัญหาที่พบบ่อย

| ปัญหา | วิธีแก้ |
|---|---|
| `No module named 'tensorflow'` | ยังไม่ได้ติดตั้ง หรือกำลังใช้ Python 3.14 ให้ใช้ Python 3.12 ใน venv |
| `No module named 'IPython'` | `pip install ipython` |
| `running scripts is disabled` ตอน activate | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `FigureCanvasAgg is non-interactive` | ไม่ใช่ error แค่ไม่มีหน้าต่างกราฟ ดูกราฟในโฟลเดอร์ `figures/` แทน |
| ขึ้น `Dataset : Digits (fallback)` | โหลด Fashion-MNIST ไม่ได้ ตรวจอินเทอร์เน็ตแล้วรันใหม่ |
| ข้อความเตือน oneDNN | เป็นข้อมูลของ TensorFlow ไม่ต้องสนใจ |
