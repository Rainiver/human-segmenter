# Human Segmenter

## 📌 Description
This repository provides a simple human segmentation tool based on a pre-trained TensorFlow frozen graph (`.pb` model).  
It generates binary masks for human regions in images, which can be used for background removal, matting, or preprocessing pipelines.

---

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/Rainiver/human-segmenter.git
cd human-segmenter
pip install -r requirements.txt
```

> Note: The script uses `tensorflow.compat.v1` and disables v2 behavior. If you have TF v2 installed, the compatibility wrapper is used in the code.

---

## 🚀 Usage

### 1. Prepare files
- Place the pre-trained model file `matting_human.pb` (or your `.pb` model) in the project root.
- Put input images into the `./data/` folder (create it if missing).
- Ensure there is an (empty) `./mask/` folder, or the script will create output files in `./mask/` automatically.

### 2. Run segmentation
```bash
python segmenter.py
```

---

## 🖼️ Example

<div style="display: flex; align-items: flex-start;">

  <div style="margin-right: 20px; text-align: center;">
    <p><strong>Input</strong></p>
    <img src="example/input.jpg" alt="Input" width="300"/>
  </div>

  <div style="text-align: center;">
    <p><strong>Output</strong></p>
    <img src="example/output.jpg" alt="Output" width="300"/>
  </div>

</div>
