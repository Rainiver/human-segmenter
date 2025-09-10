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

---

## 🚀 Usage  

### 1. Run segmentation on your own data  
Put your test images into `./data/` and run:  

```bash
python segmenter.py

---

## 🖼️ Example  

Input image (`data/sample.jpg`):  

![input](docs/input_example.jpg)  

Generated mask (`mask/sample_mask.jpg`):  

![mask](docs/output_example.jpg)  



