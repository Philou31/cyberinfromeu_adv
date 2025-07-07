# cyberinfromeu_adv
Adversarial machine learning session: hands-on

# Adversarial Machine Learning — Python Environment Setup

Welcome to this hands-on session on adversarial machine learning!

## Installation
Here are instructions to install the Python libraries you will need for the session.

The best is to start from a fresh environment. You can choose between:

* **Using conda** (recommended if you already have Anaconda)
* **Using venv** (built-in Python virtual environments)

---

## Recommended approach: using conda

### Create and activate a new conda environment
We will create an environment named `adversarial` and set Python version 3.9:

```bash
conda create -n adversarial python=3.9
conda activate adversarial_tf
```

### 3️⃣ Install python

```bash
conda install python=3.9 ipykernel -y
```

### 3️⃣ Install packages

Install required packages from the requirements file

```bash
conda install --file requirements.txt -y
```
If you have an NVIDIA GPU with CUDA support installed (check with nvidia-smi). Then use the requirements_gpu.txt file instead. Note that there may be conflicts due to your version of CUDA. Installing this can a bit of a pain...

### 4️⃣ Install additional required libraries

```bash
conda install numpy opencv adversarial-robustness-toolbox matplotlib -y
```

> 💡 **Note:** `adversarial-robustness-toolbox` (ART) is a library from IBM for creating and evaluating adversarial attacks and defenses.

### 5️⃣ Install extra packages via pip

Some packages are only available on PyPI:

```bash
pip install tf_agents[reverb]
```

---

## 💡 Alternative approach: using venv

If you prefer not to use conda, you can use Python’s built-in `venv` module.

### 1️⃣ Create a new virtual environment

```bash
python3.9 -m venv adversarial_tf
```

> ✅ Replace `python3.9` with the path to your Python 3.9 interpreter if needed.

### 2️⃣ Activate the environment

* **On Linux or macOS:**

```bash
source adversarial_tf/bin/activate
```

* **On Windows (PowerShell):**

```powershell
.\adversarial_tf\Scripts\activate
```

### 3️⃣ Upgrade pip

```bash
pip install --upgrade pip
```

### 4️⃣ Install packages

```bash
pip install tensorflow==2.8.2 keras==2.8.0 ipykernel
pip install numpy opencv-python adversarial-robustness-toolbox matplotlib tf_agents[reverb]
```

---

## ⚡ Check your installation

Once installed, verify that you can import the main libraries:

```bash
python
```

Then in the Python prompt:

```python
import tensorflow as tf
import keras
import art
import numpy
import cv2
import matplotlib
import tf_agents
print("All packages imported successfully! ✅")
```

---

## 🎯 Final notes

* Always **activate your environment** before running code or notebooks.
* Use `conda deactivate` (for conda) or `deactivate` (for venv) to exit the environment.
* To create a Jupyter kernel for your environment (so it appears as an option in notebooks):

```bash
python -m ipykernel install --user --name adversarial_tf --display-name "Python (adversarial_tf)"
```

---

### 📄 References

* [Adversarial Robustness Toolbox documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)
* [TensorFlow installation guide](https://www.tensorflow.org/install)
* [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

---

Happy hacking and stay robust! 💻🛡️
