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
pip install --upgrade pip
pip install -r requirements
```

> **Note:** the packages include `adversarial-robustness-toolbox` (ART): a library from IBM for creating and evaluating adversarial attacks and defenses.
We focus on this library.

## Recommended approach: using venv

### Create and activate a new virtual environment
We will create an environment named `adversarial`:

```bash
python -m venv adversarial
source ./adversarial/bin/activate

### 3️⃣ Install packages

Install required packages from the requirements file

```bash
pip install --upgrade pip
pip install -r requirements

## Check your installation

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
import matplotlib
```

---

## Final notes

* Be careful to activate your environment before running code or notebooks.
* Use `conda deactivate` (for conda) or `deactivate` (for venv) to exit the environment.
* To create a Jupyter kernel for your environment (so it appears as an option in notebooks):

```bash
python -m ipykernel install --user --name adversarial_tf --display-name "Python (adversarial)"
```

---

### 📄 References

* [Adversarial Robustness Toolbox documentation](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)
* [TensorFlow installation guide](https://www.tensorflow.org/install)
* [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Enjoy the Summer School !
