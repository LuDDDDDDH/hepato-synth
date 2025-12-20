# Hepato-Synth: Physics-Informed Synthesis and trustworthy Diagnosis for Hepatocellular MRI

**Hepato-Synth** (Hepatocellular Synthesizer) is a research project dedicated to building a next-generation intelligent imaging system for liver MRI. Our goal is to break the "spatiotemporal dilemma" in liver functional imaging by developing a physics-informed, disentangled generative AI framework.

This project aims to synthesize high-value functional imaging (like Hepatobiliary Phase) from rapid, low-cost scans (like standard dynamic contrast-enhanced MRI), and build a trustworthy AI-powered diagnostic system upon it.

This work is proposed as a Ph.D. research plan for the **Intelligent Medical Engineering** program at Tianjin University, in collaboration with **United Imaging Intelligence**.

---

## 🚩 Project Goals

The core mission of Hepato-Synth is to address three critical clinical pain points in liver MRI:
1.  **Accelerate Imaging**: Reduce the 20+ minute scan time for Gd-EOB-DTPA to under 5 minutes by generating the Hepatobiliary Phase (HBP) from early dynamic phases.
2.  **Virtualize Imaging**: Synthesize high-value virtual HBP images from low-cost, standard Gd-DTPA scans, dramatically reducing economic burden and increasing diagnostic accessibility.
3.  **Enable Trustworthy Diagnosis**: Build a generation-diagnosis integrated AI system that improves diagnostic accuracy for focal liver lesions and resolves the "trust crisis" of black-box AI through uncertainty quantification.

---

## 🔬 Core Scientific Hypothesis: From "Perfusion Fingerprint" to "Functional Mapping"

The central challenge is to infer intracellular function (OATP8 transporter activity) from extracellular perfusion data. Our core hypothesis is **"Structure-Function Covariation"**: the differentiation grade of a tumor is the common determinant that links its microscopic structure and physiological function.

*   **Functional Dimension**: Differentiation grade determines OATP8 expression (e.g., lost in poorly differentiated HCC).
*   **Structural Dimension**: Differentiation grade also dictates the tumor's microvascular architecture. This architecture, though microscopic, manifests at the voxel level as a measurable, high-dimensional **"Perfusion Kinetic Fingerprint"**.

**Our project leverages deep learning to learn the non-linear mapping from this macroscopic "Perfusion Fingerprint" to the microscopic cellular function.**

---

## 🏗️ Technical Architecture

The project follows a closed-loop R&D path: **Standardized Data Cohort -> Physics-Informed Feature Engineering -> Disentangled Generative Modeling -> Clinical Efficacy Validation**.

![Technical Architecture Diagram](https://i.imgur.com/your-architecture-diagram.png) 
*(Note: This is a placeholder for a flowchart you should create. A diagram is worth a thousand words!)*

The codebase is organized into four core modules, strictly corresponding to our research plan:

1.  **`data_preprocessing/`**: Implements an industrial-grade pipeline for cohort building and data standardization (Registration, Normalization).
2.  **`perfusion_modeling/`**: A GPU-accelerated engine for solving the DITC pharmacokinetic model and extracting explicit physical parameters ($K^{trans}, v_e, k_{hep}$) as the "physical compass" for the AI.
3.  **`generative_models/`**: The core generative algorithm layer, featuring a dual-track architecture:
    *   **Track A (Physics-Informed Swin UNETR)** for same-modality acceleration.
    *   **Track B (Disentangled Swin-DRIT++)** for cross-modality virtual imaging.
4.  **`diagnostic_system/``**: A downstream diagnostic system that fuses multi-modal inputs (raw MRI + physical maps + virtual HBP) for trustworthy, multi-task (segmentation & classification) analysis.

---

## 📦 Installation & Dependencies

To set up the environment for this project, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/hepato-synth.git
    cd hepato-synth
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For GPU acceleration, ensure you have a compatible version of PyTorch with CUDA support installed.*

---

## 🚀 Usage

This repository is currently structured to demonstrate the project's architectural logic. The core modules are in place, and the execution scripts are designed to work with placeholder/synthetic data for pipeline testing.

### Training a Model

To run a training process (e.g., for Study 1 - Accelerated Imaging):

```bash
python main.py --mode train --config configs/study_1_acceleration.yaml
--mode train: Specifies the operation mode.
--config: Points to the configuration file that contains all hyperparameters and paths for the specific study.
Running Inference
To perform inference on a new case:
code
Bash
python main.py --mode inference --config configs/study_2_virtual.yaml --input_path /path/to/new/patient/data --output_path /path/to/save/results
⚠️ Important Note:
Due to patient privacy regulations, this repository does not contain clinical data. The current implementation uses synthetic data generators to test the end-to-end pipeline. The physics module (perfusion_modeling) is implemented based on the standard DITC model and requires real 4D MRI data to yield meaningful results.
📈 Future Work & Contributions
This project is a long-term research endeavor. The planned development roadmap includes:

Phase 1: Complete data collection and standardization for Cohorts A & B.

Phase 2: Validate the GPU-accelerated DITC model on real patient data.

Phase 3: Release pretrained models for Study 1 (Accelerated Imaging).

Phase 4: Publish initial results for Study 2 (Virtual Imaging) and release models.

Phase 5: Complete the full clinical validation of the integrated diagnostic system (Study 3).
We welcome collaboration from researchers in medical imaging, deep learning, and clinical medicine. Please open an issue or contact us directly if you are interested in contributing.
Citing this Work
If you find this project useful in your research, please consider citing our future publications:
code
Bibtex
@article{YourLastName_HepatoSynth_2026,
  title={Hepato-Synth: Physics-Informed Synthesis and Trustworthy Diagnosis for Hepatocellular MRI},
  author={Your Name, et al.},
  journal={TBD},
  year={2026}
}
📧 Contact
For any questions or inquiries, please contact [Your Name] at [Your-Email-Address].