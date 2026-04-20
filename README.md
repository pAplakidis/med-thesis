# Med-Thesis

Code for a thesis on ML for MRI and CT scans

### Setup

Install dependencies

```bash
pip install -r requirements.txt
```

### Run

Anonymizer (removes sensitive info from dicom files)

```bash
python anonymizer.py <path_to_data>
```

Image extractor (extracts pixel arrays from dicom files to PNGs)

```bash
python image_extractor.py <path_to_data>
```

Arena is a viewer for the image, ground truth and model results

```bash
streamlit run arena.py
```

Reporter tool is a detailed viewer of the trained and evaluated models for in-depth comparisons

```bash
streamlit run reporter.py
```

### Training workflow

- Configure model
  Each model should be configured by hand inside it's python file in models/

- Configure training
  copy your configuration to configs/main.json (keep a copy for legacy)

- Train

```bash
MODEL_PATH=checkpoints/<experiment_folder>/<model_name> python3 train.py
```

If you want to resume training use add after MODEL_PATH:

```bash
CHECKPOINT=<path_to_checkpoint>
```

This trains the model, saves its checkpoint alongside other training info, converts it to ONNX and creates a full evaluation report right after the full training
