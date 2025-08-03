# Med-Thesis

Code for a thesis on ML for MRI and CT scans

### Setup

Install dependencies

```
pip install -r requirements.txt
```

### Run

Anonymizer (removes sensitive info from dicom files)

```
python anonymizer.py <path_to_data>
```

Image extractor (extracts pixel arrays from dicom files to PNGs)

```
python image_extractor.py <path_to_data>
```
