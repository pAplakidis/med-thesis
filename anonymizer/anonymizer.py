#!/usr/bin/env python3
import os
import sys
from tqdm import tqdm
from pydicom import dcmread

class Anonymizer:
  def __init__(self, root_dir):
    self.root_dir = root_dir

    self.dicom_files = []
    for dirpath, _, filenames in tqdm(os.walk(self.root_dir), desc="[*] Checking for dicom files ..."):
      for filename in filenames:
        if filename.lower().endswith('.dcm'):
          self.dicom_files.append(os.path.join(dirpath, filename))

    self.out_path = "./anonymized"
    os.makedirs(self.out_path, exist_ok=True)

  def run(self):
    print("Anonymizing DICOM files")
    for path in (t := tqdm(self.dicom_files)):
      t.set_description(path)
      self.anonymize_file(path)
    print("[+] Done")

  def anonymize_file(self, path):
    ds = dcmread(path)
    anon_ds = ds.copy()

    anon_ds.StudyDate = ""
    anon_ds.SeriesDate = ""
    anon_ds.AcquisitionDate = ""
    anon_ds.ContentDate = ""
    anon_ds.StudyTime = ""
    anon_ds.SeriesTime = ""
    anon_ds.AcquisitionTime = ""
    anon_ds.ContentTime = ""
    anon_ds.InstitutionName = ""
    anon_ds.PatientName = ""
    anon_ds.PatientID = ""
    anon_ds.PatientBirthDate = ""
    anon_ds.PatientSex = ""
    anon_ds.PatientAge = ""
    anon_ds.DeviceSerialNumber = ""
    anon_ds.IssuerOfPatientID = None
    anon_ds.IssuerOfPatientIDQualifiersSequence = None

    save_path = path.replace(self.root_dir, self.out_path + '/')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anon_ds.save_as(save_path)


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python anonymizer.py <path_to_dicom_directory>")
    sys.exit(1)

  anon = Anonymizer(sys.argv[1])
  anon.run()
