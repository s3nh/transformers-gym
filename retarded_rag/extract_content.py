import argparse
import os 
import pickle 
import pathlib 
from pathlib import Path
from extract_office_content import ExtractOfficeContent

def get_args():
  ...

def main():
  DATA_PATH: str = ''
  OUTFILE_PATH: str = ''
  file_list = list(Path(DATA_PATH).iterdir())
  files = [el for el in file_list if not str(el).endswith(('.md', '.pdf', '.ipynb_checkpoints', '.doc'))]
  n_files = len(files)
  assert n_files > 0, 'Data path is empty'
  print(f"Len of processing files {n_files}")
  extractor = ExtracOfficeContent()

  for file_path in files:
      try:
          filename = file_path.stem
          res = extractor(file_path)
          with open(f"{OUTFILE_PATH}/{filename}.pickle", "wb") as outfile:
              pickle.dump(res, outfile)
      except:
          print(f"Omitted for file {filename}")
          continue
        

