# find files with a given tag in a given path

#import stuff
import os

def find_files(path, tag):
  file_paths = []
  for root, dirs, files in os.walk(path):
    for file in files:
      if tag in file:
        path = os.path.join(root,file)
        file_paths.append(path)
  return file_paths 
