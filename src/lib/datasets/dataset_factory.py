from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.bdd100k import JointDatasetBDD, DetDataset


def get_dataset(dataset, task):
  if task == 'mot':
  #   return JointDataset
  # elif task == 'mot_bdd':
    return JointDatasetBDD
  elif task == 'det':
    return DetDataset
  else:
    return None
