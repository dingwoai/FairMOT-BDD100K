from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer
from .det import DetTrainer


train_factory = {
  'mot': MotTrainer,
  'det': DetTrainer,
}
