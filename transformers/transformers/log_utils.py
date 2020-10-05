
from datetime import datetime
from collections import defaultdict
import threading
import time
import logging
import os

from tensorboardX import SummaryWriter
from pandas import DataFrame
from collections import defaultdict


class AverageMeterSet:
  def __init__(self):
    self.meters = {}

  def __getitem__(self, key):
    return self.meters[key]

  def update_dict(self, name_val_dict, n=1):
    for name, val in name_val_dict.items():
      self.update(name, val, n)

  def update(self, name, value, n=1):
    if not name in self.meters:
      self.meters[name] = AverageMeter()
    self.meters[name].update(value, n)

  def reset(self):
    for meter in self.meters.values():
      meter.reset()

  def values(self, postfix=''):
    return {name + postfix: meter.val for name, meter in self.meters.items()}

  def averages(self, postfix='/avg'):
    return {name + postfix: meter.avg for name, meter in self.meters.items()}

  def sums(self, postfix='/sum'):
    return {name + postfix: meter.sum for name, meter in self.meters.items()}

  def counts(self, postfix='/count'):
    return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __format__(self, format):
    return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class TrainLog:
  """Saves training logs in Pandas msgpacks"""
  INCREMENTAL_UPDATE_TIME = 300

  def __init__(self, directory, name, init_tb=False):
    self.name = name
    self.log_file_path = "{}/{}.msgpack".format(directory, name)
    self._log = defaultdict(dict)
    self._log_lock = threading.RLock()
    self._last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME
    self._summary_writer = None
    self._meter_set = AverageMeterSet()
    if init_tb:
      self._summary_writer = SummaryWriter(directory, comment=name)
    self._tb_fields = None

  def set_tb_fields(self, tbf):
    self._tb_fields = set(tbf)

  def is_tb_key(self, key):
    if not self._tb_fields:
      return True
    for name in self._tb_fields:
      if name in key:
        return True
    return False

  def get_summary_writer(self):
    return self._summary_writer

  def set_summary_writer(self, tbw):
    self._summary_writer = tbw

  def record_single(self, step, column, value):
    self._record(step, {column: value})

  def record(self, step, col_val_dict, tbstep=None):
    if not tbstep:
      tbstep = step
    if self._summary_writer:
      for name, value in col_val_dict.items():
        if self.is_tb_key(name):
          self._summary_writer.add_scalar("{}/{}".format(self.name, name), value, tbstep)
    self._record(step, col_val_dict)

  def save(self):
    df = self._as_dataframe()
    df.to_msgpack(self.log_file_path, compress='zlib')

  def _record(self, step, col_val_dict):
    with self._log_lock:
      self._log[step].update(col_val_dict)
      if time.time() - self._last_update_time >= self.INCREMENTAL_UPDATE_TIME:
        self._last_update_time = time.time()
        # self.save()
        if self._summary_writer:
          self._summary_writer.file_writer.flush()

  def _as_dataframe(self):
    with self._log_lock:
      return DataFrame.from_dict(self._log, orient='index')
