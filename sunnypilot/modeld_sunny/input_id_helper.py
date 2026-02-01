import numpy as np
from dataclasses import dataclass
from openpilot.common.params import Params


@dataclass
class AlpamayoDesire:
  DRIVE_SAFELY = 0
  TURN_LEFT = 2
  TURN_RIGHT = 1
  DRIVE_FAST = 3
  STOP = 4


class InputIDHelper:
  def __init__(self):
    self.current_ids = np.zeros((1, 16), dtype=np.int64)
    self.desire = AlpamayoDesire.DRIVE_SAFELY
    self.params = Params()
    self.drive_fast = False
    self.msg_count = -1

  def update_params(self):
    if self.msg_count % 60 == 0:
      self.drive_fast = self.params.get_bool("AlpamayoDriveFast")
    self.msg_count += 1

  def update(self, sm):
    self.update_params()
    if sm is None:
      return self.current_ids

    left_blinker = False
    right_blinker = False

    if sm.seen['carState']:
      left_blinker = sm['carState'].leftBlinker
      right_blinker = sm['carState'].rightBlinker

    # Priority: STOP (TODO) > Turn > Drive Fast > Drive Safely
    new_desire = AlpamayoDesire.DRIVE_SAFELY
    if left_blinker:
      new_desire = AlpamayoDesire.TURN_LEFT
    elif right_blinker:
      new_desire = AlpamayoDesire.TURN_RIGHT
    elif self.drive_fast:
      new_desire = AlpamayoDesire.DRIVE_FAST

    self.desire = new_desire
    self.current_ids.fill(self.desire)
    return self.current_ids
