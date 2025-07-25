# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python XFP example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import app
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import fictitious_play
import pyspiel

import stopping_game
def main(_):
  iterations = 100
  print_freq = 1
  game = pyspiel.load_game("python_stopping")
  xfp_solver = fictitious_play.XFPSolver(game)
  for i in range(iterations):
    xfp_solver.iteration()
    conv = exploitability.exploitability(game, xfp_solver.average_policy())
    if i % print_freq == 0:
      print("Iteration: {} Conv: {}".format(i, conv))
      sys.stdout.flush()


if __name__ == "__main__":
  app.run(main)