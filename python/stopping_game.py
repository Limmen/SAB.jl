import enum
import numpy as np
import pyspiel
from scipy.stats import betabinom
import math


class Action(enum.IntEnum):
    CONTINUE = 0
    STOP = 1


P_A = 0.2
N = 3
T = 3
num_obs = 5
INTRUSION_RV = betabinom(n=num_obs, a=1, b=0.7)
NO_INTRUSION_RV = betabinom(n=num_obs, a=0.7, b=3)
INTRUSION_DIST = []
NO_INTRUSION_DIST = []
for i in range(num_obs + 1):
    INTRUSION_DIST.append(INTRUSION_RV.pmf(i))
    NO_INTRUSION_DIST.append(NO_INTRUSION_RV.pmf(i))
_NUM_PLAYERS = 2
_GAME_TYPE = pyspiel.GameType(
    short_name="python_stopping",
    long_name="Python Stopping",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(INTRUSION_DIST),
    num_players=_NUM_PLAYERS,
    min_utility=-math.pow(N, 5/4),
    max_utility=math.pow(N, 5/4),
    utility_sum=0.0,
    max_game_length=T)


class StoppingGame(pyspiel.Game):

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        return StoppingGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return StoppingGameObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), params)


class StoppingGameState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.s = 0
        self.o = 0
        self.t = 0
        self.T = T
        self._game_over = False
        self._next_player = 0
        self.chance = False
        self._chance_type = 0
        self._rewards = np.zeros(_NUM_PLAYERS)
        self._step_rewards = np.zeros(_NUM_PLAYERS)
        self.prev_action_0 = 0
        self.prev_action_1 = 0

    def __str__(self):
        return f"s: {self.s}, player: {self.current_player()}, chance: {self.chance}, chance type: {self._chance_type}"

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        elif self.chance:
            return pyspiel.PlayerId.CHANCE
        else:
            return self._next_player

    def _legal_actions(self, player):
        assert player >= 0
        return [Action.CONTINUE, Action.STOP]

    def chance_outcomes(self):
        assert self.is_chance_node()
        if self._chance_type == 1:
            if self.s < N:
                return [(self.s+1, P_A), (self.s, 1 - P_A)]
            else:
                return [(N, 1.0)]
        else:
            if self.s == 0:
                return [(i, INTRUSION_DIST[i]) for i in range(len(INTRUSION_DIST))]
            else:
                return [(i, NO_INTRUSION_DIST[i]) for i in range(len(NO_INTRUSION_DIST))]

    def _apply_action(self, action):
        if self.is_chance_node():
            if self._chance_type == 1:
                self.s = action
                self._chance_type = 0
            else:
                self.o = action
                self.chance = False
                sign = 0
                if self.s > 0:
                    sign = 1
                cost = math.pow(self.s, 5/4)*(1-self.prev_action_0) + action-2*(self.prev_action_0)*sign
                self._step_rewards[0] = -cost
                self._step_rewards[1] = cost
                self._rewards[0] += self._step_rewards[0]
                self._rewards[1] += self._step_rewards[1]

        else:
            if self._next_player == 0:
                if action == Action.STOP:
                    self.s = 0
                self.prev_action_0 = action
                self._next_player = 1
            elif self._next_player == 1:
                self.prev_action_1 = action
                if action == Action.STOP:
                    self.s = min(1, self.s + 1)
                self.t += 1
                if self.t >= self.T:
                    self._game_over = True
                self._next_player = 0
                self.chance = True
                if action == Action.STOP and self.s == 0:
                    self._chance_type = 1
                else:
                    self._chance_type = 0


    def _action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self._chance_type == 1:
                return f"StateTransitionTo({action})"
            return f"Observation({action})"
        return "Stop" if action == Action.STOP else "Continue"

    def is_terminal(self):
        return self._game_over

    def rewards(self):
        return self._step_rewards

    def returns(self):
        return self._rewards


class StoppingGameObserver:

    def __init__(self, iig_obs_type, params):
        self.iig_obs_type = iig_obs_type
        self.tensor = None
        self.dict = {"observation": np.array([0])}

    def set_from(self, state, player):
        self.tensor = None
        self.dict = {"observation": np.array([state.o])}

    def string_from(self, state, player):
        if self.iig_obs_type.public_info:
            h = state.history()
            return ",".join(list(map(lambda x: str(x), h)))
        else:
            return None


pyspiel.register_game(_GAME_TYPE, StoppingGame)
