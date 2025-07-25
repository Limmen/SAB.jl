from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import nfsp
import stopping_game
import numpy as np
import time


class NFSPPolicies(policy.Policy):
    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def mc_exploitability(env: rl_environment.Environment,
                      pi: policy.Policy,
                      K: int = 10,
                      depth_limit: int = 50) -> float:
    """Unbiased Monte‑Carlo estimate of exploitability (two‑player zero‑sum)."""

    def rollout(state, acting_policy, d):
        """Return terminal utilities after following `acting_policy` for ≤d steps."""
        depth = 0
        s = state.clone()
        while not s.is_terminal() and depth < d:
            cur = s.current_player()
            if cur == -1:  # chance node
                s.apply_action(np.random.choice(s.legal_actions()))
            else:
                a_prob = acting_policy.action_probabilities(s, cur)
                a = np.random.choice(list(a_prob), p=list(a_prob.values()))
                s.apply_action(a)
            depth += 1
        return s.returns()

    n_players = env.game.num_players()
    exp_sum = 0.0  # Σ_i exploitability_i
    for i in range(n_players):
        br_gain = 0.
        base_value = 0.
        for _ in range(K):
            s = env.game.new_initial_state()

            # --- base roll‑out (everyone follows π) ---
            base_value += rollout(s, pi, depth_limit)[i]

            # --- BR roll‑out: player i picks one‑step greedy actions ---
            s = env.game.new_initial_state()
            depth = 0
            while not s.is_terminal() and depth < depth_limit:
                cur = s.current_player()
                if cur == -1:
                    s.apply_action(np.random.choice(s.legal_actions()))
                elif cur != i:
                    a_prob = pi.action_probabilities(s, cur)
                    a = np.random.choice(list(a_prob), p=list(a_prob.values()))
                    s.apply_action(a)
                else:
                    # one‑step greedy
                    best_a, best_q = None, -np.inf
                    for a in s.legal_actions(cur):
                        tmp = s.clone()
                        tmp.apply_action(a)
                        q = rollout(tmp, pi, depth_limit - depth - 1)[i]
                        if q > best_q:
                            best_q, best_a = q, a
                    s.apply_action(best_a)
                depth += 1
            br_gain += s.returns()[i]

        exp_sum += (br_gain - base_value) / K
    # For two‑player zero‑sum, exploitability is half the NashConv
    return exp_sum / 2.0


def rolling_average(data, window_size):
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if not data:
        return 0.0  # or raise an error, depending on your preference

    window = data[-window_size:] if len(data) >= window_size else data
    return sum(window) / len(window)


def main():
    game = "python_stopping"
    num_players = 2
    num_train_episodes = int(3e6)
    eval_every = 100
    hidden_layers_sizes = [128]
    replay_buffer_capacity = int(3e6)
    reservoir_buffer_capacity = int(2e6)
    anticipatory_param = 0.1

    env_configs = {}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in hidden_layers_sizes]
    kwargs = {
        "replay_buffer_capacity": replay_buffer_capacity,
        "epsilon_decay_duration": num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    agents = [
        nfsp.NFSP(
            idx,
            info_state_size,
            num_actions,
            hidden_layers_sizes,
            reservoir_buffer_capacity,
            anticipatory_param,
            **kwargs
        )
        for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
    exploitability = []
    start = time.time()
    for ep in range(num_train_episodes):
        if (ep + 1) % eval_every == 0:
            approx_expl = mc_exploitability(env, expl_policies_avg, K=100)
            exploitability.append(approx_expl)
            print(f"Episode: {ep + 1}. Exploitability {rolling_average(exploitability, 20)}, time: {time.time() - start}")

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        for agent in agents:
            agent.step(time_step)


if __name__ == "__main__":
    main()
