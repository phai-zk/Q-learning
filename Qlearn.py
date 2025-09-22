import pickle
import json
import os
import random
from Environment.TicTacToe import TicTacToe

# -------------------- Parameters --------------------
alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9999
num_episodes = 5000000
block_bonus = 0.5
Q = {}
env = TicTacToe()

# -------------------- Helper Functions --------------------
def state_to_str(state):
	return ''.join(state)

def get_Q(state, action):
	return Q.get((state, action), 0.0)

def choose_action(state_str, available_actions, epsilon):
    q_values = [get_Q(state_str, a) for a in available_actions]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]

    if random.random() < epsilon:
        return random.choice(available_actions)

    return random.choice(best_actions)

# -------------------- Load Q-table if exists --------------------
if os.path.exists("qtable.pkl"):
	with open("qtable.pkl", "rb") as f:
		Q = pickle.load(f)

epsilon = epsilon_start

# -------------------- Training Loop --------------------
for episode in range(num_episodes):
	state = env.reset()
	state_str = state_to_str(state)
	done = False
	player = random.choice(["X", "O"])  # random start

	while not done:
		available_actions = env.available_actions()
		action = choose_action(state_str, available_actions, epsilon)
		next_state, reward, done = env.step(action, player)
		next_state_str = state_to_str(next_state)

		# -------------------- Block reward --------------------
		block_reward = 0
		opponent = "O" if player == "X" else "X"
		for a in available_actions:
			temp_board = list(env.board)
			temp_board[a] = opponent
			if env.check_winner(temp_board) == opponent and a == action:
				block_reward = block_bonus
				break

		# -------------------- Q-Learning update --------------------
		if not done:
			future_rewards = [get_Q(next_state_str, a) for a in env.available_actions()]
			max_future = max(future_rewards) if future_rewards else 0
		else:
			if reward == 1:
				max_future = 1 if player == "X" else -1
			elif reward == -1:
				max_future = -1 if player == "X" else 1
			else:
				max_future = 0

		total_reward = max_future + block_reward
		old_q = get_Q(state_str, action)
		Q[(state_str, action)] = old_q + alpha * (total_reward - old_q)

		state_str = next_state_str
		player = "O" if player == "X" else "X"

	# -------------------- Decay epsilon --------------------
	if epsilon > epsilon_min:
		epsilon *= epsilon_decay

	# -------------------- Print progress --------------------
	if (episode + 1) % 10000 == 0:
		print(f"Episode {episode + 1}/{num_episodes}, epsilon={epsilon:.4f}")

# -------------------- Save Q-table --------------------
# with open("qtable.pkl", "wb") as f:
# 	pickle.dump(Q, f)

Q_json = {f"{state}|{action}": value for (state, action), value in Q.items()}
with open("smart_qtable.json", "w") as f:
	json.dump(Q_json, f)

print("Training complete! Q-table saved (pickle + JSON).")
