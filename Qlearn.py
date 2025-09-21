import pickle
import json
import os
import random
from Environment.TicTacToe import TicTacToe

alpha = 0.1
gamma = 0.9
epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9999
num_episodes = 200000
Q = {}

env = TicTacToe()

def state_to_str(state):
	return ''.join(state)

def get_Q(state, action):
	return Q.get((state, action), 0.0)

def choose_action(state_str, available_actions, board, epsilon):
    # 1. Epsilon-greedy Q-Learning
    if random.random() < epsilon:
        return random.choice(available_actions)  # explore
    
    # 2. Check if opponent is about to win → block
    opponent = "O"  # สมมติ agent = X
    for a in available_actions:
        temp_board = list(board)
        temp_board[a] = opponent
        if env.check_winner(temp_board) == opponent:
            return a  # block opponent

    # 3. Q-Learning max action
    q_values = [get_Q(state_str, a) for a in available_actions]
    max_q = max(q_values)
    best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]

    # 4. Heuristic: corner first
    corners = [pos for pos in best_actions if pos in [0,2,6,8]]
    if corners:
        return random.choice(corners)
    
    # 5. Heuristic: center
    if 4 in best_actions:
        return 4
    
    # 6. fallback: any best Q-action
    return random.choice(best_actions)


# Load existing Q-table (pickle)
if os.path.exists("qtable.pkl"):
	with open("qtable.pkl", "rb") as f:
		Q = pickle.load(f)

epsilon = epsilon_start

for episode in range(num_episodes):
	state = env.reset()
	state_str = state_to_str(state)
	done = False
	player = random.choice(["X", "O"])  # random start

	while not done:
		available_actions = env.available_actions()
		action = choose_action(state_str, available_actions, env.board, epsilon)
		next_state, reward, done = env.step(action, player)
		next_state_str = state_to_str(next_state)

		# Q-Learning update
		if not done:
			future_rewards = [get_Q(next_state_str, a) for a in env.available_actions()]
			max_future = max(future_rewards) if future_rewards else 0
		else:
			# reward from perspective of X
			if reward == 1:
				max_future = 1 if player == "X" else -1
			elif reward == -1:
				max_future = -1 if player == "X" else 1
			else:
				max_future = 0

		old_q = get_Q(state_str, action)
		Q[(state_str, action)] = old_q + alpha * (max_future - old_q)

		state_str = next_state_str
		player = "O" if player == "X" else "X"

	# Decay epsilon
	if epsilon > epsilon_min:
		epsilon *= epsilon_decay

	# Print progress every 10k episodes
	if (episode + 1) % 10000 == 0:
		print(f"Episode {episode + 1}/{num_episodes}, epsilon={epsilon:.4f}")

# Save Q-table (pickle + JSON)
with open("qtable.pkl", "wb") as f:
	pickle.dump(Q, f)

Q_json = {f"{state}|{action}": value for (state, action), value in Q.items()}
with open("qtable.json", "w") as f:
	json.dump(Q_json, f)

print("Training complete! Q-table saved.")
