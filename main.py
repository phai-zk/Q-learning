import pickle
import random
from Environment.TicTacToe import TicTacToe

# --------------------
# Load Q-table
# --------------------
with open("qtable.pkl", "rb") as f:
	Q = pickle.load(f)

# --------------------
# Helpers
# --------------------
def state_to_str(state):
	return ''.join(state)

def get_Q(state, action):
	return Q.get((state, action), 0.0)

def choose_best_action(state, board):
	actions = [i for i, spot in enumerate(board) if spot == " "]
	if not actions:
		return None
	q_vals = [get_Q(state, a) for a in actions]
	maxQ = max(q_vals)
	best_actions = [a for a in actions if get_Q(state, a) == maxQ]
	return random.choice(best_actions)

# --------------------
# Play loop
# --------------------
env = TicTacToe()
state = tuple(env.board)
done = False
human = "O"
ai = "X"

env.reset()

while not done:
	# Human move
	env.render()
	try:
		human_action = int(input(f"Choose ({human}) position (1-9): ")) - 1
		if human_action not in env.available_actions():
			print("Invalid move. Try again.")
			continue
		env.board[human_action] = human
	except (ValueError, IndexError):
		print("Invalid move. Try again.")
		continue

	winner = env.check_winner()
	if winner:
		done = True
		if winner == human:
			print("You win!")
		elif winner == ai:
			print("AI wins!")
		else:
			print("Draw!")
		break

	# AI move
	state_str = state_to_str(tuple(env.board))
	ai_action = choose_best_action(state_str, env.board)
	if ai_action is not None:
		env.board[ai_action] = ai
		print(f"AI chooses position {ai_action + 1}")

	# env.render()
	winner = env.check_winner()
	if winner:
		done = True
		if winner == human:
			print("You win!")
		elif winner == ai:
			print("AI wins!")
		else:
			print("Draw!")
