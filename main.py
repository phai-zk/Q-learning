from Environment.TicTacToe import TicTacToe
import pickle
import random

Q = {}
env = TicTacToe()
human = "O"
ai = "X"

try:
	with open("qtable.pkl", "rb") as f:
		Q = pickle.load(f)
except FileNotFoundError:
	print("Q-table not found. Please run Qlearn.py first.")
	exit()

def choose_best_action(state):
	actions = env.available_actions()
	q_vals = [Q.get((state, a), 0) for a in actions]
	maxQ = max(q_vals)
	best_actions = [a for a in actions if Q.get((state, a), 0) == maxQ]
	return random.choice(best_actions)

print(f"Tic-Tac-Toe: You ({human}) vs AI ({ai})")
state = env.reset()
env.render()

while not env.done:
	# Human move
	try:
		human_action = int(input("Choose position (1-9): ")) - 1;
		state, reward, done = env.step(human_action, player=human)
	except ValueError:
		print("Invalid input. Try again.")
		continue

	env.render()
	if done:
		if reward == 1:
			print("You win!")
		elif reward == 0:
			print("Draw!")
		break

	# AI move
	ai_action = choose_best_action(state)
	print(f"AI chooses: {ai_action}")
	state, reward, done = env.step(ai_action, player=ai)
	env.render()
	if done:
		if reward == 1:
			print("AI wins!")
		elif reward == 0:
			print("Draw!")
		break