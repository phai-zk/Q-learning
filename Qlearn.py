import random
import pickle
import json
import os
from Environment.TicTacToe import TicTacToe

# --------------------
# Q-Learning Parameters
# --------------------
alpha = 0.1
gamma = 0.9
epsilon_start = 0.5
epsilon_min = 0.01
epsilon_decay = 0.999

wins = [
	(0,1,2), (3,4,5), (6,7,8),  # แถว
	(0,3,6), (1,4,7), (2,5,8),  # คอลัมน์
	(0,4,8), (2,4,6)            # ทแยง
]

Q = {}
env = TicTacToe()

# สถิติการชนะ
wins_X = 0
wins_O = 0
draws = 0

# --------------------
# Helper functions
# --------------------
def get_Q(state, action):
	return Q.get((state, action), 0.0)

def state_to_str(state):
	return ''.join(state)

def check_winner(board):
	for a,b,c in wins:
		if board[a] != " " and board[a] == board[b] == board[c]:
			return board[a]
	if " " not in board:
		return "D"
	return None

def available_actions(board, token):
	actions = [i for i, spot in enumerate(board) if spot == " "]
	opponent = "X" if token == "O" else "O"
	
	# 1) ถ้าศัตรูมี 2 ตัวติดกันใน line เดียว → ดักก่อน
	for a,b,c in wins:
		line = [board[a], board[b], board[c]]
		if line.count(opponent) == 2 and line.count(" ") == 1:
			if board[a] == " ": return [a]
			if board[b] == " ": return [b]
			if board[c] == " ": return [c]
	
	# 2) ถ้าศัตรูวางแบบ X _ X (สองข้างมีหมาก) → ดักตรงกลาง
	for a,b,c in wins:
		if board[a] == opponent and board[b] == " " and board[c] == opponent:
			return [b]
	
	# 3) ถ้าไม่มี case ดักพิเศษ → คืน action ปกติ
	return actions

def minimax_opponent(board, token):
	"""Opponent แบบง่าย: ชนะทันที / บล็อก / random"""
	actions = available_actions(board, token)
	# ชนะทันที
	for a in actions:
		tmp = board.copy()
		tmp[a] = "O"
		if check_winner(tmp) == "O":
			return a
	# บล็อก AI
	for a in actions:
		tmp = board.copy()
		tmp[a] = "X"
		if check_winner(tmp) == "X":
			return a
	# random
	return random.choice(actions)

# --------------------
# Training loop (Self-Play)
# --------------------
for episode in range(50000):
	board = [" "] * 9
	state_str = state_to_str(board)
	done = False
	epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
	
	# กำหนดให้ X เริ่มเสมอ
	player_turn = "X"
	first_move_done = False  # ไว้เช็คว่าตาแรก X ลงไปแล้ว

	while not done:
		actions = available_actions(board, player_turn)
		if player_turn == "X":
			# ถ้าเป็นตาแรก → ลงตรงกลางเสมอ
			if not first_move_done:
				action = 4
				first_move_done = True
			else:
				# Q-Learning AI move
				if random.random() < epsilon:
					action = random.choice(actions)
				else:
					q_vals = [get_Q(state_str, a) for a in actions]
					maxQ = max(q_vals)
					best_actions = [a for a in actions if get_Q(state_str, a) == maxQ]
					action = random.choice(best_actions)
			board[action] = "X"
		else:
			# Opponent (minimax/self-play)
			action = minimax_opponent(board, player_turn)
			board[action] = "O"

		winner = check_winner(board)
		if winner is not None:
			done = True
			if winner == "X":
				reward = 1
				wins_X += 1
			elif winner == "O":
				reward = -1
				wins_O += 1
			else:
				reward = 0
				draws += 1
		else:
			reward = 0
			# Reward เล็ก ๆ สำหรับ block opponent (เฉพาะ X)
			if player_turn == "X":
				for a in available_actions(board, "X"):
					tmp = board.copy()
					tmp[a] = "O"
					if check_winner(tmp) == "O":
						reward += 0.5

		# Update Q-table (เฉพาะ AI)
		if player_turn == "X":
			maxQ_next = max(
				[get_Q(state_to_str(board), a) for a in available_actions(board, "X")],
				default=0
			) if not done else 0
			oldQ = get_Q(state_str, action)
			Q[(state_str, action)] = oldQ + alpha * (reward + gamma * maxQ_next - oldQ)
			state_str = state_to_str(board)

		# สลับผู้เล่น
		player_turn = "O" if player_turn == "X" else "X"
	if episode % 10000 == 0:
		print(f"Episode {episode}, Epsilon: {epsilon:.4f}, Q-table size: {len(Q)}, X Wins: {wins_X}, O Wins: {wins_O}, Draws: {draws}")

# --------------------
# Save Q-table
# --------------------
if os.path.exists("qtable.pkl"):
	os.remove("qtable.pkl")

with open("qtable.pkl", "wb") as f:
	pickle.dump(Q, f)

if os.path.exists("qtable.json"):
	os.remove("qtable.json")

Q_json = {str(k): v for k, v in Q.items()}
with open("qtable.json", "w") as f:
	json.dump(Q_json, f)

print("Self-play training complete! Q-table saved.")
print(f"Final stats → X Wins: {wins_X}, O Wins: {wins_O}, Draws: {draws}")
