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
epsilon_decay = 0.99995  # decay ช้าลงเพื่อ explore นานขึ้น

wins = [
    (0,1,2), (3,4,5), (6,7,8),  # แถว
    (0,3,6), (1,4,7), (2,5,8),  # คอลัมน์
    (0,4,8), (2,4,6)            # ทแยง
]

Q = {}
if os.path.exists("qtable.pkl"):
	with open("qtable.pkl", "rb") as f:
		Q = pickle.load(f)
env = TicTacToe()

# --------------------
# สถิติการชนะ
# --------------------
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
    
    # ดัก opponent 2 ตัวติดกัน
    for a,b,c in wins:
        line = [board[a], board[b], board[c]]
        if line.count(opponent) == 2 and line.count(" ") == 1:
            if board[a] == " ": return [a]
            if board[b] == " ": return [b]
            if board[c] == " ": return [c]
    
    # X _ X → ดักตรงกลาง
    for a,b,c in wins:
        if board[a] == opponent and board[b] == " " and board[c] == opponent:
            return [b]
    
    return actions

def minimax_opponent(board, token):
    actions = [i for i, spot in enumerate(board) if spot == " "]
    # ชนะทันที
    for a in actions:
        tmp = board.copy()
        tmp[a] = token
        if check_winner(tmp) == token:
            return a
    # บล็อก X
    opponent = "X" if token == "O" else "O"
    for a in actions:
        tmp = board.copy()
        tmp[a] = opponent
        if check_winner(tmp) == opponent:
            return a
    # heuristic: เลือกมุม > กลาง > random
    corners = [i for i in [0,2,6,8] if i in actions]
    if corners: return random.choice(corners)
    if 4 in actions: return 4
    return random.choice(actions)

# --------------------
# Training loop
# --------------------
# --------------------
# Training loop
# --------------------
for episode in range(1000000):
    board = [" "] * 9
    state_str = state_to_str(board)
    done = False
    epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))
    
    player_turn = "X"

    while not done:
        actions = available_actions(board, player_turn)
        
        if player_turn == "X":
            # เลือก action แบบ epsilon-greedy ทุกตา
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_vals = [get_Q(state_str, a) for a in actions]
                maxQ = max(q_vals)
                best_actions = [a for a in actions if get_Q(state_str, a) == maxQ]
                action = random.choice(best_actions)
            board[action] = "X"
        else:
            # Opponent
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
            if player_turn == "X":
                # reward สำหรับ block O
                for a in available_actions(board, "X"):
                    tmp = board.copy()
                    tmp[a] = "O"
                    if check_winner(tmp) == "O":
                        reward += 1.0
                # reward สำหรับสร้างโอกาสชนะ
                for a,b,c in wins:
                    line = [board[a], board[b], board[c]]
                    if line.count("X") == 2 and line.count(" ") == 1:
                        reward += 0.5

        # Q-update
        if player_turn == "X":
            maxQ_next = max(
                [get_Q(state_to_str(board), a) for a in available_actions(board, "X")],
                default=0
            ) if not done else 0
            oldQ = get_Q(state_str, action)
            Q[(state_str, action)] = oldQ + alpha * (reward + gamma * maxQ_next - oldQ)
            state_str = state_to_str(board)

        player_turn = "O" if player_turn == "X" else "X"
    
    if episode % 10000 == 0:
        print(f"Episode {episode}, Epsilon: {epsilon:.5f}, Q-table size: {len(Q)}, X Wins: {wins_X}, O Wins: {wins_O}, Draws: {draws}")

# --------------------
# Save Q-table
# --------------------
with open("qtable.pkl", "wb") as f:
    pickle.dump(Q, f)

Q_json = {str(k): v for k, v in Q.items()}
with open("qtable.json", "w") as f:
    json.dump(Q_json, f)

print("Training complete!")
print(f"Final stats → X Wins: {wins_X}, O Wins: {wins_O}, Draws: {draws}")
