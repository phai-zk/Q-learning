import pickle
import os
from Environment.TicTacToe import TicTacToe

alpha = 0.1
gamma = 0.9
Q = {}

if os.path.exists("qtable.pkl"):
    with open("qtable.pkl", "rb") as f:
        Q = pickle.load(f)

env = TicTacToe()

def state_to_str(state):
    return ''.join(state)

def get_Q(state, action):
    return Q.get((state, action), 0.0)

def init_Q():
    # generate all possible states recursively
    def all_states(board=[" "]*9, player="X"):
        results = []
        winner = env.check_winner(board)
        if winner or " " not in board:
            return [tuple(board)]
        results.append(tuple(board))
        for i in range(9):
            if board[i] == " ":
                new_board = board.copy()
                new_board[i] = player
                next_player = "O" if player == "X" else "X"
                results += all_states(new_board, next_player)
        return list(set(results))
    
    all_boards = all_states()
    for board in all_boards:
        state_str = ''.join(board)
        for a in env.available_actions(list(board)):
            if (state_str, a) not in Q:
                Q[(state_str, a)] = 0.0
    print(f"Initialized Q-table with {len(Q)} entries")

# --------------------
# Recursive play for all state-action
# --------------------
def play_all_states(board, player):
    state_str = state_to_str(board)
    winner = env.check_winner(board)

    # terminal
    if winner:
        if winner == "X": return 1
        elif winner == "O": return -1
        else: return 0

    actions = env.available_actions(board)
    rewards = []

    for a in actions:
        new_board = board.copy()
        new_board[a] = player
        next_player = "O" if player == "X" else "X"
        reward = play_all_states(new_board, next_player)

        # update Q for X only
        if player == "X":
            oldQ = get_Q(state_str, a)
            Q[(state_str, a)] = oldQ + alpha * (reward - oldQ)
        
        rewards.append(reward)

    return max(rewards) if player == "X" else min(rewards)
# --------------------
# Run training
# --------------------
init_Q()
# Play exhaustive starting from O and X
play_all_states([" "]*9, "O")  # O starts
play_all_states([" "]*9, "X")  # X starts

# --------------------
# Save Q-table
# --------------------
with open("qtable.pkl", "wb") as f:
    pickle.dump(Q, f)
print("Training complete! Q-table saved.")