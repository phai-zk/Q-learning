class TicTacToe:
    def __init__(self):
        self.wins = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        self.done = False
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def available_actions(self, board=None):
        b = self.board if board is None else board
        return [i for i, spot in enumerate(b) if spot == " "]

    def check_winner(self, board=None):
        b = self.board if board is None else board
        for a,b1,c in self.wins:
            if b[b1] != " " and b[b1] == b[a] == b[c]:
                return b[a]
        if " " not in b:
            return "D"
        return None

    def step(self, action, player):
        if self.done or self.board[action] != " ":
            raise ValueError("Invalid move")
        self.board[action] = player
        winner = self.check_winner()
        if winner:
            self.done = True
            if winner == "D":
                return self.get_state(), 0, True
            return self.get_state(), 1 if winner=="X" else -1, True
        return self.get_state(), 0, False

    def render(self):
        b = self.board
        print(f"{b[0]}|{b[1]}|{b[2]}")
        print("-+-+-")
        print(f"{b[3]}|{b[4]}|{b[5]}")
        print("-+-+-")
        print(f"{b[6]}|{b[7]}|{b[8]}")
        print()
