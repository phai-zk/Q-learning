class TicTacToe:
	def __init__(self):
		self.reset()

	def reset(self):
		self.board = [" "] * 9
		self.done = False
		return tuple(self.board)

	def available_actions(self):
		return [i for i, spot in enumerate(self.board) if spot == " "]

	def step(self, action, player="X"):
		if self.board[action] != " ":
			raise ValueError("Invalid action")
		self.board[action] = player

		winner = self.check_winner()
		if winner == player:
			self.done = True
			return tuple(self.board), 1, True
		elif winner == "D":
			self.done = True
			return tuple(self.board), 0, True
		else:
			return tuple(self.board), 0, False

	def check_winner(self):
		wins = [(0,1,2),(3,4,5),(6,7,8),
				(0,3,6),(1,4,7),(2,5,8),
				(0,4,8),(2,4,6)]
		for a,b,c in wins:
			if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
				return self.board[a]
		if " " not in self.board:
			return "D"
		return None

	def render(self):
		print("\n")
		for i in range(0, 9, 3):
			print(self.board[i:i+3])
		print("\n")
