import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")
        self.board = [' ' for _ in range(9)]
        self.buttons = []
        self.current_player = "X"
        self.create_widgets()

    def create_widgets(self):
        for i in range(9):
            button = tk.Button(self.root, text=" ", font=('normal', 40), width=5, height=2,
                               command=lambda i=i: self.on_click(i))
            button.grid(row=i//3, column=i%3)
            self.buttons.append(button)
        
        self.reset_button = tk.Button(self.root, text="Restart", command=self.restart_game)
        self.reset_button.grid(row=3, column=0, columnspan=3, sticky='we')

    def on_click(self, index):
        if self.board[index] == ' ':
            self.board[index] = self.current_player
            self.update_buttons()
            if self.check_winner(self.current_player):
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.disable_buttons()
            elif ' ' not in self.board:
                messagebox.showinfo("Game Over", "It's a draw!")
                self.disable_buttons()
            else:
                self.current_player = "O" if self.current_player == "X" else "X"
                if self.current_player == "O":
                    self.root.after(500, self.ai_move)

    def ai_move(self):
        index = self.minimax(self.board, "O")['index']
        self.board[index] = "O"
        self.update_buttons()
        if self.check_winner("O"):
            messagebox.showinfo("Game Over", "AI wins!")
            self.disable_buttons()
        elif ' ' not in self.board:
            messagebox.showinfo("Game Over", "It's a draw!")
            self.disable_buttons()
        else:
            self.current_player = "X"

    def minimax(self, new_board, player):
        avail_spots = [i for i, spot in enumerate(new_board) if spot == ' ']
        if self.check_winner("X"): return {"score": -10}
        elif self.check_winner("O"): return {"score": 10}
        elif not avail_spots: return {"score": 0}
        moves = []
        for spot in avail_spots:
            move = {"index": spot}
            new_board[spot] = player
            result = self.minimax(new_board, "O" if player == "X" else "X")
            move['score'] = result['score']
            new_board[spot] = ' '
            moves.append(move)
        best_move = max(moves, key=lambda x: x['score']) if player == "O" else min(moves, key=lambda x: x['score'])
        return best_move

    def update_buttons(self):
        for i, button in enumerate(self.buttons):
            button.config(text=self.board[i])

    def check_winner(self, player):
        win_conditions = [
            [self.board[0], self.board[1], self.board[2]],
            [self.board[3], self.board[4], self.board[5]],
            [self.board[6], self.board[7], self.board[8]],
            [self.board[0], self.board[3], self.board[6]],
            [self.board[1], self.board[4], self.board[7]],
            [self.board[2], self.board[5], self.board[8]],
            [self.board[0], self.board[4], self.board[8]],
            [self.board[2], self.board[4], self.board[6]],
        ]
        return [player, player, player] in win_conditions

    def disable_buttons(self):
        for button in self.buttons:
            button.config(state=tk.DISABLED)

    def restart_game(self):
        self.board = [' ' for _ in range(9)]
        self.current_player = "X"
        self.update_buttons()
        for button in self.buttons:
            button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()
