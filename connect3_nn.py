import random
from functools import lru_cache

import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

ROWS, COLS, CONNECT = 4, 3, 3
EMPTY, X, O = 0, 1, -1


def ix(r, c):
    return r * COLS + c


def legal_moves(board):
    return [c for c in range(COLS) if board[ix(0, c)] == EMPTY]


def drop_piece(board, col, player):
    board = list(board)

    for r in range(ROWS - 1, -1, -1):
        if board[ix(r, col)] == EMPTY:
            board[ix(r, col)] = player
            return tuple(board)

    raise ValueError("Column is full")


def winner(board):
    directions = [
        (0, 1),    # horizontal
        (1, 0),    # vertical
        (1, 1),    # diagonal down-right
        (1, -1),   # diagonal down-left
    ]

    for r in range(ROWS):
        for c in range(COLS):
            player = board[ix(r, c)]
            if player == EMPTY:
                continue

            for dr, dc in directions:
                count = 0
                rr, cc = r, c

                while 0 <= rr < ROWS and 0 <= cc < COLS and board[ix(rr, cc)] == player:
                    count += 1
                    if count >= CONNECT:
                        return player
                    rr += dr
                    cc += dc

    return EMPTY


def perspective(board, player):
    """
    Make the board relative to the player to move.

    Current player's pieces are +1.
    Opponent's pieces are -1.
    Empty spaces are 0.
    """
    return tuple(cell * player for cell in board)


@lru_cache(None)
def solve_position(board, player):
    """
    Perfect solver for this tiny game.

    Return value from current player's perspective:
    +1 = current player can force a win
     0 = current player can force a draw
    -1 = current player loses with best play
    """
    win = winner(board)

    if win == player:
        return 1

    if win == -player:
        return -1

    moves = legal_moves(board)

    if not moves:
        return 0

    best_value = -2

    for move in moves:
        next_board = drop_piece(board, move, player)

        # After our move, opponent moves.
        # Their value is our negative value.
        value = -solve_position(next_board, -player)
        best_value = max(best_value, value)

    return best_value


def enumerate_positions():
    """
    Generate all reachable non-terminal positions for this small game.
    """
    seen = set()
    positions = []

    def visit(board, player):
        key = (board, player)

        if key in seen:
            return

        seen.add(key)

        if winner(board) != EMPTY:
            return

        if not legal_moves(board):
            return

        value = solve_position(board, player)
        positions.append((board, player, value))

        for move in legal_moves(board):
            next_board = drop_piece(board, move, player)
            visit(next_board, -player)

    empty_board = tuple([EMPTY] * (ROWS * COLS))
    visit(empty_board, X)

    return positions


class BoardValueNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(ROWS * COLS, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.layers(x)


def train_model():
    data = enumerate_positions()
    random.shuffle(data)

    inputs = torch.tensor(
        [perspective(board, player) for board, player, value in data],
        dtype=torch.float32,
    )

    # Convert value -1, 0, +1 into class 0, 1, 2.
    targets = torch.tensor(
        [value + 1 for board, player, value in data],
        dtype=torch.long,
    )

    split = int(len(data) * 0.80)

    train_x = inputs[:split]
    train_y = targets[:split]

    test_x = inputs[split:]
    test_y = targets[split:]

    model = BoardValueNet()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {len(train_x)} positions.")
    print(f"Testing on {len(test_x)} unseen positions.")

    for epoch in range(1, 401):
        logits = model(train_x)
        loss = loss_fn(logits, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                predictions = model(test_x).argmax(dim=1)
                accuracy = (predictions == test_y).float().mean().item()

            print(
                f"epoch {epoch:3d} | "
                f"loss {loss.item():.4f} | "
                f"test accuracy {accuracy:.3f}"
            )

    torch.save(model.state_dict(), "connect3_value_net.pt")
    print("Saved neural network to connect3_value_net.pt")

    return model


def neural_value(model, board, player):
    """
    Ask the neural net for its estimated value of this board.
    """
    x = torch.tensor([perspective(board, player)], dtype=torch.float32)

    with torch.no_grad():
        probabilities = torch.softmax(model(x)[0], dim=0)

    # Class 0 = loss, class 1 = draw, class 2 = win.
    return float(
        probabilities[0] * -1.0
        + probabilities[1] * 0.0
        + probabilities[2] * 1.0
    )


def ai_move(model, board, player):
    best_move = None
    best_value = -999.0

    for move in legal_moves(board):
        next_board = drop_piece(board, move, player)

        if winner(next_board) == player:
            return move

        if not legal_moves(next_board):
            value = 0.0
        else:
            # Opponent moves next.
            # If opponent's position is good, our position is bad.
            value = -neural_value(model, next_board, -player)

        if value > best_value:
            best_value = value
            best_move = move

    return best_move


def print_board(board):
    symbols = {
        EMPTY: ".",
        X: "X",
        O: "O",
    }

    print()

    for r in range(ROWS):
        print(" ".join(symbols[board[ix(r, c)]] for c in range(COLS)))

    print(" ".join(str(c) for c in range(COLS)))
    print()


def play_against_ai(model):
    board = tuple([EMPTY] * (ROWS * COLS))
    player = X

    print("You are X. The neural network is O. Connect 3 wins.")

    while True:
        print_board(board)

        if player == X:
            moves = legal_moves(board)

            while True:
                try:
                    move = int(input(f"Your move {moves}: "))
                    if move in moves:
                        break
                except ValueError:
                    pass

                print("Invalid move.")
        else:
            move = ai_move(model, board, O)
            print(f"AI plays column {move}")

        board = drop_piece(board, move, player)

        win = winner(board)

        if win != EMPTY:
            print_board(board)
            print("You win." if win == X else "AI wins.")
            break

        if not legal_moves(board):
            print_board(board)
            print("Draw.")
            break

        player = -player


if __name__ == "__main__":
    random.seed(1)
    torch.manual_seed(1)

    model = train_model()
    play_against_ai(model)