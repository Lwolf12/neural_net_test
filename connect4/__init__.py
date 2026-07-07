"""AlphaZero-style self-play trainer for Connect 4.

The AI learns purely from games it plays against itself — no perfect solver, no
hand-labelled data. A single network predicts a move policy and a position
value; MCTS uses it to search, self-play produces training data, and training
sharpens the network. Repeat.

Modules
-------
config   : all hyperparameters and board size (edit here)
game     : Connect 4 rules on a numpy board (no torch)
model    : the policy+value residual conv net
mcts     : batched PUCT Monte-Carlo Tree Search
selfplay : self-play data generation + arena evaluation
train    : runnable trainer  ->  python -m connect4.train

A separate program to play against the trained net can import `game`, `model`,
and `mcts` and load the saved checkpoint (config.CHECKPOINT).
"""
