# CS 230 Final Project: Everybody Compose: Deep Beats To Music 

Log into AWS server:
`ssh -i "~/.ssh/cs230_aws_key.pem" ubuntu@ec2-34-220-120-34.us-west-2.compute.amazonaws.com`

Using tmux is recommended to avoid potential lost of progress due to disconnction. Start a new tmux session:
`tmux new`

Activate virtual environment:
`conda activate deepbeats`

Go to project directory:
`cd /home/ubuntu/src/cs230-proj`

Make sure code is up-to-date:
`git pull`

Run training command:
`python main.py --n_files 10`

Detach from the current tmux session: `Ctrl + b` first, then press `d`



tmux cheatsheet:

* Create a new session with tmux: `tmux new`

* Go to a previously created tmux session: `tmux attach-session -t session_id`

* Detach from the current session (without shutting it down):`Ctrl + b` first, then press `d`

* List all sessions: `tmux ls`
