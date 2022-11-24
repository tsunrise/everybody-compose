# CS 230 Final Project: Everybody Compose: Deep Beats To Music 

## Install
```
git clone https://github.com/tsunrise/cs230-proj.git
cd cs230-proj
pip -r requirements.txt
```
## Training
- Training
```
python ./main.py -m <model_name> -n <num_epochs>
```
Example: 
```
python ./main.py -m lstm -n 1000
```
- Training with checkpoint
```
python ./main.py -m <model_name> -n <num_epochs> -c <checkpoint_path>
```
Example:
```
python ./main.py -m lstm -n 600 -c .\.project_data\snapshots\lstm_all_300.pth
```
## Prediction
- Predict Using Interactive Input
```
python .\predict_stream.py -m <model_name> -c <checkpoint_path>
```
- Predict using recorded interactive input
```
python .\predict_stream.py -m <model_name> -c <checkpoint_path> --source <record_file_path>
```
- "Recolor" a random MIDI file in dataset
```
python .\predict_stream.py -m <model_name> -c <checkpoint_path> --source dataset
```

# Cheatsheets (For Dev Only)

Log into AWS server:
`ssh -i "~/.ssh/cs230_aws_key.pem" ubuntu@<AWS address>`

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
