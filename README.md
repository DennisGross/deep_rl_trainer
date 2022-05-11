# Remote Storage of RL Experiments
Start the MLFlow (Remote) Storage on a (remote) machine:

`mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts`

Pass the following argument:

`tracking_uri: {type: string, default: "http://IP:5000"}`

# Train RL Agent

`mlflow run https://github.com/DennisGross/deep_rl_trainer`

# Deploy RL Agent

`mlflow run https://github.com/DennisGross/deep_rl_trainer -e deploy -P run_id=RUN_ID_FROM_MLFLOW`

# Ray Cluster Setup
Ray Head-Node:
`ray start --head --include-dashboard=true`

Connect Worker-Nodes to cluster:

`ray start --address='HEAD_NODE_IP:6379`

