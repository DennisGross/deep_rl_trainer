# Remote Storage of RL Experiments
Start the MLFlow (Remote) Storage on a (remote) machine:

`mlflow server --host 0.0.0.0 --port 5000 --serve-artifacts`

Pass the following argument:

`tracking_uri: {type: string, default: "http://IP:5000"}`