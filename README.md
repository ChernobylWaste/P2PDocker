
# XGBoost Federated Learning Using P2P Docker
This is repository for assignment Advanced Network Security. The nodes roles as Server and Client.

## How to Use
Make sure u must have docker on your computer.
- Build docker-compose.yml
```bash
  docker-compose build
```
- Run docker-compose.yml
```bash
  docker-compose up
```

## Example for Sending Files to Another Nodes
- List Running container
```bash
docker container ls
```
- Entering docker container bash
```bash
  docker exec -it <container-name> bash 
```

- Change Directory to /files
```bash
  cd /files
```

- Make File Using nano
```bash
  nano <filename>
```

# IMPORTANT!
on lokalml.py line 80, change the model name in nodes to model1.json, model2.json, and model3.json.