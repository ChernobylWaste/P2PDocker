
# Peer to Peer Network Using Docker
This is repository for assignment Advanced Network Security. Node 1 as Server and 3 Nodes as Client. This project can be improved to Federated Learning.

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

- Change Directory to /Files
```bash
  cd /Files
```

- Make File Using nano
```bash
  nano <filename>
```