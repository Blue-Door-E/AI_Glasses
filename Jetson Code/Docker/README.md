# EDITH Jetson Docker Deployment

This folder contains everything needed to run the EDITH app on an NVIDIA Jetson using Docker, plus optional auto-start on reboot with systemd.

## What Is In This Folder

- `edith-glasses_jp64.tar`: Docker image archive file. Load this with `docker load -i ...`.
- `entrypoint.sh`: Script executed inside the container at startup. It launches `Demo.py` and restarts it if it exits.
- `dockerrun`: One-line helper script showing the `docker run` command used for manual launch.
- `systemd`: Captured snippet from `systemctl cat edith.service` (currently truncated).
- `readme.txt`: Older setup notes.

## Runtime Flow (Simple)

1. Jetson boots.
2. `systemd` starts `edith.service` (if installed/enabled).
3. Service runs `docker run` for image `edith-glasses:jp64`.
4. Container starts and runs `entrypoint.sh`.
5. `entrypoint.sh` runs `python3 -u Demo.py` in `/home/erick/EDITH_CODE`.
6. If `Demo.py` exits, entrypoint waits 2 seconds and starts it again.
7. If container exits, systemd restarts container (`Restart=always`).

## Prerequisites

- NVIDIA Jetson device
- Docker installed
- NVIDIA runtime available in Docker
- Bluetooth enabled on host
- User can run `sudo docker ...`

Check Docker runtime:

```bash
docker --version
sudo docker info | grep -i Runtimes
```

You should see `nvidia` in the runtimes list.

## Manual Start

Load image:

```bash
sudo docker load -i edith-glasses_jp64.tar
```

Install entrypoint:

```bash
sudo mkdir -p /opt/edith
sudo cp entrypoint.sh /opt/edith/entrypoint.sh
sudo chmod +x /opt/edith/entrypoint.sh
```

Run container:

```bash
sudo docker run -d \
  --name edith-glasses \
  --runtime nvidia \
  --network host \
  -v "/home/CI_Pipeline/AI_Glasses/Jetson Code/Code:/home/erick/EDITH_CODE" \
  -v /opt/edith/entrypoint.sh:/entrypoint.sh:ro \
  -v /var/run/dbus:/var/run/dbus \
  -v /run/dbus:/run/dbus \
  -w /workspace \
  edith-glasses:jp64
```
Run just the container 
sudo docker run -it \
  --name edith-glasses \
  --runtime nvidia \
  --network host \
  -v "/home/CI_Pipeline/AI_Glasses/Jetson Code/Code:/home/erick/EDITH_CODE" \
  -v /opt/edith/entrypoint.sh:/entrypoint.sh:ro \
  -v /var/run/dbus:/var/run/dbus \
  -v /run/dbus:/run/dbus \
  -w /workspace \
  edith-glasses:jp64
```



## Auto-Start On Reboot (systemd)

Use this as a recommended complete service file:

```ini
[Unit]
Description=E.D.I.T.H Docker (keep running)
After=docker.service network-online.target bluetooth.service
Wants=network-online.target
Requires=docker.service bluetooth.service

[Service]
TimeoutStartSec=0
Restart=always
RestartSec=2
Environment=CONTAINER_NAME=edith-glasses
Environment=IMAGE_NAME=edith-glasses:jp64
ExecStartPre=-/usr/bin/docker rm -f ${CONTAINER_NAME}
ExecStart=/usr/bin/docker run --pull=never --name ${CONTAINER_NAME} \
  --runtime nvidia \
  --network host \
  -v /home/erick:/home/erick \
  -v /opt/edith/entrypoint.sh:/entrypoint.sh:ro \
  -v /var/run/dbus:/var/run/dbus \
  -v /run/dbus:/run/dbus \
  -w /workspace \
  ${IMAGE_NAME}
ExecStop=/usr/bin/docker stop ${CONTAINER_NAME}

[Install]
WantedBy=multi-user.target
```

Install/enable:

```bash
sudo cp /path/to/edith.service /etc/systemd/system/edith.service
sudo systemctl daemon-reload
sudo systemctl enable --now edith.service
sudo systemctl status edith.service
```

## Commands You Will Use Most

```bash
# Logs from container
sudo docker logs -f edith-glasses

# Stop/start manually
sudo docker stop edith-glasses
sudo docker start edith-glasses

# If using systemd
sudo systemctl restart edith.service
sudo systemctl status edith.service
sudo systemctl cat --no-pager edith.service
```

## Notes

- The `entrypoint.sh` loop is what keeps restarting `Demo.py` inside the container.
- The systemd `Restart=always` is what restarts the Docker container on host reboot or container exit.
- Your pasted `systemctl cat` output ended at `lines 1-23`, so the currently shared unit content is still incomplete.
- I could not inspect `edith-glasses_jp64.tar` contents in this environment because the archive is root-readable only (`sudo` password required).
