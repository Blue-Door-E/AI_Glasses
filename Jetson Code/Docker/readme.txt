Requirements

NVIDIA Jetson device

Docker installed

NVIDIA container runtime enabled

Bluetooth enabled on host

Docker running with sudo access

Verify Docker:

docker --version
sudo docker info | grep Runtimes


You should see nvidia listed.

Installation (Manual Run)
1) Load Docker Image
sudo docker load -i edith-glasses_jp64.tar

2) Install Entrypoint Script
sudo mkdir -p /opt/edith
sudo cp deploy/entrypoint.sh /opt/edith/entrypoint.sh
sudo chmod +x /opt/edith/entrypoint.sh

3) Run Container
sudo docker run -d \
  --name edith-glasses \
  --runtime nvidia \
  --network host \
  -v /home/$USER:/home/$USER \
  -v /opt/edith/entrypoint.sh:/entrypoint.sh:ro \
  -v /var/run/dbus:/var/run/dbus \
  -v /run/dbus:/run/dbus \
  -w /workspace \
  edith-glasses:jp64

Enable Auto Boot (Systemd)
1) Install Service
sudo cp deploy/systemd/edith.service /etc/systemd/system/edith.service

2) Reload + Enable
sudo systemctl daemon-reload
sudo systemctl enable --now edith.service


Check status:

sudo systemctl status edith.service

Networking

The container uses:

--network host


This allows:

Direct WiFi access

BLE communication

No explicit port mapping required

Bluetooth Support

Bluetooth access is provided via DBus socket binding:

-v /var/run/dbus:/var/run/dbus
-v /run/dbus:/run/dbus


Ensure Bluetooth is enabled on the host:

sudo systemctl status bluetooth

GPU Acceleration

The container runs with:

--runtime nvidia


Verify inside container:

nvidia-smi


(Or Jetson equivalent GPU validation.)

Stopping the Container

Manual stop:

sudo docker stop edith-glasses


If using systemd:

sudo systemctl stop edith.service

