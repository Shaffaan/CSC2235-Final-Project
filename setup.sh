#!/bin/bash
echo "Setup script is running as $(whoami)..." >> /local/repository/setup.log

sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-venv python3-pip openjdk-17-jdk-headless

USER_NAME=$(stat -c '%U' /local/repository)
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)

if [ -z "$USER_NAME" ] || [ -z "$USER_HOME" ]; then
    echo "Could not find project owner or home directory. Exiting." >> /local/repository/setup.log
    exit 1
fi

echo "Found project user: $USER_NAME with home: $USER_HOME" >> /local/repository/setup.log

sudo -u "$USER_NAME" bash -c "
    echo 'Creating venv as $USER_NAME...' >> /local/repository/setup.log
    python3 -m venv /local/repository/venv
    
    echo 'Activating venv and installing requirements...' >> /local/repository/setup.log
    source /local/repository/venv/bin/activate
    pip install --upgrade pip
    pip install -r /local/repository/requirements.txt
"

echo "Setup complete. venv is created and packages are installed." >> /local/repository/setup.log