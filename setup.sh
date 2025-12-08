#!/bin/bash
echo "Setup script is running as $(whoami) on $(hostname)..." | sudo tee -a /local/repository/setup.log

USER_NAME=$(stat -c '%U' /local/repository)

echo "Checking for unmounted high-capacity disks..." | sudo tee -a /local/repository/setup.log

BIG_DISK=$(lsblk -dn -o NAME,SIZE,TYPE,MOUNTPOINT -b | awk '$3=="disk" && $4=="" {print $1, $2}' | sort -rn -k2 | head -1 | awk '{print $1}')

if [ -n "$BIG_DISK" ]; then
    DISK_PATH="/dev/$BIG_DISK"
    MOUNT_POINT="/mnt/fast_storage"
    
    echo "Found unmounted large disk: $DISK_PATH. Formatting and mounting..." | sudo tee -a /local/repository/setup.log
    
    sudo mkfs.ext4 -F "$DISK_PATH"
    
    sudo mkdir -p "$MOUNT_POINT"
    sudo mount "$DISK_PATH" "$MOUNT_POINT"
    
    sudo chown -R "$USER_NAME" "$MOUNT_POINT"
    
    mkdir -p "$MOUNT_POINT/data"

    if [ -d "/local/repository/data" ] && [ ! -L "/local/repository/data" ]; then
        mv /local/repository/data/* "$MOUNT_POINT/data/" 2>/dev/null
        rmdir /local/repository/data
    fi
    ln -sfn "$MOUNT_POINT/data" /local/repository/data
    
    mkdir -p "$MOUNT_POINT/tmp"
    if [ -d "/local/repository/.tmp" ] && [ ! -L "/local/repository/.tmp" ]; then
        rm -rf /local/repository/.tmp
    fi
    mkdir -p /local/repository
    ln -sfn "$MOUNT_POINT/tmp" /local/repository/.tmp
    
    echo "Storage setup complete. /local/repository/data and .tmp now point to $DISK_PATH" | sudo tee -a /local/repository/setup.log
else
    echo "No unmounted disk found. Proceeding with root storage (RISKY)." | sudo tee -a /local/repository/setup.log
fi


sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-venv python3-pip openjdk-17-jdk-headless wget

USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)

if [ -z "$USER_NAME" ] || [ -z "$USER_HOME" ]; then
    echo "Could not find project owner. Exiting." | sudo tee -a /local/repository/setup.log
    exit 1
fi

SPARK_VERSION="3.5.0"
SPARK_DIR="/opt/spark"
if [ ! -d "$SPARK_DIR" ]; then
    echo "Installing Spark Binaries to $SPARK_DIR..." | sudo tee -a /local/repository/setup.log
    
    cd /tmp
    wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz
    
    sudo tar xf spark-${SPARK_VERSION}-bin-hadoop3.tgz
    sudo mv spark-${SPARK_VERSION}-bin-hadoop3 $SPARK_DIR
    sudo chown -R "$USER_NAME" "$SPARK_DIR"
    
    rm spark-${SPARK_VERSION}-bin-hadoop3.tgz
    cd - > /dev/null
fi

sudo -H -u "$USER_NAME" bash -c "
    # Setup Venv
    if [ ! -d /local/repository/.venv ]; then
        echo 'Creating venv...' | sudo tee -a /local/repository/setup.log
        python3 -m venv /local/repository/.venv
        source /local/repository/.venv/bin/activate
        pip install --upgrade pip
        pip install -r /local/repository/requirements.txt
    else
        source /local/repository/.venv/bin/activate
    fi

    echo 'Downloading Data...' | sudo tee -a /local/repository/setup.log
    
    export KAGGLE_CONFIG_DIR=/local/repository/.kaggle_config
    export XDG_CACHE_HOME=/local/repository/.cache
    mkdir -p /local/repository/.kaggle_config
    mkdir -p /local/repository/.cache

    export PYTHONPATH=/local/repository
    python3 /local/repository/common/download_utils.py
"

echo "Setup complete on $(hostname)." | sudo tee -a /local/repository/setup.log