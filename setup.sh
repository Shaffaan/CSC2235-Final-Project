#!/bin/bash
echo "Setup script is running as $(whoami) on $(hostname)..." | sudo tee -a /local/repository/setup.log

# 1. Install System Dependencies
sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3-venv python3-pip openjdk-17-jdk-headless wget

# 2. Identify User
USER_NAME=$(stat -c '%U' /local/repository)
USER_HOME=$(getent passwd "$USER_NAME" | cut -d: -f6)

if [ -z "$USER_NAME" ] || [ -z "$USER_HOME" ]; then
    echo "Could not find project owner. Exiting." | sudo tee -a /local/repository/setup.log
    exit 1
fi

# 3. Install Standalone Spark Binaries (Required for Cluster Management)
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

# 4. Setup Python Environment and Download Data
sudo -u "$USER_NAME" bash -c "
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

    # Trigger Data Download on THIS node
    echo 'Downloading Aircheck Data locally...' | sudo tee -a /local/repository/setup.log
    # We execute the module as a script to trigger the download logic
    export PYTHONPATH=/local/repository
    python3 /local/repository/common/download_utils.py
"

echo "Setup complete on $(hostname)." | sudo tee -a /local/repository/setup.log