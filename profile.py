"""A profile to run the CSC2235 project.

Instructions:
Wait for the node to be "Ready". This means the setup script
has finished installing pip and all Python requirements.
Your project code is in /local/repository. To run all benchmarks, execute:
    /local/repository/run_benchmarks.sh
To run a specific benchmark:
    /local/repository/run_benchmarks.sh <benchmark_name>
To run a specific benchmark with a specific framework:
    /local/repository/run_benchmarks.sh <benchmark_name> <framework_name>
For example, to run the NYC Taxi benchmark with DuckDB, execute:
    /local/repository/run_benchmarks.sh nyc_taxi duckdb
"""

import geni.portal as portal
import geni.rspec.pg as rspec

WISCONSIN_NODE_TYPES = [
    # Skylake (Original)
    ('c220g5', 'Wisconsin: c220g5 (Intel Skylake, 20 core, 2 disks)'),
    ('c240g5', 'Wisconsin: c240g5 (Intel Skylake, 20 core, 2 disks, 1 P100 GPU)'),
    
    # Haswell (Gen 1 & 2)
    ('c220g1', 'Wisconsin: c220g1 (Intel Haswell, 16 core, 3 disks)'),
    ('c240g1', 'Wisconsin: c240g1 (Intel Haswell, 16 core, 14 disks)'),
    ('c220g2', 'Wisconsin: c220g2 (Intel Haswell, 20 core, 3 disks)'),
    ('c240g2', 'Wisconsin: c240g2 (Intel Haswell, 20 core, 8 disks)'),
    
    # Broadwell (GPU node)
    ('c4130', 'Wisconsin: c4130 (Intel Broadwell, 16 core, 4 V100 GPUs)'),
    
    # Ice Lake (Gen 3)
    ('sm110p', 'Wisconsin: sm110p (Intel Ice Lake, 16 core, 5 disks)'),
    ('sm220u', 'Wisconsin: sm220u (Intel Ice Lake, 32 core, 9 disks)'),
    
    # AMD EPYC Rome (GPU nodes)
    ('d7525', 'Wisconsin: d7525 (AMD EPYC Rome, 16 core, 1 A30 GPU)'),
    ('d8545', 'Wisconsin: d8545 (AMD EPYC Rome, 48 core, 1 A100 GPU)')
]

portal.context.defineParameter(
    "hwtype", "Physical Node Type",
    portal.ParameterType.STRING, "c220g5",
    WISCONSIN_NODE_TYPES,
    "Select a physical node type from the Wisconsin cluster. If your first choice is unavailable, try another."
)

params = portal.context.bindParameters()

request = portal.context.makeRequestRSpec()

node = request.RawPC("node")

node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"

node.hardware_type = params.hwtype

node.addService(rspec.Execute(shell="bash", 
                              command="/local/repository/setup.sh"))

portal.context.printRequestRSpec()