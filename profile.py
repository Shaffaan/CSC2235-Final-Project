"""A profile to run the CSC2235 project.

Instructions:
Wait for the node to be "Ready".
Your project code is in /local/repository.
To run all benchmarks, execute:
    /local/repository/run_benchmarks.sh
"""

import geni.portal as portal
import geni.rspec.pg as rspec

WISCONSIN_NODE_TYPES = [
    ('c220g5', 'Wisconsin: c220g5 (Intel Skylake, 20 core, 2 disks)'),
    ('c240g5', 'Wisconsin: c240g5 (Intel Skylake, 20 core, 2 disks, 1 P100 GPU)'),
    ('c220g1', 'Wisconsin: c220g1 (Intel Haswell, 16 core, 3 disks)'),
    ('c220g2', 'Wisconsin: c220g2 (Intel Haswell, 20 core, 3 disks)'),
    ('d7525', 'Wisconsin: d7525 (AMD EPYC Rome, 16 core, 1 A30 GPU)'),
]

portal.context.defineParameter(
    "hwtype", "Physical Node Type",
    portal.ParameterType.STRING, "c220g5",
    WISCONSIN_NODE_TYPES,
    "Select a physical node type."
)

portal.context.defineParameter(
    "node_count", "Number of Nodes",
    portal.ParameterType.INTEGER, 1,
    [(1, "Single Node"), (5, "5 Nodes (Distributed)")],
    "Select 1 for local testing or 5 for distributed benchmarking."
)

params = portal.context.bindParameters()
request = portal.context.makeRequestRSpec()

# Create a LAN for communication if we have multiple nodes
lan = request.LAN("lan")

for i in range(params.node_count):
    node_name = "node-{}".format(i)
    node = request.RawPC(node_name)
    node.hardware_type = params.hwtype
    node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU24-64-STD"
    
    # Run setup on EVERY node (installs deps and downloads data locally)
    node.addService(rspec.Execute(shell="bash", 
                                  command="/local/repository/setup.sh"))
    
    # Add interface to LAN
    iface = node.addInterface("if1")
    lan.addInterface(iface)

portal.context.printRequestRSpec()