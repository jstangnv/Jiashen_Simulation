import sys, os

#
# ACTION REQUIRED!
# CHANGE THE FOLLOWING LINE TO SHOW PATH OF LOCAL NV-sims DIRECTORY
#
dirpath_root = "/home/walsworth1/Jiashen_Simulation/NV_HyperfineDriving/"

dirpath_base = os.path.join(dirpath_root, r'BaseScripts_by_JJ')
if dirpath_base not in sys.path:
    sys.path.insert(0, dirpath_base)
