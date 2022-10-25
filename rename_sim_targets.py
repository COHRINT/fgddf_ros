from rosbag import Bag

# Define remappings
remap = [['cohrint_tycho_bot_1','cohrint_tycho_bot_6'],
         ['cohrint_tycho_bot_2','cohrint_tycho_bot_7'],
         ['cohrint_tycho_bot_3','cohrint_tycho_bot_8'],
         ['cohrint_tycho_bot_4','cohrint_tycho_bot_9'],
         ['cohrint_tycho_bot_5','cohrint_tycho_bot_10']]

# Remap agent/target names
with Bag('remapped_sim_target_data.bag','w') as Y:
    for topic, msg, t in Bag('sim_target_data.bag'):
        for i in range(len(remap)):
            if msg.target == remap[i][0]:
                msg.target = remap[i][1]
        Y.write('/sim_truth_data',msg,t)