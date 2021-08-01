import tensorflow as tf
import os, csv
import numpy as np
import pandas as pd


def get_performance(episode, agentObj, env, elapsed_time, test=False):
    L_avg = 0
    if len(agentObj.loss_history) > 0: L_avg = np.mean(agentObj.loss_history)
    kpi = agentObj.record.get_KPI()
    util, cmax, total_setup_time, avg_satisfaction_rate = kpi.get_util(), kpi.get_makespan(), kpi.get_total_setup(), kpi.get_throughput_rate()
    cmax_str = '%02dd%dmin' % (cmax // (24 * 60 * 60), (cmax % (24 * 60 * 60)) // 60)

    tardiness_hour = -kpi.get_total_tardiness()/60.0

    flag = 'Test' if test else 'Run'
    performance_msg = '%s: %07d(%s) / Util: %.5f / cR: %5.2f / cQ : %5.2f(real:%5.2f) / Setup: %d(%.2f hour)' \
                      '/ cmax: %s / loss: %3.5f / Demand: %.5f(%.2f) / Elapsed t: %5.2fsec / Decision: %d, %dstep ' % (
                          flag, episode, env.config_load, util, agentObj.reward_total, agentObj.cumQ, agentObj.cumV_real,
                          env.setup_cnt, total_setup_time/60, cmax_str, L_avg, avg_satisfaction_rate, tardiness_hour,
                          elapsed_time, agentObj.getDecisionNum(), agentObj.trigger_cnt)
    print(performance_msg)
    performance = ['%07d' % episode,
                   '%s' % env.config_load,
                   '%.5f' % util,
                   '%5.2f' % agentObj.reward_total,
                   '%5.2f' % agentObj.cumQ,
                   '%5.2f' % agentObj.cumV_real,
                   '%5.2f' % env.get_mean_tardiness(tardiness_hour),
                   '%d' % agentObj.getDecisionNum(),
                   '%d' % env.setup_cnt,
                   '%d' % int(total_setup_time / 3600),
                   '%s' % str(cmax),
                   '%5.2f' % elapsed_time,
                   '%3.5f' % L_avg,
                   '%.5f' % avg_satisfaction_rate]
    return performance
def call_config_list_dir(dir_name):
    file_list=os.listdir('env/config/'+dir_name)
    config_list = list()
    for name in file_list:
        config_str = '_'.join(name.split('_')[:-1])
        config_list.append(dir_name+'/'+config_str)
    return sorted(list(set(config_list)))
def call_config_list(multiprocess=None, is_valid=False):
    if is_valid:
        wip_list = ['w4']#, 'wall', 'w2']
        slice_type = ['o']  # ,'e']
        slice_idx = list(range(1, 6))
    else:
        wip_list = ['w4', 'wall', 'w2']
        slice_type = ['oo']  # ,'e']
        slice_idx = list(range(1, 16))
    config_list = list()
    for config_wip in wip_list:
        config_sublist = list()
        for t in slice_type:
            for i in slice_idx:
                config_slice = '{}{}_{}'.format(t,i,config_wip)
                config_sublist.append(config_slice)
        if multiprocess is None: config_list.extend(config_sublist)
        else: config_list.append(config_sublist)

    return config_list

def save(sess, save_dir, saver):
    """
    Save all model parameters and replay memory to self.save_dir folder.
    The save_path should be models/env_name/name_of_agent.
    """
    # path to the checkpoint name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, "AkC")
    print("Saving the model to path %s" % path)
    # self.memory.save(self.save_dir)
    print(saver.save(sess, path))
    print("Done saving!")


def restore(sess, save_dir, saver):
    """
    Restore model parameters and replay memory from self.save_dir folder.
    The name of the folder should be models/env_name
    """
    # TODO: Need to find a better way to store memory data. Storing all states is not efficient.
    ckpts = tf.train.get_checkpoint_state(save_dir)
    if ckpts and ckpts.model_checkpoint_path:
        ckpt = ckpts.model_checkpoint_path
        saver.restore(sess, ckpt)
        # self.memory.restore(save_dir)
        print("Successfully load the model %s" % ckpt)
        # print("Memory size is:")
        # self.memory.size()
    else:
        print("Model Restore Failed %s" % save_dir)
