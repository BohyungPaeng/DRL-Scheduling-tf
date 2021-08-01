import argparse, datetime, os, sys, csv
import tensorflow as tf
import numpy as np
from collections import defaultdict

from agent.trainer import Trainer
from env.simul_pms import PMSSim
from config import *
from utils.util import *
from utils.visualize.logger import instance_log

if args.viz:
    from utils.visualize.viz_state import VizState
    viz = VizState(file_path=args.summary_dir,
                   colormap=[ list(range(12)), list(range(12,20)) ],
                   imshow=True)
                   # colormap=[[0, 1], list(range(2, 7)), list(range(7, 15)), list(range(15,19)), [19,20,21], list(range(22,40))])

class TestRecord(object):
    def __init__(self, save_dir='C:/results', filename='test_performance', format = 'csv'):
        self.save_dir = save_dir
        self.filename = filename+'.'+format
        # self.log_columns = list()
        self.KPIs = defaultdict(list)
    def add_performance(self, column, value):
        self.KPIs[column].append(value)
    def __len__(self):
        if len(self.KPIs.keys())==0: return 0
        firstkey = list(self.KPIs.keys())[0]
        return len(self.KPIs[firstkey])
    def summary_performance(self, msg=None):
        if len(self) == 0: return
        idx = len(self)-1
        temp = list()
        for c, v in self.KPIs.items():
            if type(c) is str: continue
            temp.append(float(v[idx]))
        avg = np.mean(temp)
        self.add_performance('avg', avg)
        if msg is None: print('average : ' , avg)
        else: print(msg, avg)
    def summary_performance_last10(self):
        if len(self) == 0: return
        best_dict_last = defaultdict(list)
        for idx in range(len(self)-1):
            for c, v in self.KPIs.items():
                if type(c) is str: continue
                temp_performance = float(v[idx])
                if idx < len(self)-2: best_dict_last[c].append(temp_performance)
        for c, temp_list in best_dict_last.items():
            self.add_performance(c,np.max(temp_list))
        self.summary_performance('10last average : ')

    def summary_performance_whole(self):
        if len(self) == 0: return
        best_dict_last = defaultdict(list)
        best_dict_sample = defaultdict(list)
        for idx in range(len(self)):
            for c, v in self.KPIs.items():
                if type(c) is str: continue
                temp_performance = float(v[idx])
                if idx >= len(self)-10: best_dict_last[c].append(temp_performance)
                if idx == len(self)-1 or idx<9: best_dict_sample[c].append(temp_performance)
        for c, temp_list in best_dict_sample.items():
            self.add_performance(c,np.max(temp_list))
        self.summary_performance('sample average : ')
        for c, temp_list in best_dict_last.items():
            self.add_performance(c,np.max(temp_list))
        self.summary_performance('10last average : ')

    def write(self):
        with open(os.path.join(self.save_dir, self.filename), mode='a', newline='\n') as f:
            f_writer = csv.DictWriter(f, fieldnames=self.KPIs.keys())
            temp_dict = dict()
            for column in self.KPIs.keys():
                temp_dict.update({column: column})
            f_writer.writerow(temp_dict)
            for row in range(len(self)):
                temp_dict.clear()
                for column in self.KPIs.keys():
                    temp_dict.update({column: self.KPIs[column][row]})
                f_writer.writerow(temp_dict)
            f.close()

def test_logic_seed_pkg(logic):
    for did in [3]:
        record = instance_log(args.gantt_dir, 'test_logic_instances_{}'.format(args.timestamp))
        rslt = TestRecord(save_dir=args.summary_dir, filename='test_performance_logic')
        env = PMSSim(config_load=None, record=record, opt_mix='geomsort', data_name=args.DATASET[did])

        seed_list = range(10000,10030)
        ST_TIME = datetime.datetime.now()

        rslt.add_performance(str(did)+'logic', logic)
        for seed in seed_list:
            env.set_random_seed(seed)
            env.reset()

            done = False
            observe = env.observe()
            # run experiment
            total_reward = 0
            while not done:
                # interact with environment
                observe, reward, done = env.step_logic(pol=logic)
                total_reward += reward

            rslt.add_performance(seed, total_reward)

        elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
        print('elapsed time per problem: ', elapsed_time/len(seed_list))
        rslt.summary_performance()
        rslt.write()

def test_logic_seed():
    # logics = ['ssu', 'seq_needs', 'seq_lst', 'seq_spt', 'wcovert']
    logics = ['ssu', 'lst', 'wcovert', 'seq_lst']
    # logics = ['ssu', 'seq_needs'] # for stoch
    for did in [args.did]:
        record = instance_log(args.gantt_dir, 'test_logic_instances_{}'.format(args.timestamp))
        rslt = TestRecord(save_dir=args.summary_dir, filename='test_performance_logic')
        env = PMSSim(config_load=None, log=record, opt_mix='geomsort', data_name=args.DATASET[did])

        seed_list = list(range(300,330)) * 1
        args.bucket = 0
        args.auxin_dim=0
        for logic in logics:

            ST_TIME = datetime.datetime.now()

            rslt.add_performance(str(did)+'logic', logic)
            cnt = 0
            for seed in seed_list:
                env.set_random_seed(seed)
                env.reset()

                done = False
                observe = env.observe()
                # run experiment
                total_reward = 0
                record.clearInfo()
                while not done:
                    # interact with environment
                    observe, reward, done = env.step_logic(pol=logic)
                    total_reward += reward
                    record.saveInfo()
                # record.fileWrite(cnt//30*30+seed, logic)
                # rslt.add_performance(cnt//30*30+seed, total_reward)
                record.fileWrite(cnt, logic)
                rslt.add_performance(cnt, total_reward)
                cnt += 1

            elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
            print('elapsed time per problem: ', elapsed_time/len(seed_list))
            rslt.summary_performance()
        rslt.write()

def test_logic(config: str):
    # logics = ['ssu', 'seq_needs', 'atcs', 'seq_lst', 'seq_spt', 'wcovert']
    # logics = ['seq_needs']
    logics = ['ssu', 'seq_needs', 'lst', 'spt', 'wcovert']
    record = instance_log(args.gantt_dir, 'test_logic_instances_{}'.format(args.timestamp))
    if len(config)<4:
        env = PMSSim(config_load=None, log=record, opt_mix='geomsort')
        env.set_random_seed(int(config))
    else:
        env = PMSSim(config_load=config, log=record)
        env.set_random_seed(0)
    ST_TIME = datetime.datetime.now()

    args.bucket=0
    for logic in logics:
        record.clearInfo()
        env.reset()
        done = False
        observe = env.observe()
        # run experiment
        total_reward = 0
        while not done:
            # interact with environment
            observe, reward, done = env.step_logic(pol=logic)
            total_reward += reward
            record.saveInfo()

        elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
        L_avg = 0
        kpi = record.get_KPI()
        util, cmax, total_setup_time, avg_satisfaction_rate = kpi.get_util(), kpi.get_makespan(), kpi.get_total_setup(), kpi.get_throughput_rate()

        performance_msg = 'Run: %s(%s) / Util: %.5f / Reward: %5.2f / cQ : %5.2f(real:%5.2f) / Lot Choice: %d(Call %d) / Setup : %d / Setup Time : %.2f hour / Makespan : %s / Elapsed t: %5.2f sec / loss: %3.5f / Demand Rate : %.5f(%.2f)' % (
            logic, config, util, total_reward, 0, 0, env.decision_number, 0, env.setup_cnt, total_setup_time//60, str(cmax), elapsed_time, L_avg, avg_satisfaction_rate, kpi.get_total_tardiness())
        print(performance_msg)
        return elapsed_time

        # record.fileWrite(logic, 'viewer')

def test_online(agentObj, env, episode, showFlag=False):
    args.is_train = False
    # env = SimEnvSim(agentObj.record)

    agentObj.SetEpisode(episode)
    if args.viz: viz.new_episode()
    ST_TIME = datetime.datetime.now()

    env.reset()
    done = False
    observe = env.observe(args.oopt)
    # run experiment
    while not done:
        state, action, curr_time = agentObj.get_action(observe)
        act_vec = np.zeros([1, args.action_dim])
        act_vec[0, action] = 1
        # interact with environment
        if args.bucket == 0 or (isinstance(env, TDSim) and 'learner_decision_' in DARTSPolicy):
            observe, reward, done = env.step(action)
        else:
            observe, reward, done = env.step_bucket(action)
        if args.viz:
            if type(args.state_dim) is int or len(args.state_dim)<=2:
                viz.viz_img_2d(state['state'], prod=args.action_dim)
            else:
                viz.viz_img_3d(state['state'])
        # agentObj.remember(state, act_vec, observe, reward, done)
        agentObj.remember_record(state, act_vec, reward, done)

    if showFlag is False:
        args.is_train = True
        if isinstance(env, PMSSim):
            return env.get_mean_tardiness(env.get_tardiness_hour(agentObj.reward_total))
        elif isinstance(env, TDSim):
            return agentObj.reward_total
    elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()

    if isinstance(env, PMSSim): performance = get_performance(episode, agentObj, env, elapsed_time, True)
    else: performance = get_performance_pkg(episode, agentObj, env, elapsed_time)
    # L_avg = 0
    # if len(agentObj.loss_history) > 0: L_avg = np.mean(agentObj.loss_history)
    agentObj.record.fileWrite(episode, 'test_viewer')
    kpi = agentObj.record.get_KPI()
    # util=kpi.get_util();
    # cmax_str = '%02dday %dmin' % (kpi.get_makespan() // (24 * 60), (kpi.get_makespan() % (24 * 60)))
    # performance_msg = 'TEST: %07d(%d) / Util: %.5f / cR: %5.2f / cQ : %5.2f(real:%5.2f) / Setup : %d / Setup Time : %.2f ' \
    #                   'hour / cmax : %s / loss: %3.5f / Demand Rate : %.5f / Elapsed t: %5.2f sec / Decision: %d(Call %d) ' % (
    #                       episode, 0, kpi.get_util(), agentObj.reward_total, agentObj.cumQ, agentObj.cumV_real,
    #                       env.setup_cnt, kpi.get_total_setup(), cmax_str, L_avg, kpi.get_throughput_rate(),
    #                       elapsed_time, agentObj.getDecisionNum(), agentObj.trigger_cnt)
    # print(performance_msg)
    # performance = ['%07d' % episode,
    #                '%d' % 0,
    #                '%.5f' % kpi.get_util(),
    #                '%5.2f' % agentObj.reward_total,
    #                '%5.2f' % agentObj.cumQ,
    #                '%5.2f' % agentObj.cumV_real,
    #                '%d' % agentObj.getDecisionNum(),
    #                '%d' % env.setup_cnt,
    #                '%d' % int(kpi.get_total_setup() / 3600),
    #                '%s' % str(kpi.get_makespan()),
    #                '%5.2f' % elapsed_time,
    #                '%3.5f' % L_avg,
    #                '%.5f' % kpi.get_throughput_rate()]
    # agentObj.writeSummary()
    perform_summary = tf.Summary()
    perform_summary.value.add(simple_value=agentObj.reward_total, node_name="reward/test_cR", tag="reward/test_cR")
    perform_summary.value.add(simple_value=agentObj.cumQ, node_name="reward/test_cQ", tag="reward/test_cQ")
    perform_summary.value.add(simple_value=agentObj.cumV_real, node_name="reward/test_cV_real", tag="reward/test_cV_real")
    perform_summary.value.add(simple_value=agentObj.setupNum, node_name="KPI/test_nst", tag="KPI/test_nst")
    perform_summary.value.add(simple_value=kpi.get_total_setup() / 60, node_name="KPI/test_tst", tag="KPI/test_tst")
    perform_summary.value.add(simple_value=kpi.get_makespan(), node_name="KPI/test_cmax", tag="KPI/test_cmax")
    perform_summary.value.add(simple_value=kpi.get_throughput_rate(), node_name="KPI/test_thr", tag="KPI/test_thr")
    perform_summary.value.add(simple_value=kpi.get_total_tardiness() / 60, node_name="KPI/total_tard", tag="KPI/total_tard")
    if agentObj.getSummary():
        agentObj.getSummary().add_summary(perform_summary, episode)
        agentObj.getSummary().flush()

    total_time = datetime.datetime.now() - ST_TIME
    # performances.writeSummary()
    print("Online test elapsed time: {}\t hour: {} sec ".format(episode, total_time))
    args.is_train = True
    return performance

def test_procedure(tf_config, key=None, best_model_idx=None):
    MAX_EPISODE = 1
    episode = 0
    args.is_train = False
    exp_idx = args.eid
    if key is None: key = args.key
    rslt = TestRecord(save_dir=args.summary_dir, filename='test_performance' + str(key))
    with tf.Session(config=tf_config) as sess:
        FIRST_ST_TIME = datetime.datetime.now()
        print('Activate Neural network start ...')
        global_step = tf.Variable(0, trainable=False)
        lr = args.lr
        if 'upm' in args.oopt:
            agentObj = Trainer(sess, tf.train.GradientDescentOptimizer(lr),
                               global_step=global_step, use_hist=False, exp_idx=exp_idx)
        elif 'fab' in args.oopt:
            agentObj = Trainer(sess, tf.train.AdamOptimizer(lr),
                               global_step=global_step, use_hist=False, exp_idx=exp_idx)
        else:
            agentObj = Trainer(sess, tf.train.RMSPropOptimizer(lr, 0.99, 0.0, 1e-6),
                               global_step=global_step, use_hist=False, exp_idx=exp_idx)
        sess.run(tf.global_variables_initializer())

        # config_list = call_config_list()
        config_list = list(range(300, 330))
        # config_list=[300]
        # config_list.append(args.config_load)
        # config_list = [args.config_load]#[485]

        saver = tf.train.Saver(max_to_keep=args.max_episode)

        model_files = os.listdir(args.model_dir)
        model_files.sort()
        length = len(model_files)
        if best_model_idx is not None:
            best_model_file = model_files[best_model_idx]
        model_files = [model_files[k] for k in range(length) if k >= length - 10]  # last 10 selection
        # model_files = model_files[-1:]
        test_did = list()
        if best_model_idx is not None:  # automatic test procedure
            model_files.append(best_model_file)
            if args.did == 0 or args.did == 4:
                # add DID: te_tau, te_eta
                test_did.extend([args.did + 1, args.did + 2, args.did + 3])
            # only add te_base
            elif args.did < 4:
                test_did.append(0)
            elif args.did < 8:
                test_did.append(4)
            else: # pilot large scale
                test_did.extend([0,4])
            # add DID: te_Nm
            if args.did >= 4:
                test_did.append(args.did - 4)
            else:
                test_did.append(args.did + 4)
            test_did.append(100)
        else:  # manual test procedure
            # test_did = [0, 3, 4, 7]
            # test_did = [99]
            test_did = [4,5,6,7,100]
        summary_str = ''
        # test_did = [10]
        print("CHECK LENGTH", args.model_dir, len(model_files))
        for data_idx in test_did:
            # print('START', args.DATASET[data_idx])
            if data_idx == 100:
                env = PMSSim(config_load=None, log=agentObj.record, opt_mix='geomsort',data_name=args.DATASET[args.did])
            else:
                env = PMSSim(config_load=None, log=agentObj.record, opt_mix='geomsort', data_name=args.DATASET[data_idx])
            elapsed_total = 0
            for model_file_name in model_files:
                if args.is_train is False:
                    # model_saved_dir = os.path.join(os.curdir, 'results', args.key)
                    # model_file_name = os.listdir(os.path.join(model_saved_dir, 'models'))[0]
                    model_dir = '{}/{}/'.format(args.model_dir, model_file_name)  # str((episode)*freq_save))
                    restore(sess, model_dir, saver)
                    rslt.add_performance('models', 'DS{}_{}'.format(data_idx, str(model_file_name)))
                for config_load in config_list:
                    episode += 1
                    agentObj.SetEpisode(episode)
                    ST_TIME = datetime.datetime.now()

                    if args.env == 'pms':
                        if type(config_load) == int:
                            env.set_random_seed(config_load)
                        else:
                            env = PMSSim(config_load=config_load, log=agentObj.record)
                    elif args.env == 'pkg':
                        from utils.problemIO.problem_reader import ProblemReaderDB
                        pr = ProblemReaderDB("problemSet_PCG_darts_bh")
                        pi = pr.generateProblem(1, False)
                        pi.twistInTarget(0.1, 0.1)
                        # pi.setInTarget('-SDP_01 16000 22000 27000 -2MCP_01 27000 21000 12000 -3MCP_01 15000 6000 9000')  # 148
                        env = TDSim(pi, agentObj.record)
                    env.reset()
                    done = False
                    observe = env.observe(args.oopt)
                    # run experiment
                    while not done:
                        state, action, curr_time = agentObj.get_action(observe)
                        act_vec = np.zeros([1, args.action_dim])
                        act_vec[0, action] = 1
                        # interact with environment
                        if args.bucket == 0 or (isinstance(env, TDSim) and 'learner_decision_' in DARTSPolicy):
                            observe, reward, done = env.step(action)
                        else:
                            observe, reward, done = env.step_bucket(action)
                        agentObj.remember_record(state, act_vec, reward, done)  # test에서는 불필요

                    elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
                    elapsed_total += elapsed_time
                    if isinstance(env, PMSSim):
                        performance = get_performance(episode, agentObj, env, elapsed_time, True)
                    else:
                        performance = get_performance_pkg(episode, agentObj, env, elapsed_time)
                    rslt.add_performance(column=config_load, value=performance[6])
                    agentObj.writeSummary()
                    if True:  agentObj.record.fileWrite(episode, 'viewer')
                print('average elapsed time: ', elapsed_total / len(config_list))
                rslt.summary_performance()
            if best_model_idx is None:
                rslt.add_performance('models', 'DS{}_last10'.format(data_idx))
                rslt.summary_performance_last10()
                # rslt.summary_performance_whole()
                # rslt.add_performance('models', 'DS{}_sample10'.format(data_idx))
                # rslt.add_performance('models', 'DS{}_last10'.format(data_idx))
            else:
                rslt.add_performance('models', 'DS{}_last10'.format(data_idx))
                rslt.summary_performance_last10()
                print(rslt.KPIs['avg'])
                best_avg = rslt.KPIs['avg'][-2]
                last10_avg = rslt.KPIs['avg'][-1]
                summary_str += '{:.3f}|{:.3f},'.format(best_avg,last10_avg)
                # rslt_list.extend(rslt.KPIs['avg'][-2:])
                # print('Final results print', rslt_list)

            rslt.write()
            rslt.KPIs.clear()
        total_time = datetime.datetime.now() - FIRST_ST_TIME
        # performances.writeSummary()
        print("Total elapsed time: {}\t hour: {} sec ".format(MAX_EPISODE, total_time))
        sess.close()
        return summary_str

def test_model_multiprocesser(tf_config, key=None):
    MAX_EPISODE = 1
    episode = 0
    args.is_train = False
    exp_idx = args.eid
    if key is None: key = args.key
    rslt = TestRecord(save_dir=args.summary_dir,filename='test_performance'+str(key))
    with tf.Session(config=tf_config) as sess:
        FIRST_ST_TIME = datetime.datetime.now()

        print('Activate Neural network start ...')
        global_step = tf.Variable(0, trainable=False)
        lr = args.lr
        if 'upm' in args.oopt:
            agentObj = Trainer(sess, tf.train.GradientDescentOptimizer(lr),
                               global_step=global_step, use_hist=False, exp_idx=exp_idx)
        elif 'fab' in args.oopt:
            agentObj = Trainer(sess, tf.train.AdamOptimizer(lr),
                               global_step=global_step, use_hist=False, exp_idx=exp_idx)
        else:
            agentObj = Trainer(sess, tf.train.RMSPropOptimizer(lr, 0.99, 0.0, 1e-6),
                           global_step=global_step, use_hist=False, exp_idx=exp_idx)
        sess.run(tf.global_variables_initializer())

        # config_list = call_config_list()
        config_list = list(range(300, 330)) * 1
        # config_list = list(range(300, 330)) * 30
        # config_list=[300]
        # config_list.append(args.config_load)
        # config_list = [args.config_load]#[485]

        saver = tf.train.Saver(max_to_keep=args.max_episode)
        if args.is_train is False:
            freq_save = args.save_freq
        # model_saved_dir = args.save_dir + key
        # model_files = os.listdir(os.path.join(model_saved_dir, 'models'))
        model_files= os.listdir(args.model_dir)
        model_files.sort()
        length = len(model_files)
        # model_files = [model_files[k] for k in range(length) if (k+1) % (length/10) == 0 or k>=length-10]
        model_files = [model_files[k] for k in range(length) if k>=length-10] # last 10 selection
        # model_files = model_files[-1:]
        print("CHECK LENGTH", args.model_dir, len(model_files))
        for data_idx in [4]:
            print('START', args.DATASET[data_idx])
            env = PMSSim(config_load=None, log=agentObj.record, opt_mix='geomsort', data_name=args.DATASET[data_idx])
            elapsed_total = 0
            for model_file_name in model_files:
                episode = 0
                if args.is_train is False:
                    # model_saved_dir = os.path.join(os.curdir, 'results', args.key)
                    # model_file_name = os.listdir(os.path.join(model_saved_dir, 'models'))[0]
                    model_dir = '{}/{}/'.format(args.model_dir, model_file_name)  # str((episode)*freq_save))
                    restore(sess, model_dir, saver)
                    rslt.add_performance('models','DS{}_{}'.format(data_idx,str(model_file_name)))
                for config_load in config_list:
                    agentObj.SetEpisode(episode)
                    ST_TIME = datetime.datetime.now()

                    if args.env == 'pms':
                        if type(config_load) == int:
                            env.set_random_seed(config_load)
                        else:
                            env = PMSSim(config_load=config_load, log=agentObj.record)
                    elif args.env == 'pkg':
                        from utils.problemIO.problem_reader import ProblemReaderDB
                        pr = ProblemReaderDB("problemSet_PCG_darts_bh")
                        pi = pr.generateProblem(1, False)
                        pi.twistInTarget(0.1, 0.1)
                        # pi.setInTarget('-SDP_01 16000 22000 27000 -2MCP_01 27000 21000 12000 -3MCP_01 15000 6000 9000')  # 148
                        env = TDSim(pi, agentObj.record)
                    env.reset()
                    done = False
                    observe = env.observe(args.oopt)
                    # run experiment
                    while not done:
                        state, action, curr_time = agentObj.get_action(observe)
                        act_vec = np.zeros([1, args.action_dim])
                        act_vec[0, action] = 1
                        # interact with environment

                        if args.bucket == 0 or (isinstance(env, TDSim) and 'learner_decision_' in DARTSPolicy):
                            observe, reward, done = env.step(action)
                        else:
                            observe, reward, done = env.step_bucket(action)
                        agentObj.remember_record(state, act_vec, reward, done)  # test에서는 불필요

                    elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
                    elapsed_total += elapsed_time
                    if isinstance(env, PMSSim):
                        performance = get_performance(episode, agentObj, env, elapsed_time, True)
                    else:
                        performance = get_performance_pkg(episode, agentObj, env, elapsed_time)
                    # rslt.add_performance(column=episode//30*30+config_load, value=performance[6])
                    rslt.add_performance(column=episode, value=performance[6])
                    agentObj.writeSummary()
                    if True:  agentObj.record.fileWrite(episode, 'viewer')
                    episode += 1
                print('average elapsed time: ', elapsed_total/len(config_list))
                rslt.summary_performance()
            rslt.summary_performance_whole()
            rslt.add_performance('models', 'DS{}_sample10'.format(data_idx))
            rslt.add_performance('models', 'DS{}_last10'.format(data_idx))
            rslt.write()
            rslt.KPIs.clear()
        total_time = datetime.datetime.now() - FIRST_ST_TIME
        # performances.writeSummary()
        print("Total elapsed time: {}\t hour: {} sec ".format(MAX_EPISODE, total_time))
        sess.close()
def test_model_singleprocesser(idx: int, tf_config=None, config_load=None):
    rslt = TestRecord(save_dir=args.summary_dir, filename='test_performance' + str(idx))
    with tf.Session(config=tf_config) as sess:
        episode = 0
        args.is_train=False
        FIRST_ST_TIME = datetime.datetime.now()
        print('Activate Neural network start ...')
        global_step = tf.Variable(0, trainable=False)
        agentObj = Trainer(sess, tf.train.RMSPropOptimizer(args.lr, 0.99, 0.0, 1e-6), global_step, use_hist=False)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=args.max_episode)
        model_saved_dir = 'D:\BH-PAPER/2105_PKG\mixinf_d4f9_da_BLnoswap' # args.save_dir # args.save_dir + args.key
        if config_load is None: config_load = args.config_load

        model_files = os.listdir(os.path.join(model_saved_dir, 'models'))
        model_files.sort()
        model_files = model_files[-10:]
        for model_file_name in model_files:
            episode += 1
            if args.is_train is False:
                # model_saved_dir = os.path.join(os.curdir, 'results', args.key)
                # model_file_name = os.listdir(os.path.join(model_saved_dir, 'models'))[0]
                model_dir = '{}/models/{}/'.format(model_saved_dir, model_file_name) #str((episode)*freq_save))
                restore(sess, model_dir, saver)
                rslt.add_performance('models','singleTDS_{}'.format(str(model_file_name)))
            if args.env == 'pms':
                env = PMSSim(config_load=config_load, log=agentObj.record)
            elif args.env == 'pkg':
                from utils.problemIO.problem_reader import ProblemReaderDB
                from main_pkg import set_problem
                pr = ProblemReaderDB("problemSet_darts_bh")
                pi = pr.generateProblem(8, False)
                set_problem(8, pi)
                # pi.twistInTarget(0.1, 0.1)
                # pi.setInTarget('-SDP_01 16000 22000 27000 -2MCP_01 27000 21000 12000 -3MCP_01 15000 6000 9000')  # 148
                env = TDSim(pi, agentObj.record)
            agentObj.SetEpisode(episode)
            ST_TIME = datetime.datetime.now()

            env.reset()
            done = False
            observe = env.observe(args.oopt)
            # run experiment
            while not done:
                state, action, curr_time = agentObj.get_action(observe)
                act_vec = np.zeros([1, args.action_dim])
                act_vec[0, action] = 1
                # interact with environment
                if args.bucket == 0:
                    observe, reward, done = env.step(action)
                else:
                    observe, reward, done = env.step_bucket(action)
                agentObj.remember(state,act_vec,observe, reward, done)  # test에서는 불필요


            elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
            if isinstance(env, PMSSim):
                performance = get_performance(episode, agentObj, env, elapsed_time, True)
            else:
                performance = get_performance_pkg(episode, agentObj, env, elapsed_time)
            # L_avg = 0
            # if len(agentObj.loss_history) > 0: L_avg = np.mean(agentObj.loss_history)
            # kpi = agentObj.record.get_KPI()
            # util, cmax, total_setup_time, avg_satisfaction_rate = kpi.get_util(), kpi.get_makespan(), kpi.get_total_setup(), kpi.get_throughput_rate()
            #
            # performance_msg = 'Run: %07d(%d) / Util: %.5f / Reward: %5.2f / cQ : %5.2f(real:%5.2f) / Lot Choice: %d(Call %d) / Setup : %d / Setup Time : %.2f hour / Makespan : %s / Elapsed t: %5.2f sec / loss: %3.5f / Demand Rate : %.5f' % (
            #         episode, idx, util, agentObj.reward_total, agentObj.cumQ, agentObj.cumV_real, agentObj.getDecisionNum(), agentObj.trigger_cnt,
            #         env.setup_cnt, total_setup_time, str(cmax), elapsed_time, L_avg, avg_satisfaction_rate)
            # print(performance_msg)
            agentObj.writeSummary()
            rslt.add_performance(column='cR', value=performance[3])
            rslt.add_performance(column='int', value=sum(list(env.counters.cumulativeInTargetCompletion.values())) / 1000.0)
            if True:  agentObj.record.fileWrite(episode, 'viewer')
            # perform_summary = tf.Summary()
            # perform_summary.value.add(simple_value=agentObj.reward_total, node_name="cR", tag="cR")
            # perform_summary.value.add(simple_value=agentObj.cumQ, node_name="cQ", tag="cQ")
            # perform_summary.value.add(simple_value=L_avg, node_name="L_episode", tag="L_episode")
            # if agentObj.getSummary() and args.is_train:
            #     agentObj.getSummary().add_summary(perform_summary, episode)
            #     agentObj.getSummary().flush()
        rslt.write()
        total_time = datetime.datetime.now() - FIRST_ST_TIME
        # performances.writeSummary()
        print("Total elapsed time: {}\t hour: {} sec ".format(len(model_files), total_time))

if __name__ == "__main__":
    # from tensorflow.python.client import device_lib
    # for x in device_lib.list_local_devices():
    #     print(x.name, x.device_type)
    """ 200229
    comment: I realized that my code had a call to an undocumented method (device_lib.list_local_devices)[
    https://github.com/tensorflow/tensorflow/blob/d42facc3cc9611f0c9722c81551a7404a0bd3f6b/tensorflow/python/client/device_lib.py#L27]
    which was creating a default session.
    device_count{} opt still doesn't work
    'GPU':0 has same effect with 'CUDA_V...':-1 
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
    tf_config = tf.ConfigProto(device_count={'GPU': 0}, gpu_options=gpu_options)
    # with tf.device('/cpu:0'):
    # for i in range(args.repeat):
    if args.test_mode == 'single_logic':
        config_list = []
        # config_list = call_config_list()
        config_list.append(args.config_load)
        for config in config_list:
            test_logic(config)
    elif args.test_mode == 'logic':
        test_logic_seed()
    elif args.test_mode == 'single':
        # for i in range(30):
        test_model_singleprocesser(1,tf_config)
    elif args.test_mode == 'multi_pool':
        # TypeError: can't pickle _thread.RLock objects (Maybe tensorflow doesn't support POOL)
        from functools import partial
        import multiprocessing as mp
        p = mp.Pool()
        with tf.Session(config=tf_config) as sess:
            aa = 1
            ab = sess
            func = partial(test_model_singleprocesser, aa, ab)
            p.map(func, ['o1_wall', 'o2_wall'])
            # test_model_multiprocesser(1,tf_config)
            p.close()
            p.join()
            sess.close()
    elif args.test_mode =='multi':
        # for key in os.listdir(args.save_dir):
        #     test_model_multiprocesser(tf_config=tf_config, key=key)
        test_model_multiprocesser(tf_config=tf_config)
    elif args.test_mode == 'manual':
        test_procedure(tf_config)
    elif args.test_mode == 'ig':
        from IG import run_env, read_sequence
        import copy
        # ig_dir = args.gantt_dir
        ig_dir = 'D:\PythonSpace\TDSA/results\mixinf_sd5_default_ri\ig_dir'
        file_list = os.listdir(ig_dir)
        rslt = TestRecord(save_dir='./', filename='test_performance_ig')
        rslt.add_performance('ig','ig')
        for file in file_list:
            if 'csv' not in file: continue
            config_load = file.split('.')[0].split('_')[-1]
            print(config_load)
            test_sequence = read_sequence(os.path.join(ig_dir,file))
            for iter in range(30):
                reward = run_env(config=str(int(config_load)+iter), sequence=copy.deepcopy(test_sequence))
                print(reward)
                rslt.add_performance(int(config_load)+30*iter,reward)
        rslt.summary_performance()
        rslt.write()