import argparse, datetime, numpy, os, sys, csv, random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

from agent.trainer import Trainer
from env.simul_pms import PMSSim
from test import test_online, test_procedure, call_config_list, viz_factory
from config import *
from utils.util import *
import pandas as pd

class PerformanceRecord(object):
    def __init__(self, save_dir='C:/results', filename='performance', format = 'csv'):
        self.save_dir = save_dir
        self.filename = filename+'.'+format
        self.log_columns = ['Episode', 'idx', 'Util', 'Reward', 'cQ', 'real_cV', 'Lot Choice', 'Setup', 'Setup Time',
                            'Makespan', 'Time', 'Loss', 'Satisfaction Rate']
        # self.exp_key = vars(args)['key']
        with open(os.path.join(self.save_dir, self.filename), mode='a', newline='\n') as f:
            csv.writer(f).writerow(vars(args))
            csv.writer(f).writerow(vars(args).values())
            f_writer = csv.DictWriter(f, fieldnames=self.log_columns)
            temp_dict = dict()
            for column in self.log_columns:
                temp_dict.update({column: column})
            f_writer.writerow(temp_dict)
            f.close()
        self.KPIs = []
        self.best = None
        self.current_episode=0
        self.best_episodes=[]
        self.best_model_idx_valid = 0
        self.best_dict=dict()
    def update_best(self, key, value):
        if key not in self.best_dict:
            self.best_dict[key]=value
        elif self.best_dict[key] < value:
            self.best_dict[key] = value
            if key == 'single':
                self.best_model_idx_valid = int(self.current_episode) // args.save_freq - 1
        else:
            pass
    def get_best(self, key, listFlag=True):
        if listFlag: # key:valid -> get average of every best results on validation problems 'valid##'
            best_list = [self.best_dict[k] for k in self.best_dict.keys() if key in k]
            return sum(best_list) / len(best_list)
        else:
            return self.best_dict[key]

    def write(self, performance, stat, reverse=True):
        with open(os.path.join(self.save_dir, self.filename), mode='a', newline='\n') as f:
            f_writer = csv.DictWriter(f, fieldnames=self.log_columns)
            temp_dict = dict()
            for i, column in enumerate(self.log_columns):
                temp_dict.update({column: performance[i]})
            f_writer.writerow(temp_dict)
            f.close()
        # 0: epi, 1:dataid, 2:util, 3:cR, 4:cQ, 5:cV, 6:total_tardiness
        # 7:decisions, 8:setupnum, 9:setup time, 10:makespan, 11: elapsed_time, 12:L_avg
        kpi = float(performance[6])
        self.current_episode = performance[0]
        if stat: self.KPIs.append(kpi)
        if self.best is None or (kpi>self.best if reverse else kpi<self.best):
            self.best = kpi
            self.best_episodes = [self.current_episode] # best episode for training
            return True
        if kpi==self.best:
            self.best_episodes.append(self.current_episode)
        return False
    def writeSummary(self, exp_summary=''):
        s = pd.Series(self.KPIs)
        f = open(os.path.join(self.save_dir, self.filename), mode='a', newline='\n')
        f_report = open('./rslt_report_210313.csv', mode='a', newline='\n')
        rslt = s.describe(include='all')
        print(rslt)
        exp_msg = self.save_dir.split('/')[-1]
        for key, value in rslt.items():
            f.write(str(key)+','+str(value)+'\n')
            if key in ['mean', 'std', 'max']:
                exp_msg += ','+str(value)
        exp_msg += ','+exp_summary+'\n'
        f_report.writelines(exp_msg)
        cnt = s.value_counts()
        f.write('unique count results\n')
        for key, value in cnt.items():
            f.write(str(key)+','+str(value)+'\n')
        listmsg=''
        for episode_num in self.best_episodes:
            listmsg+=str(episode_num)+'-'
        f.write('Best KPI,{},{}'.format(str(self.best), listmsg))
        f.close()

def train(idx: int, tf_config):

    MAX_EPISODE = 100000

    """tf_config in tf2.x
    import tensorflow as tf
    tf.config.gpu.set_per_process_memory_fraction(0.75)
    tf.config.gpu.set_per_process_memory_growth(True)
    """
    with tf.Session(config=tf_config) as sess:
        FIRST_ST_TIME = datetime.datetime.now()

        print('Activate Neural network start ...')
        global_step = tf.Variable(0, trainable=False)
        # global_step = tf.Variable(tf.float32, [], name='gs')
        # lr = tf.train.exponential_decay(args.lr, global_step=global_step,
        #                                                     decay_steps=args.max_episode*25,
        #                                                     decay_rate=0.1)
        lr = args.lr
        if 'upm' in args.oopt:
            agentObj = Trainer(sess, tf.keras.optimizers.SGD(lr),
                               global_step=global_step, use_hist=args.use_hist, exp_idx=idx)
        elif 'fab' in args.oopt:
            agentObj = Trainer(sess, tf.keras.optimizers.Adam(lr),
                               global_step=global_step, use_hist=args.use_hist, exp_idx=idx)
        else:
            # agentObj = Trainer(sess, tf.keras.optimizers.RMSprop(lr, 0.99, 0.0, 1e-6),
            #                global_step=global_step, use_hist=args.use_hist, exp_idx=idx)
            agentObj = Trainer(sess, tf.train.RMSPropOptimizer(lr, 0.99, 0.0, 1e-6),
                           global_step=global_step, use_hist=args.use_hist, exp_idx=idx)
        sess.run(tf.global_variables_initializer())
        # tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=args.max_episode)
        sr = PerformanceRecord(save_dir=args.save_dir, filename='performance_{}'.format(idx), format='csv')
        if args.is_train is False:
            freq_save = args.save_freq
        model_saved_dir = os.path.join('D:', 'results', '20190831', 'buffer_neg_0')

        if idx < 5:
            valid_seed_list=list(range(300,330))
        elif idx < 300:
            envlist = []
            for config_load in call_config_list():
                envlist.append(PMSSim(config_load=config_load, log=agentObj.record))
            valid_envlist = []
            for config_load in call_config_list(is_valid=True):
                valid_envlist.append(PMSSim(config_load=config_load, log=agentObj.record))
        else:
            valid_seed_list=[]

        env = PMSSim(config_load=None, log=agentObj.record, opt_mix='geomsort', opt_inirem='random', data_name=args.DATASET[args.did]
                     # , opt_inist=['w4','w2','wall']
                     )
        valid_env = env
        # restore(sess, 'results/pretrain_rp/', saver)
        for episode in range(1, args.max_episode + 1):
            # print(sess.run(decaying_learning_rate), sess.run(global_step))
            agentObj.SetEpisode(episode)
            # if args.max_episode-episode<=1000 and args.max_episode-episode>=100: agentObj.SetEpisode(args.max_episode-episode/2) #make epsilon
            ST_TIME = datetime.datetime.now()
            if idx >= 300:
                env.set_random_seed(idx)
            else:
                # if (episode-1) % args.chg_freq==0:
                #     env = random.Random().choice(envlist)
                # env.set_random_seed(random.Random().randint(0,300))
                # env.set_random_seed(random.Random().randint(0,100))
                env.set_random_seed(episode+500)
                # if agentObj.eps == 0: env.set_random_seed(random.Random().randint(0,300))
                # env.set_random_seed((episode//3) % 300)
                # if args.max_episode-episode<1000:
                #     env = test_env
            env.reset()
            done = False
            observe = env.observe(args.oopt)
            # run experiment
            while not done:
                pre_observe, action, curr_time = agentObj.get_action(observe)
                act_vec = np.zeros([1, args.action_dim])
                act_vec[0, action] = 1
                # interact with environment
                if args.bucket == 0:
                    observe, reward, done = env.step(action)
                    agentObj.remember(pre_observe, act_vec, observe, reward, done)
                else:
                    observe, reward, done = env.step_bucket(action)
                    # if env.wall_time.curr_bucket <= 1:
                    #     agentObj.remember_record(pre_observe, act_vec, reward, done)
                    # else:
                    agentObj.remember(pre_observe, act_vec, observe, reward, done)

            elapsed_time = (datetime.datetime.now() - ST_TIME).total_seconds()
            performance = get_performance(episode, agentObj, env, elapsed_time)
            bestFlag = sr.write(performance, stat=True if episode > args.max_episode * 0.9 else False)

            exp_idx = 0
            if episode % args.save_freq == 0:
                model_dir = '{}/models/{}_{}_{:07d}/'.format(args.save_dir, str(idx), str(exp_idx), episode)
                save(sess, model_dir, saver)
                # agentObj.record.fileWrite(episode)
                perform_summary = tf.Summary()
                perform_summary.value.add(simple_value=sr.best, node_name="reward/train_bestR",tag="reward/train_bestR")
                if len(valid_seed_list) == 0: # single test
                    performance = test_online(agentObj=agentObj, env=env, episode=episode, showFlag=True)
                    reward_test = float(performance[6])
                    sr.update_best('single', reward_test)
                    perform_summary.value.add(simple_value=reward_test, node_name="reward/test", tag="reward/test")
                    if episode == args.max_episode:
                        sr.writeSummary('{:2f},{:2f}'.format(reward_test,sr.get_best('single', listFlag=False)))
                else: # normal validation
                    # if last save_freq uses single_env strategy
                    # if episode == args.max_episode:
                    if idx == -1:
                        performance = test_online(agentObj=agentObj, env=env, episode=episode, showFlag=True)
                        reward_test = float(performance[6])
                        perform_summary.value.add(simple_value=reward_test, node_name="reward/test", tag="reward/test")
                        env = valid_env
                    reward_avg = list()
                    for valid_seed in valid_seed_list:
                        env.set_random_seed(valid_seed)
                        reward_valid = test_online(agentObj=agentObj, env=env, episode=episode, showFlag=False)
                        if valid_seed % 30 == 0:
                            agentObj.record.fileWrite(episode)
                            agentObj.record.ganttWrite(
                                img_path='{}/env{}_{:07d}.png'.format(args.gantt_dir, str(idx), episode))
                        reward_avg.append(reward_valid)
                        sr.update_best('valid{}'.format(valid_seed), reward_valid)
                        if episode + args.save_freq * 10 >= args.max_episode: sr.update_best(
                            '10last{}'.format(valid_seed), reward_valid)
                        if (episode * 10) % args.max_episode == 0: sr.update_best('10sample{}'.format(valid_seed),
                                                                                  reward_valid)
                    reward_avg = sum(reward_avg) / len(reward_avg)
                    sr.update_best('single', reward_avg)
                    best_avg = sr.get_best('valid')
                    print('Validation result : ', reward_avg)
                    perform_summary.value.add(simple_value=reward_avg, node_name="reward/valid_avgR", tag="reward/valid_avgR")
                    perform_summary.value.add(simple_value=best_avg, node_name="reward/valid_bestR", tag="reward/valid_bestR")
                    viz_factory()

                if agentObj.getSummary(): agentObj.getSummary().add_summary(perform_summary, episode)
                print('Best Validation result : ', sr.best_dict)
            if bestFlag and agentObj.reward_total>181:   # FIXME : Change Time
                model_dir = '{}/best_models/{}_{}_{:07d}/'.format(args.save_dir, str(idx), str(exp_idx), episode)
                save(sess, model_dir, saver)
                if episode % args.save_freq != 0: agentObj.record.fileWrite(episode)

            if agentObj.getSummary() and args.is_train: agentObj.writeSummary()

        sess.close()

    tf.reset_default_graph()
    total_training_time = datetime.datetime.now() - FIRST_ST_TIME
    tt_hour = (total_training_time.days * 86400 + total_training_time.seconds) / 3600
    test_multi_rslt = test_procedure(tf_config=tf_config, best_model_idx=sr.best_model_idx_valid)
    sr.writeSummary('{:2f},{:2f},{:2f},{:2f},{:2f},{:2f},{}'.
                    format(reward_avg, sr.get_best('single', listFlag=False), sr.get_best('10last'),
                           sr.get_best('10sample'), best_avg, tt_hour, test_multi_rslt))
    print("Total elapsed time: {}\t hour: {} sec ".format(MAX_EPISODE, total_training_time))

if __name__ == "__main__":
    import tensorflow.compat.v1 as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333, allow_growth=True)
    config = tf.ConfigProto(device_count={'GPU': 0}, gpu_options=gpu_options)
    if args.is_train:
        # for i in range(-1, 0):
        # for i in range(args.repeat):
        # for i in range(324, 325):
            # with tf.device('/cpu:0'):
        # args.did=9
        # args.bucket = 5400
        # args.save_freq = int(20 * (args.bucket / 5400)) * 4
        # args.max_episode = 100 * args.save_freq
        train(args.eid, config)
    else:
        from test import test_model_singleprocesser
        test_model_singleprocesser(1, config)
