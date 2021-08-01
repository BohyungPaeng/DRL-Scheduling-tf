from collections import defaultdict, OrderedDict
from copy import copy
import json
import env.util_sim as util


class log_decision(object):
    def __init__(self, idx, job_id, job_type, job_due, res_id, res_type, decision_time, time_setup, time_proc, reward
                 ,action_vector=None, predicted_Q=None):
        self.idx =idx
        self.job_id=job_id
        self.job_type=job_type
        self.job_due=job_due
        self.res_id=res_id
        self.res_type=res_type
        self.decision_time=decision_time
        self.time_setup=time_setup
        self.time_proc=time_proc
        self.reward=reward

class instance_log(object):
    def __init__(self, dir, filename):
        self.rts_info = list(dict())
        self.rts_info_dict = defaultdict(dict)
        self.key_list = []
        self.filepath = '{}/{}'.format(dir, str(filename))
        self.decision_list = list()
        self.decision_list_recent = list()
        self.res_record_dict = defaultdict(list)
        self.job_record_dict = defaultdict(list)

    def __len__(self): return len(self.decision_list)

    def appendInfo(self, key, value):
        if key in self.rts_info_dict.keys():
            if self.rts_info_dict[key] > value: #Do not replace the smaller value
                return
        else:
            self.key_list.append(key)
        self.rts_info_dict[key] = value

    def append_history(self, dec: log_decision):
        self.decision_list.append(dec)
        self.decision_list_recent.append(dec)
        self.res_record_dict[dec.res_id].append(dec)
        self.job_record_dict[dec.job_id].append(dec)
        #FIXME: calculate KPI here if necessary

    def get_res_history(self, idx): return self.res_record_dict[idx]
    def get_job_history(self, idx): return self.job_record_dict[idx]
    def saveInfo(self):
        self.rts_info.append(copy(self.rts_info_dict))
        self.rts_info_dict.clear()
        self.decision_list_recent.clear()

    def clearInfo(self):
        self.rts_info.clear()
        self.key_list.clear()
        self.rts_info_dict.clear()
        self.decision_list.clear()
        self.decision_list_recent.clear()
        self.res_record_dict.clear()
        self.job_record_dict.clear()

    def get_timestamp(self, raw_time):
        result = raw_time * 1000 * 60/util.TimeUnit - 32400000
        return result

    def fileWrite(self, simul_iter, viewer_key=None):
        if len(self.rts_info) == 0:
            return
        f = open(self.filepath+'.csv', 'a')
        msg = ""
        keylist = sorted(set(self.key_list))
        for row in range(len(self.rts_info)+1):
            for col in range(len(keylist)):
                if row == 0:
                    msg += '%s,' % (keylist[col])
                else:
                    info = self.rts_info[row - 1][keylist[col]]
                    if type(info) is str:
                        msg += '%s,' % (info)
                    elif type(info) is float:
                        msg += '%f,' % (info)
                    elif type(info) is int:
                        msg += '%d,' % (info)
                    elif type(info) is bool:
                        msg += '%s,' % str(info)
                    elif type(info) is list:
                        float_flag=False
                        for temp in info:
                            if type(temp) == float:
                                float_flag=True
                        if float_flag:
                            # msg += '['
                            # for temp in info:
                            #     msg += '(%3.3f) ' % temp
                            # msg += ']'
                            msg += '%s,' % str(info)
                        else:
                            msg += '['
                            for temp in info:
                                msg += str(temp) + ' '
                            msg += ']'
                        msg += ','
                    else:
                        msg += '%s,' % str(info) #''not pre defined type,'
            msg += '\n'
        msg += '%s\n' % ('above result from iteration {}'.format(simul_iter))
        f.write(msg)
        f.close()

    def get_KPI(self):
        rslt = KPI(self.__len__())
        for decision_row in range(self.__len__()):
            decision_info = self.decision_list[decision_row]
            assert isinstance(decision_info, log_decision)
            res = str(decision_info.res_id)
            if float(decision_info.time_setup) > 0:
                st = decision_info.decision_time + decision_info.time_setup
                rslt.tst += decision_info.time_setup / util.TimeUnit
            else:
                st = decision_info.decision_time
            et = st + decision_info.time_proc
            if et > rslt.max_dict[res]:
                rslt.max_dict[res] = et
            rslt.tpt += decision_info.time_proc
            # print('process', res, decision_info['time_proc'], st, et)
            prod = decision_info.job_type
            if et <= decision_info.job_due:
                rslt.thr_dict[prod] += 1
                rslt.thr_dict_per_mach[res] += 1
            else:
                rslt.tard_dict[prod] += (et - decision_info.job_due) / util.TimeUnit

        rslt.calculate()
        return rslt

class KPI(object):
    def __init__(self, decision_num):
        self.total = decision_num
        self.thr_dict = defaultdict(int)
        self.tard_dict = defaultdict(int)
        self.thr_dict_per_mach = defaultdict(int)
        self.max_dict = defaultdict(float)
        self.nst=0
        self.tst=0 #total setup time, min
        self.tpt=0 #total processing duration, min
        self.cmax=0
        self.util=0
    def calculate(self):
        max_list = list(self.max_dict.values())
        self.cmax = max(max_list)
        self.util = self.tpt / sum(max_list)
    def get_total_setup(self):
        return self.tst
    def get_num_setup(self):
        return self.nst
    def get_makespan(self):
        return self.cmax
    def get_util(self):
        return self.util
    def get_throughput_rate(self):
        return sum(self.thr_dict.values()) / self.total
    def get_total_tardiness(self):
        return sum(self.tard_dict.values())
