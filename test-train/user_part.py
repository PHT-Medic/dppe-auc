import os
import numpy as np
from main import *
from paillier import *


class Train:
    def __init__(self, model=None, results=None):
        """
        :param results:
        """
        self.results = results
        self.encoded_model = model

    def load_results(self):
        """
        If a result file exists, loads the results, otherwise will return empty results.
        :return:
        """
        try:
            with open(self.results, 'rb') as results_file:
                return pickle.load(file=results_file)
        except Exception:
            return {'initial': False,
                    'proxy': False,
                    'per': {'approx': {'samples': [],
                                       'flags': [],
                                       'total_time': []},
                            'exact': {'samples': [],
                                      'flags': [],
                                      'total_time': []}
                            },
                    'times': {'approx': {"station_1": [],
                                         "proxy": [],
                                         "station_2": [],
                                         "s_1_total": []},
                              'exact': {"station_1": [],
                                        "proxy": [],
                                        "station_2": [],
                                        "s_1_total": []}},
                    'approx': {'enc_rx': {},
                               'users_rsa_pk': [],
                               'pp_auc_tables': {},
                               'encrypted_ks': [],
                               'encrypted_r1': {},  # index is used by station i
                               'encrypted_r2': {},
                               'aggregator_rsa_pk': {},
                               'aggregator_paillier_pk': {},
                               'stations_paillier_pk': {},
                               'stations_rsa_pk': {},
                               'proxy_encrypted_r_N': {},  # index 0 = r1_iN; 1 = r2_iN
                               'D1': [],
                               'D2': [],
                               'D3': [],
                               'N1': [],
                               'N2': [],
                               'N3': []
                               },
                    'exact': {'enc_rx': {},
                              'users_rsa_pk': [],
                              'enc_agg_sk_1': {},
                              'enc_agg_sk_2': {},
                              'pp_auc_tables': {},
                              'encrypted_ks': [],
                              'encrypted_r1': {},  # index is used by station i
                              'encrypted_r2': {},
                              'aggregator_rsa_pk': {},
                              'aggregator_paillier_pk': {},
                              'stations_paillier_pk': {},
                              'stations_rsa_pk': {},
                              'proxy_encrypted_r_N': {},  # index 0 = r1_iN; 1 = r2_iN
                              'D1': [],
                              'D2': [],
                              'D3': [],
                              'N1': [],
                              'N2': [],
                              'N3': []
                              }
                    }

    def save_results(self, results):
        """
        Saves the result file of the train
        :param results:
        :return:
        """
        try:
            with open(self.results, 'wb') as results_file:
                return pickle.dump(results, results_file)
        except Exception as err:
            print(err)
            raise FileNotFoundError("Result file cannot be saved")

    def save_model(self, model):
        with open(self.encoded_model, "wb") as model_file:
            pickle.dump(model, model_file)

    def load_model(self):
        try:
            with open(self.encoded_model, "rb") as model_file:
                model = pickle.load(model_file)
            print("Loading previous results")
            return model
        except Exception:
            print("No previous data")
            return None


if __name__ == '__main__':
    DIRECTORY = os.getcwd()
    #  print("Comparing both approaches in same run")
    MAX = 100000
    no_of_decision_points = 200

    os.environ["RESULT_PATH"] = '/PATH/results.pkl'
    os.environ["PRIVATE_KEY_PATH"] = "/PATH/KEY.pem"
    os.environ["PRIVATE_KEY_PASS"] = "PW"

    approx_auc_diff, exact_auc_diff = [], []
    approx_total_times, exact_total_times = [], []
    total_repetitions = 1  # 10 before

    best_time = 100
    best_diff = 10

    decision_points = np.linspace(0, 1, num=no_of_decision_points)[::-1]

    MODEL_PATH = DIRECTORY + '/pht_results/model.pkl'

    train = Train(model=MODEL_PATH, results=os.getenv("RESULT_PATH"))

    results = train.load_results()

    times = results['times']
    per = results['per']
    data_approx = results['data_approx']
    data_exact = results['data_exact']
    stations = len(times["approx"]['station_1'])
    print('Station - DPPE-AUC & DPPA-AUC protocol - Step II')

    auc_gt_approx, per['approx'] = calculate_regular_auc(1, per['approx'], data=data_approx, APPROX=True)
    print('Approx GT-AUC: ', auc_gt_approx)
    auc_gt_exact, per['exact'] = calculate_regular_auc(1, per['exact'], data=data_exact, APPROX=False)
    print('Exact GT-AUC: ', auc_gt_exact)

    t5 = time.perf_counter()
    auc_pp_exact = pp_auc_station_final(results["exact"], APPROX=False)
    t6 = time.perf_counter()
    times['approx']['station_2'].append(t6 - t5)
    times['approx']['station_2'].append(t6 - t5)
    total_time_approx = times["approx"]['s_1_total'][-1] + times["approx"]['proxy'][-1] + (
            times["approx"]['station_2'][-1] * stations)
    print(f'Approx execution time by station - Step II {times["approx"]["station_2"][-1]:0.4f} seconds')
    print('Approx total time {}'.format(total_time_approx))

    t5 = time.perf_counter()
    auc_pp_approx = pp_auc_station_final(results["approx"], APPROX=True)
    t6 = time.perf_counter()
    times['exact']['station_2'].append(t6 - t5)
    total_time_exact = times["exact"]['s_1_total'][-1] + times["exact"]['proxy'][-1] + (
                times["exact"]['station_2'][-1] * len(times["approx"]['station_1']))
    print(f'Exact execution time by User - Step II {times["exact"]["station_2"][-1]:0.4f} seconds')
    print('Exact total time {}'.format(total_time_exact))
    per['exact']['total_time'].append(total_time_exact)
    exact_total_times.append(total_time_exact)

    total_time_exact = times["exact"]['s_1_total'][-1] + times["exact"]['proxy'][-1] + (
                times["exact"]['station_2'][-1] * stations)
    print(f'Exact execution time by station - Step II {times["exact"]["station_2"][-1]:0.4f} seconds')
    print('Exact total time {}'.format(total_time_exact))
    per['exact']['total_time'].append(total_time_exact)
    exact_total_times.append(total_time_exact)
    diff_exact = auc_gt_exact - auc_pp_exact
    exact_auc_diff.append(diff_exact)

    exact_avg_diff = sum(exact_auc_diff) / len(exact_auc_diff)
    print('Exact average differences over {} runs with by {} and all {}'.format(len(exact_auc_diff),
                                                                               exact_avg_diff, exact_auc_diff))
    diff_approx = auc_gt_approx - auc_pp_approx
    approx_auc_diff.append(diff_approx)

    approx_avg_diff = sum(approx_auc_diff) / len(approx_auc_diff)
    print('Exact average differences over {} runs with by {} and all {}'.format(len(approx_auc_diff),
                                                                                approx_avg_diff, approx_auc_diff))
    # exact_avg_time = sum(exact_total_times) / len(exact_total_times)
    # print('Exact average time total {} and each runtime {}'.format(exact_avg_time, exact_total_times))
    print('\n')

    exit(0)
