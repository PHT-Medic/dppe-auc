from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from main import *
import pickle
from collections import Counter
from FHM_approx import *

d = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'R': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'C': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'E': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'G': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'I': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


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
            return {'approx':{'enc_rx': {},
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


def data_generation(pre, label, data_path, station, run, save, APPROX):
    if APPROX:
        real_data = {
            "Pre": pre,
            "Label": label
        }
        df_real = pd.DataFrame(real_data, columns=['Pre', 'Label'])
        df_real.sort_values('Pre', ascending=False, inplace=True)
        if save:
            df_real.to_pickle(data_path + '/data_s' + str(station + 1) + '.pkl')
        return df_real
    else:
        real_data = {'Pre': pre, 'Label': label,
                     'Flag': np.random.choice([1], size=len(label))}
        df_real = pd.DataFrame(real_data, columns=['Pre', 'Label', 'Flag'])
        print("Size real: {}".format(len(df_real)))
        # TODO fake patients creation with only consiredering real values
        # tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        # values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]  # unique values
        # counts = list(df_real['Pre'].value_counts(ascending=False))
        # max_a = counts[0] + int(counts[0] * 0.1)
        # v = [max_a - counts[i] for i in range(len(counts))]  # probabilities
        # s = pd.Series(np.repeat(values[i], v[i]) for i in range(len(v)))
        # list_fakes = s.explode(ignore_index=True)
        # fakes = len(list_fakes)
        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]  # unique values
        counts = list(df_real['Pre'].value_counts(ascending=False))
        highest = counts[0] + int(counts[0] * 0.1)
        v = [highest - counts[x] for x in range(len(counts))]  # probabilities
        if sum(v) == 0:
            v = [x + 1 for x in v]
        s = pd.Series(np.repeat(values[i], v[i]) for i in range(len(v)))
        list_fakes = s.explode(ignore_index=True)
        fakes = len(list_fakes)

        fake_data = {"Pre": list_fakes,
                     "Label": np.random.choice([0], size=fakes),
                     "Flag": np.random.choice([0], size=fakes)
                     }
        df_fake = return_df(fake_data)
        print("Size fake: {}".format(len(df_fake)))
        df = [df_real, df_fake]
        merged = pd.concat(df, axis=0)
        df = merged.sample(frac=1).reset_index(drop=True)
        plot_input_data(df, df_real, df_fake, station, run, proxy=False)

        df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0
        print("Size complete: {}".format(len(df)))
        if save:
            df.to_pickle(data_path + '/data_s' + str(station + 1) + '.pkl')
        return df


if __name__ == '__main__':
    DIRECTORY = './showcase'

    SIMULATE_PUSH_PULL = False
    SAVE_KEYS = False
    SAVE_DATA = False

    print("Comparing both approaches in same run")

    MAX = 100000
    no_of_decision_points = 200

    approx_auc_diff, exact_auc_diff = [], []
    approx_total_times, exact_total_times = [], []
    total_repetitions = 10
    stations = 3

    best_time = 100
    best_diff = 10
    per = {'approx': {'samples': [],
           'flags': [],
           'total_time': []},
           'exact': {'samples': [],
                     'flags': [],
                     'total_time': []}}

    decision_points = np.linspace(0, 1, num=no_of_decision_points)[::-1]

    for repetition in range(total_repetitions):
        DATA_STORAGE_PATH = DIRECTORY + '/decrypted'
        SUB_DIR = DIRECTORY + "/" + str(repetition)
        MODEL_PATH = SUB_DIR + '/pht_results/model.pkl'
        RESULT_PATH = SUB_DIR + '/pht_results/results.pkl'

        train = Train(model=MODEL_PATH, results=RESULT_PATH)

        try:
            remove_dirs = [SUB_DIR + '/synthetic', SUB_DIR + '/encrypted',
                           SUB_DIR + '/keys', SUB_DIR + '/pht_results']
            for rm_dir in remove_dirs:
                shutil.rmtree(rm_dir)
        except Exception as e:
            pass
        directories = [SUB_DIR + '/pht_results', SUB_DIR + '/decrypted']
        if SAVE_DATA:
            directories = [SUB_DIR, SUB_DIR + '/encrypted', SUB_DIR + '/pht_results', SUB_DIR + '/decrypted']
        elif SAVE_KEYS:
            directories.append(SUB_DIR)
            directories.append(SUB_DIR + '/keys')
        for dir in directories:
            if not os.path.exists(dir):
                os.makedirs(dir)

        results = train.load_results()
        results['approx'], keys_approx = generate_keys(stations, SUB_DIR, results['approx'], SAVE_KEYS)
        results['exact'], keys_exact = generate_keys(stations, SUB_DIR, results['exact'], SAVE_KEYS)
        if SIMULATE_PUSH_PULL:
            train.save_results(results)

        times = {'approx': {"station_1": [],
                            "proxy": [],
                            "station_2": [],
                            "s_1_total": []},
                 'exact': {"station_1": [],
                           "proxy": [],
                           "station_2": [],
                           "s_1_total": []}}
        data_approx, data_exact = [], []
        for i in range(stations):
            print('Station {}'.format(i + 1))
            filename = DIRECTORY + '/data/sequences_s' + str(i + 1) + '.txt'
            data = defaultdict(list)
            model = train.load_model()

            feature_len = None
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    spt = line.split()
                    seq = spt[1].strip()
                    label = spt[2].strip()
                    item = []
                    for ch in seq:
                        item.extend(d[ch])
                    N = len(item)
                    if feature_len is None:
                        feature_len = N
                    else:
                        if feature_len != N:
                            raise ValueError
                    data[label].append(item)
            print("Number of data points for training:", {key: len(value) for (key, value) in data.items()})
            data = {'CXCR4': data['CXCR4'],
                    'CCR5': data['CCR5']}

            list_key_value = [[k, v] for k, v in data.items()]

            X = []
            Y = []
            total_s1_approx, total_s1_exact = 0, 0

            for j in range(len(data['CCR5'])):
                X.append(data['CCR5'][j])
                Y.append(1)
                try:
                    X.append(data['CXCR4'][j])
                    Y.append(0)
                except Exception:
                    pass

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=1, shuffle=True)
            print('Hold out test size for comparison: {}'.format(Counter(y_test)))

            if model is None:
                model = GradientBoostingClassifier()

            classes = np.array([0, 1])
            model.fit(x_train, y_train)

            y_pred_prob = model.predict_proba(x_test)[:, -1]

            if SIMULATE_PUSH_PULL:
                results = train.load_results()

            pre = np.array(y_pred_prob)

            label = y_test
            exact_stat_df = data_generation(pre, label, DATA_STORAGE_PATH, station=i, run=repetition, save=False, APPROX=False)
            approx_stat_df = data_generation(pre, label, DATA_STORAGE_PATH, station=i, run=repetition, save=False, APPROX=True)
            data_approx.append(approx_stat_df.copy())
            data_exact.append(exact_stat_df.copy())

            t1 = time.perf_counter()
            print('Station - DPPA-AUC protocol - Step I')
            results_approx = dppa_auc_protocol(approx_stat_df, decision_points, results["approx"], SUB_DIR, station=i + 1, max_value=MAX,
                                            save_data=SAVE_DATA, save_keys=SAVE_KEYS, keys=keys_approx)
            t2 = time.perf_counter()
            times['approx']["station_1"].append(t2 - t1)

            t1 = time.perf_counter()
            print('Station - DPPE-AUC protocol - Step I')
            results_exact = dppe_auc_protocol(exact_stat_df, results["exact"], SUB_DIR, station=i + 1, max_value=MAX,
                                              save_data=SAVE_DATA, save_keys=SAVE_KEYS, keys=keys_exact)
            t2 = time.perf_counter()
            times['exact']["station_1"].append(t2 - t1)

            total_s1_approx += times['approx']["station_1"][-1]
            total_s1_exact += times['exact']["station_1"][-1]
            train.save_model(model)

            if SIMULATE_PUSH_PULL:
                train.save_results(results)
            print('\n')
        print(f'Total exact execution time at stations - Step 1 {sum(times["exact"]["station_1"]):0.4f} seconds')
        print(f'Average exact execution time at stations - Step 1 {sum(times["exact"]["station_1"]) /  len(times["exact"]["station_1"]):0.4f}' ' seconds')
        print(f'Total approx execution time at stations - Step 1 {sum(times["approx"]["station_1"]):0.4f} seconds')
        print(f'Average approx execution time at stations - Step 1 {sum(times["approx"]["station_1"]) / len(times["approx"]["station_1"]):0.4f}' ' seconds')
        print('Starting proxy protocol')
        if SIMULATE_PUSH_PULL:
            results = train.load_results()

        times['approx']['s_1_total'].append(total_s1_approx)
        times['exact']['s_1_total'].append(total_s1_exact)
        t3 = time.perf_counter()
        approx_results = dppa_auc_proxy(SUB_DIR, results["approx"], max_value=MAX, save_keys=SAVE_KEYS, keys=keys_approx,
                                     no_dps=no_of_decision_points)
        t4 = time.perf_counter()
        times["approx"]['proxy'].append(t4 - t3)
        print(f'Approx execution time by proxy station {times["approx"]["proxy"][-1]:0.4f} seconds')

        t3 = time.perf_counter()
        exact_results = dppe_auc_proxy(SUB_DIR, results["exact"], max_value=MAX, save_keys=SAVE_KEYS, run=repetition, keys=keys_exact)
        t4 = time.perf_counter()
        times["exact"]['proxy'].append(t4 - t3)
        print(f'Exact execution time by proxy station {times["exact"]["proxy"][-1]:0.4f} seconds')
        results = {'approx': results_approx, 'exact': results_exact}
        if SIMULATE_PUSH_PULL:
            train.save_results(results)
            results = train.load_results()

        print('Station - DPPE-AUC & DPPA-AUC protocol - Step II')

        auc_gt_approx, per['approx'] = calculate_regular_auc(stations, per['approx'], DATA_STORAGE_PATH, save=False, data=data_approx, APPROX=True)
        print('Approx GT-AUC: ', auc_gt_approx)
        auc_gt_exact, per['exact'] = calculate_regular_auc(stations, per['exact'], DATA_STORAGE_PATH, save=False, data=data_exact, APPROX=False)
        print('Exact GT-AUC: ', auc_gt_exact)

        t5 = time.perf_counter()
        auc_pp_exact = pp_auc_station_final(SUB_DIR, results["exact"], SAVE_KEYS, keys_exact, APPROX=False)
        t6 = time.perf_counter()

        times['exact']['station_2'].append(t6 - t5)
        total_time_exact = times["exact"]['s_1_total'][-1] + times["exact"]['proxy'][-1] + (times["exact"]['station_2'][-1] * stations)
        print(f'Exact execution time by station - Step II {times["exact"]["station_2"][-1]:0.4f} seconds')
        print('Exact total time {}'.format(total_time_exact))
        per['exact']['total_time'].append(total_time_exact)
        exact_total_times.append(total_time_exact)
        diff_exact = auc_gt_exact - auc_pp_exact
        exact_auc_diff.append(diff_exact)

        exact_avg_diff = sum(exact_auc_diff) / len(exact_auc_diff)
        print('Exact average differences over {} runs with by {} and all {}'.format(len(exact_auc_diff),
                                                                                    exact_avg_diff, exact_auc_diff))
        exact_avg_time = sum(exact_total_times) / len(exact_total_times)
        print('Exact average time total {} and each runtime {}'.format(exact_avg_time, exact_total_times))
        print('\n')

        t5 = time.perf_counter()
        auc_pp_approx = pp_auc_station_final(SUB_DIR, results["approx"], SAVE_KEYS, keys_approx, APPROX=True)
        t6 = time.perf_counter()
        times['approx']['station_2'].append(t6 - t5)
        total_time_approx = times["approx"]['s_1_total'][-1] + times["approx"]['proxy'][-1] + (
                    times["approx"]['station_2'][-1] * stations)
        print(f'Approx execution time by station - Step II {times["approx"]["station_2"][-1]:0.4f} seconds')
        print('Approx total time {}'.format(total_time_approx))
        per['approx']['total_time'].append(total_time_approx)
        diff_approx = auc_gt_approx - auc_pp_approx
        approx_auc_diff.append(diff_approx)
        approx_total_times.append(total_time_approx)

        approx_avg_diff = sum(approx_auc_diff) / len(approx_auc_diff)
        print('Approx average differences over {} runs with by {} and all {}'.format(len(approx_auc_diff), approx_avg_diff, approx_auc_diff))
        approx_avg_time = sum(approx_total_times) / len(approx_total_times)
        print('Approx average time total {} and each runtime {}'.format(approx_avg_time, approx_total_times))

        print('Difference DPPA-AUC to GT: {} and Difference DPPE-AUC to GT: {}'.format(diff_approx, diff_exact))
        print('\n')

        print(per)
