from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from main import *
import pickle
from collections import Counter

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
            return {'enc_rx': {},
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


def data_generation(pre, label, data_path, station, fake, consider_dist, save):
    real_data = {'Pre': pre, 'Label': label,
                 'Flag': np.random.choice([1], size=len(label))}
    df_real = pd.DataFrame(real_data, columns=['Pre', 'Label', 'Flag'])
    fake_data_val = int(len(pre) * fake)
    print('Fake subjects for dppe-auc {}'.format(fake_data_val))

    if consider_dist:
        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]
        prob = list(df_real['Pre'].value_counts(normalize=True, ascending=False))
        fake_data = {"Pre": random.choices(values, weights=prob, k=fake_data_val),
                     "Label": np.random.choice([0], size=fake_data_val),
                     "Flag": np.random.choice([0], size=fake_data_val)
                     }
    else:
        fake_data = {"Pre": np.random.random(size=fake_data_val),
                     "Label": np.random.choice([0], size=fake_data_val),
                     "Flag": np.random.choice([0], size=fake_data_val)
                     }
    df_fake = pd.DataFrame(fake_data, columns=['Pre', 'Label', 'Flag'])

    dfs = [df_real, df_fake]
    merged = pd.concat(dfs, axis=0)
    df = merged.sample(frac=1).reset_index(drop=True)
    # plot_input_data(df, df_real, df_fake, station)
    if save:
        df.to_pickle(data_path + '/data_s' + str(station + 1) + '.pkl')

    return df


if __name__ == '__main__':
    DIRECTORY = './showcase'

    SIMULATE_PUSH_PULL = False
    SAVE_KEYS = False
    SAVE_DATA = False
    CONSIDER_DIST = False  # consider distribution of real data for flag generation

    EXPERIMENT_1 = True

    max_values = [10, 100, 1000, 10000, 100000]
    max = 100

    size_fake = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 3]
    auc_diff = []
    total_times = []
    total_repetitions = 5
    stations = 3

    best_time = 100
    best_diff = 10
    per = {'samples': [],
           'flags': [],
           'total_time': []}

    # for run in range(len(max_values)):  # uncomment to evaluate max value
    #    max = max_values[run]
    for run in range(len(size_fake)):  # comment these two lines, if max values should be determined
        fakes = size_fake[run]
        for repetition in range(total_repetitions):
            DATA_STORAGE_PATH = DIRECTORY + '/decrypted'
            MODEL_PATH = DIRECTORY + '/pht_results/model.pkl'
            RESULT_PATH = DIRECTORY + '/pht_results/results.pkl'

            train = Train(model=MODEL_PATH, results=RESULT_PATH)

            try:
                remove_dirs = [DIRECTORY + '/synthetic', DIRECTORY + '/encrypted',
                               DIRECTORY + '/keys', DIRECTORY + '/pht_results']
                for rm_dir in remove_dirs:
                    shutil.rmtree(rm_dir)
            except Exception as e:
                pass
            directories = [DIRECTORY + '/pht_results', DIRECTORY + '/decrypted']
            if SAVE_DATA:
                directories = [DIRECTORY, DIRECTORY + '/encrypted']
            elif SAVE_KEYS:
                directories.append(DIRECTORY)
                directories.append(DIRECTORY + '/keys')
            for dir in directories:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            results = train.load_results()
            results, keys = generate_keys(stations, DIRECTORY, results, SAVE_KEYS)

            if SIMULATE_PUSH_PULL:
                train.save_results(results)

            times = {"station_1": [],
                     "proxy": [],
                     "station_2": [],
                     "s_1_total": []}

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
                total_s1 = 0

                for j in range(len(data['CCR5'])):
                    X.append(data['CCR5'][j])
                    Y.append(1)
                    try:
                        X.append(data['CXCR4'][j])
                        Y.append(0)
                    except Exception:
                        pass

                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=True)
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
                stat_df = data_generation(pre, label, DATA_STORAGE_PATH, station=i, fake=fakes,
                                          consider_dist=CONSIDER_DIST, save=True)
                print('Station - DPPE-AUC protocol - Step I')

                t1 = time.perf_counter()
                results = pp_auc_protocol(stat_df, results, DIRECTORY, station=i + 1, max_value=max,
                                          save_data=SAVE_DATA, save_keys=SAVE_KEYS, keys=keys)
                t2 = time.perf_counter()
                times["station_1"].append(t2 - t1)
                print('Station {} step 1 time {}'.format(i + 1, times["station_1"][-1]))
                total_s1 += times["station_1"][-1]
                train.save_model(model)

                if SIMULATE_PUSH_PULL:
                    train.save_results(results)
                print('\n')

            print('Starting proxy protocol')
            if SIMULATE_PUSH_PULL:
                results = train.load_results()

            times['s_1_total'].append(total_s1)
            t3 = time.perf_counter()
            results = dppe_auc_proxy(DIRECTORY, results, max_value=max, save_keys=SAVE_KEYS, keys=keys)
            t4 = time.perf_counter()
            times['proxy'].append(t4 - t3)
            print(f'Execution time by proxy station {times["proxy"][-1]:0.4f} seconds')

            if SIMULATE_PUSH_PULL:
                train.save_results(results)
                results = train.load_results()

            print('Station - DPPE-AUC protocol - Step II')
            t5 = time.perf_counter()
            dppe_auc = dppe_auc_station_final(DIRECTORY, results, SAVE_KEYS, keys)
            t6 = time.perf_counter()
            times['station_2'].append(t6 - t5)

            total_time = times['s_1_total'][-1] + times['proxy'][-1] + (times['station_2'][-1] * stations)
            print(f'Execution time by station - Step II {times["station_2"][-1]:0.4f} seconds')
            print('Total time {}'.format(total_time))

            auc_gt, per = calculate_regular_auc(stations, per, DATA_STORAGE_PATH, save=True, data=data)
            diff = auc_gt - dppe_auc
            print('GT-AUC: ', auc_gt)
            print('Difference DPPE-AUC to GT: ', diff)
            per['total_time'].append(total_time)
            total_times.append(total_time)

            auc_diff.append(diff)
        print('\n')
        avg_diff = sum(auc_diff) / len(auc_diff)
        print('Average differences over {} runs with %fake {} by {} and all {}'.format(len(auc_diff), fakes * 100,
                                                                                       avg_diff, auc_diff))
        print('Max values for range of parameters: {}'.format(max))
        avg_time = sum(total_times) / len(total_times)
        print('Average time total {} and each runtime {}'.format(avg_time, total_times))

        auc_diff = []
        total_times = []
        print('\n')
    print(per)
