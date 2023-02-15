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
                    'N3': [],
                    'floats': []
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


def data_generation(pre, label, data_path, station, fake):
    real_data = {'Pre': pre, 'Label': label,
                 'Flag': np.random.choice([1], size=len(label))}
    df_real = pd.DataFrame(real_data, columns=['Pre', 'Label', 'Flag'])
    fake_data_val = int(len(pre) * fake)
    print('Fake subjects for dppe-auc {}'.format(fake_data_val))

    tmp_val = list(df_real['Pre'].sort_values(ascending=False))
    values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]
    prob = list(df_real['Pre'].value_counts(normalize=True, ascending=False))
    fake_data = {"Pre": random.choices(values, weights=prob, k=fake_data_val),
                 "Label": np.random.choice([0], size=fake_data_val),
                 "Flag": np.random.choice([0], size=fake_data_val)
                 }
    df_fake = pd.DataFrame(fake_data, columns=['Pre', 'Label', 'Flag'])

    dfs = [df_real, df_fake]
    merged = pd.concat(dfs, axis=0)
    df = merged.sample(frac=1).reset_index(drop=True)

    df.to_pickle(data_path + '/data_s' + str(station + 1) + '.pkl')

    return df


if __name__ == '__main__':
    DIRECTORY = './showcase'
    auc_diff = []

    # Privacy related
    max_values = [10, 100, 1000, 10000, 100000]
    max = 100 # best max is 100

    size_fake = [0.1, 0.3, 0.5, 0.6, 0.8, 1, 2, 3]
    total_times = []
    total_repetitions = 10

    best_time = 40
    best_diff = 10

    # for run in range(len(size_fake)):  # uncomment to evaluate max value
        # max = max_values[run]

    for run in range(len(size_fake)):  # comment these two lines, if max values are derived
        fakes = size_fake[run]
        for repetition in range(total_repetitions):
            DATA_STORAGE_PATH = DIRECTORY + '/decrypted'
            MODEL_PATH = DIRECTORY + '/pht_results/model.pkl'
            RESULT_PATH = DIRECTORY + '/pht_results/results.pkl'

            train = Train(model=MODEL_PATH, results=RESULT_PATH)

            stations = 3
            directories = [DIRECTORY + '/keys', DIRECTORY + '/encrypted', DIRECTORY + '/decrypted',
                           DIRECTORY + '/pht_results']
            for path in directories:
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    print(e)
            directories = [DIRECTORY + '/keys', DIRECTORY + '/decrypted',
                           DIRECTORY + '/encrypted', DIRECTORY + '/pht_results']
            for path in directories:
                if not os.path.exists(path):
                    os.makedirs(path)

            results = train.load_results()
            results = generate_keys(stations, DIRECTORY, results)
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
                prev_results = train.load_results()

                pre = np.array(y_pred_prob)

                if pre.dtype == 'float64':
                    prev_results['floats'].insert(0, True)
                else:
                    prev_results['floats'].insert(0, False)

                label = y_test
                stat_df = data_generation(pre, label, DATA_STORAGE_PATH, station=i, fake=fakes)
                print('Station - DPPE-AUC protocol - Step I')

                t1 = time.perf_counter()
                new_results = pp_auc_protocol(stat_df, prev_results, DIRECTORY, station=i + 1, max_value=max)
                t2 = time.perf_counter()
                times["station_1"].append(t2 - t1)
                print('Station {} step 1 time {}'.format(i + 1, times["station_1"][-1]))
                total_s1 += times["station_1"][-1]
                train.save_model(model)
                train.save_results(new_results)
                print('\n')

            print('Starting proxy protocol')
            results = train.load_results()
            times['s_1_total'].append(total_s1)
            t3 = time.perf_counter()
            new_results = dppe_auc_proxy(DIRECTORY, results, max)
            t4 = time.perf_counter()
            times['proxy'].append(t4-t3)
            print(f'Execution time by proxy station {times["proxy"][-1]:0.4f} seconds')

            train.save_results(new_results)

            results = train.load_results()
            final_model = train.load_model()
            print('Station - DPPE-AUC protocol - Step II')
            t5 = time.perf_counter()
            dppe_auc = dppe_auc_station_final(DIRECTORY, results)
            t6 = time.perf_counter()
            times['station_2'].append(t6-t5)

            total_time = times['s_1_total'][-1] + times['proxy'][-1] + (times['station_2'][-1] * stations)
            print(f'Execution time by station - Step II {times["station_2"][-1]:0.4f} seconds')
            per = {'samples': []}
            auc_gt, _ = calculate_regular_auc(stations, per, DATA_STORAGE_PATH)
            diff = auc_gt - dppe_auc
            print('GT-AUC: ', auc_gt)
            print('Difference DPPE-AUC to GT: ', diff)
            total_times.append(total_time)

            auc_diff.append(diff)
        print('\n')
        avg_diff = sum(auc_diff) / len(auc_diff)
        print('Average differences over {} runs with %fake {} by {} and all {}'.format(len(auc_diff), fakes, avg_diff , auc_diff))
        print('Max values for range of parameters: {}'.format(max))
        avg_time = sum(total_times)/len(total_times)
        print('Times each {} and average runtime {}'.format(total_times, avg_time))
        if avg_time <= best_time and avg_diff <= best_diff:
            best_time = avg_time
            best_diff = avg_diff
            best_f = fakes
        auc_diff = []
        total_times = []
        print('\n')
    print('Best diff {} and best time {} with {} fake samples on average over {} runs'.format(best_diff, best_time, best_f, total_repetitions))