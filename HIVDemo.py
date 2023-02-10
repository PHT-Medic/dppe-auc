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


def data_generation(pre, label, data_path, station):
    real_data = {'Pre': pre, 'Label': label,
                 'Flag': np.random.choice([1], size=len(label))}
    df_real = pd.DataFrame(real_data, columns=['Pre', 'Label', 'Flag'])

    fake_patients = [int(len(pre) * .20), int(len(pre) * .50)]
    fake_data_val = randint(fake_patients[0], fake_patients[1])
    print('Fake subjects for dppe-auc {}'.format(fake_data_val))

    tmp_val = list(df_real['Pre'].sort_values(ascending=False))
    values = [tmp_val[i] for i in sorted(np.unique(tmp_val, return_index=True)[1])]
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

    for run in range(10):

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

        # For comparison of gt-auc and dppe-auc
        global_test_x = []
        global_test_y = []

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
            global_test_y.extend(y_test)
            global_test_x.extend(x_test)
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

            stat_df = data_generation(pre, label, DATA_STORAGE_PATH, station=i)

            print('Station - DPPE-AUC protocol - Step I')
            new_results = pp_auc_protocol(stat_df, prev_results, DIRECTORY, station=i + 1)

            train.save_model(model)
            train.save_results(new_results)
            print('\n')

        print('Starting proxy protocol')
        results = train.load_results()
        new_results = dppe_auc_proxy(DIRECTORY, results)
        train.save_results(new_results)

        results = train.load_results()
        final_model = train.load_model()
        print('Station - DPPE-AUC protocol - Step II')
        dppe_auc = dppe_auc_station_final(DIRECTORY, results)

        per = {'samples': []}
        auc_gt, _ = calculate_regular_auc(stations, per, DATA_STORAGE_PATH)
        diff = auc_gt - dppe_auc
        print('GT-AUC: ', auc_gt)
        print('Difference DPPE-AUC to GT: ', diff)
        auc_diff.append(diff)
    print('Average difference over {} runs: {}'.format(len(auc_diff), sum(auc_diff) / len(auc_diff)))