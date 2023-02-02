from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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
    def __init__(self, model=None ,results=None):
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
            print("Loading previous results")
            with open(self.encoded_model, "rb") as model_file:
                model = pickle.load(model_file)
            return model
        except:
            return None

def data_generation(pre, label):
    subjects = len(pre)
    fake_patients = [int(subjects*.20), int(subjects*.50)]
    fake_data_val = randint(fake_patients[0], fake_patients[1])
    print(fake_data_val)
    fake_data = {"Pre": np.random.randint(low=5, high=100, size=fake_data_val),
                    "Label": np.random.choice([0], size=fake_data_val),
                    "Flag": np.random.choice([0], size=fake_data_val)
                 }
    data = {'Pre': pre,
            'Label': label,
            'Flag': np.random.choice([1], size=len(label))}

    df_real = pd.DataFrame(data, columns=['Pre', 'Label', 'Flag'])
    df_fake = pd.DataFrame(fake_data, columns=['Pre', 'Label', 'Flag'])
    dfs = [df_real, df_fake]
    merged = pd.concat(dfs, axis=0)
    stat_df = merged.sample(frac=1).reset_index(drop=True)
    return stat_df


if __name__ == '__main__':
    DIRECTORY = './showcase'
    MODEL_PATH = DIRECTORY + '/pht_results/model.pkl'
    RESULT_PATH = DIRECTORY + '/pht_results/results.pkl'
    train = Train(model=MODEL_PATH, results=RESULT_PATH)

    stations = 3
    directories = [DIRECTORY + '/keys', DIRECTORY + '/encrypted', DIRECTORY + '/pht_results']
    for dir in directories:
        try:
            shutil.rmtree(dir)
        except Exception as e:
            print(e)
    directories = [DIRECTORY + '/keys',
                   DIRECTORY + '/encrypted', DIRECTORY + '/pht_results']
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)



    results = train.load_results()
    results = generate_keys(stations, DIRECTORY, results)
    train.save_results(results)

    for i in range(stations):
        print('Station {}'.format(i+1))
        filename = DIRECTORY + '/data/sequences_s' + str(i+1) + '.txt'
        # encode
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

        print("Number of Data Points:", {key: len(value) for (key, value) in data.items()})

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
        print('Hold out test size: {}'.format(Counter(y_test)))
        if model is None:
            model = MultinomialNB(alpha=0.01)
            acc = 0
        else:
            acc = model.score(x_test, y_test)
            print('GT AUC with new test data on previous model: {}'.format(acc))
        classes = np.array([0, 1])
        model.partial_fit(x_train, y_train, classes=classes)
        y_pred = model.predict(x_test)
        auc_gt = metrics.roc_auc_score(y_test, y_pred)
        print('GT AUC with test data after training model: {}'.format(auc_gt))

        ## TODO DPPE-AUC difference to gt protocol fix
        y_pred_prob = model.predict_proba(x_test)[:, -1]
        pre = np.array(y_pred_prob*100).astype(int)
        label = y_test
        print(metrics.roc_auc_score(label, pre))
        # Generate flag data within df
        stat_df = data_generation(pre, label)

        prev_results = train.load_results()
        new_results = pp_auc_protocol(stat_df, prev_results, DIRECTORY, station=i+1)

        train.save_model(model)
        train.save_results(new_results)
        print('\n')

    # proxy
    results = train.load_results()
    new_results = dppe_auc_proxy(DIRECTORY, results)
    train.save_results(new_results)

    results = train.load_results()
    dppe_auc = dppe_auc_station_final(DIRECTORY, results)

    diff = auc_gt - dppe_auc
    print('GT-AUC: ', auc_gt)
    print('Difference DPPE-AUC to GT: ', diff)