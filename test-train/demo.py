from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from main import *
import json
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
            return {'initial': False,
                    'proxy': False,
                    'station': -1,
                    'data_approx': [],
                    'data_exact': [],
                    'per': {'approx': {'samples': [],
                                       'flags': [],
                                       'total_time': []},
                            'exact': {'samples': [],
                                      'flags': [],
                                      'total_time': []}
                            },
                    'times': {'approx': {"init": [],
                                         "station_1": [],
                                         "proxy": [],
                                         "station_2": [],
                                         "s_1_total": []},
                              'exact': {"init": [],
                                        "station_1": [],
                                        "proxy": [],
                                        "station_2": [],
                                        "s_1_total": []}},
                    'approx': {'enc_rx': {},
                               'enc_s_p_sks': [],
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
                              'enc_s_p_sks': [],
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
        #plot_input_data(df, df_real, df_fake, station, run, proxy=False)

        df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0
        print("Size complete: {}".format(len(df)))
        if save:
            df.to_pickle(data_path + '/data_s' + str(station + 1) + '.pkl')
        return df


def initial_station(results):
    # print(os.getenv("PRIVATE_KEY_PATH"))
    # Get users, and all stations PK
    with open(os.getenv("CONF_PATH"), 'r') as f:
        trainConfig = json.load(f)

    rsa_pks = []
    for i in range(len(trainConfig['route'])):
        rsa_pks.append(bytes.fromhex(trainConfig['route'][i]['rsa_public_key']))

    print('prepared keys for {} stations'.format(len(rsa_pks)))
    results['stations_rsa_pk'] = rsa_pks  # 0 init 1 station 2 station n+1 proxy
    results['aggregator_rsa_pk'] = rsa_pks[-1]

    user_pk = bytes.fromhex(trainConfig['creator']['rsa_public_key'])
    results['users_rsa_pk'] = user_pk

    # Stations Paillier Keys
    env_symm_key = Fernet.generate_key()
    for i in range(len(rsa_pks)):
        sk, pk = generate_keypair(3072)  # paillier keys for stations

        enc_sk = {'n': Fernet(env_symm_key).encrypt(bytes(str(sk.n), 'utf-8')),
                  'x': Fernet(env_symm_key).encrypt(bytes(str(sk.x), 'utf-8'))
                }
        results['enc_s_p_sks'].append(enc_sk)
        results['stations_paillier_pk'][i] = pk

    # Aggregator Paillier Keys
    sk, pk = generate_keypair(3072)
    sk_1 = copy.copy(sk)
    sk_2 = copy.copy(sk)

    enc_sk_1 = {'n': Fernet(env_symm_key).encrypt(bytes(str(sk_1.n), 'utf-8')),
                'nsqr': Fernet(env_symm_key).encrypt(bytes(str(sk_1.nsqr), 'utf-8')),
                'x1': Fernet(env_symm_key).encrypt(bytes(str(sk_1.x1), 'utf-8'))
                }
    results['enc_agg_sk_1'] = enc_sk_1

    enc_sk_2 = {'n': Fernet(env_symm_key).encrypt(bytes(str(sk_2.n), 'utf-8')),
                'nsqr': Fernet(env_symm_key).encrypt(bytes(str(sk_2.nsqr), 'utf-8')),
                'x2': Fernet(env_symm_key).encrypt(bytes(str(sk_1.x2), 'utf-8'))
                }

    enc_env_key = []
    for i in range(len(rsa_pks)):
        enc_env_key.append(rsa_encrypt(env_symm_key, rsa_pks[i]))
    enc_env_key.append(rsa_encrypt(env_symm_key, user_pk))
    results['enc_symm_key'] = enc_env_key
    results['enc_agg_sk_2'] = enc_sk_2  # for all stations and users partial Pailler private key

    results['aggregator_paillier_pk'] = pk

    # simulate private key separation for the aggregator
    del sk_1.x2
    del sk_1.x

    del sk_2.x1
    del sk_2.x
    return results


if __name__ == '__main__':
    DIRECTORY = os.getcwd()

    #  print("Comparing both approaches in same run")
    MAX = 100000
    no_of_decision_points = 200
    print(os.getcwd())
    approx_auc_diff, exact_auc_diff = [], []
    approx_total_times, exact_total_times = [], []
    total_repetitions = 1  # 10 before

    best_time = 100
    best_diff = 10

    decision_points = np.linspace(0, 1, num=no_of_decision_points)[::-1]

    MODEL_PATH = '/opt/pht_results/model.pkl'  # if prod
    RESULT_PATH = '/opt/pht_results/results.pkl'

    #MODEL_PATH = DIRECTORY + '/model.pkl' # if local
    #RESULT_PATH = DIRECTORY + '/results.pkl'

    train = Train(model=MODEL_PATH, results=RESULT_PATH)

    # Init station: create keys, save init
    results = train.load_results()
    #print(results)
    times = results['times']
    per = results['per']
    data_exact = results['data_exact']
    data_approx = results['data_approx']
    # stations = len(results['approx']['stations_rsa_pk']) - 1  #
    stations = results['station'] + 1
    print('Station: {}'.format(stations))

    results['station'] = stations  # save new value
    # results['proxy'] = True  # TODO remove in prod
    if not results['initial']:
        print('Station Init - Create and encrypt keys')

        t0 = time.perf_counter()
        results['approx'] = initial_station(results['approx'])
        t1 = time.perf_counter()
        times['approx']["init"].append(t1 - t0)
        print(f'Key creation approximation method time {sum(times["approx"]["init"]):0.4f} seconds')
        t0 = time.perf_counter()
        results['exact'] = initial_station(results['exact'])
        t1 = time.perf_counter()
        times['exact']["init"].append(t1 - t0)
        print(f'Key creation approximation method time {sum(times["exact"]["init"]):0.4f} seconds')
        results['initial'] = True
        results['times'] = times
        train.save_results(results)
        print('Keys saved')
        exit(0)
    elif not results['proxy']:
        # Station part I: load data, train model, save model, save data

        filename = '/opt/pht_train/sequences_s' + str(stations) + '.txt'
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


        #### START DPPE Protocol
        y_pred_prob = model.predict_proba(x_test)[:, -1]
        pre = np.array(y_pred_prob)

        label = y_test

        exact_stat_df = data_generation(pre, label, DIRECTORY + '/pht_results/', station=stations, run=1,
                                        save=False,
                                        APPROX=False)
        approx_stat_df = data_generation(pre, label, DIRECTORY + '/pht_results/', station=stations, run=1,
                                         save=False,
                                         APPROX=True)

        data_approx.append(approx_stat_df.copy())
        data_exact.append(exact_stat_df.copy())

        print('Station - DPPA-AUC protocol - Step I')
        t1 = time.perf_counter()
        results_approx = dppa_auc_protocol(approx_stat_df, decision_points, results["approx"],
                                           station=int(stations), max_value=MAX)
        t2 = time.perf_counter()
        times['approx']["station_1"].append(t2 - t1)
        print(f'Approx execution time by station {times["approx"]["station_1"][-1]:0.4f} seconds')

        print('Station - DPPE-AUC protocol - Step I')
        t1 = time.perf_counter()
        results_exact = dppe_auc_protocol(exact_stat_df, results["exact"], station=int(stations), max_value=MAX)
        t2 = time.perf_counter()
        times['exact']["station_1"].append(t2 - t1)
        print(f'Exact execution time by station {times["exact"]["station_1"][-1]:0.4f} seconds')

        total_s1_approx += times['approx']["station_1"][-1]
        total_s1_exact += times['exact']["station_1"][-1]

        times['approx']['s_1_total'].append(total_s1_approx)
        times['exact']['s_1_total'].append(total_s1_exact)

        results['data_approx'] = data_approx
        results['data_exact'] = data_exact

        train.save_model(model)

        if stations == len(results['approx']['stations_rsa_pk']) - 2:  # in dev -1
            results['proxy'] = True

        train.save_results(results)
    elif results['proxy']:
        #  Proxy Computation
        print('Starting proxy protocol')

        #times['approx']['s_1_total'].append(total_s1_approx)
        #times['exact']['s_1_total'].append(total_s1_exact)
        t3 = time.perf_counter()
        results['approx'] = dppa_auc_proxy(results["approx"], max_value=MAX, no_dps=no_of_decision_points)
        t4 = time.perf_counter()
        times["approx"]['proxy'].append(t4 - t3)
        print(f'Approx execution time by proxy station {times["approx"]["proxy"][-1]:0.4f} seconds')

        t3 = time.perf_counter()
        results['exact'] = dppe_auc_proxy(results["exact"], max_value=MAX)
        t4 = time.perf_counter()
        times["exact"]['proxy'].append(t4 - t3)
        print(f'Exact execution time by proxy station {times["exact"]["proxy"][-1]:0.4f} seconds')

        results['times'] = times
        train.save_results(results)

    # Final: User has locally to run last step
