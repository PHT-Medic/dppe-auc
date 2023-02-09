import copy
import os
import pickle
import shutil
import struct
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from sklearn import metrics
from paillier.paillier import *

class Train:
    def __init__(self, results=None):
        """
        :param results:
        """
        self.results = results

    def load_results(self):
        """
        If a result file exists, loads the results, otherwise will return empty results.
        :return:
        """
        try:
            if not os.path.isdir('./data/pht_results'):
                os.makedirs('./data/pht_results')
                print('Created results directory')
            with open('./data/pht_results/' + self.results, 'rb') as results_file:
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
            with open('./data/pht_results/' + self.results, 'wb') as results_file:
                return pickle.dump(results, results_file)
        except Exception as err:
            print(err)
            raise FileNotFoundError("Result file cannot be saved")


def make_df(data: object, valid, station) -> object:
    df = pd.DataFrame(data, columns=['Pre', 'Label', 'Flag'])
    df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0
    if not np.all(df["Label"] == 1):
        valid = True
        df.to_pickle('./data/synthetic/data_s' + str(station + 1) + '.pkl')
    return valid


def create_synthetic_data(num_stations=int, samples=int, fake_patients=None):
    """
    Create and save synthetic data of given number of samples and number of stations. Including flag patients
    """
    for station_i in range(num_stations):
        valid = False
        while not valid:
            fake_data_val = randint(fake_patients[0], fake_patients[1]) # random number flag value in given percentage range
            data = {"Pre": np.random.random(size=samples+fake_data_val),
                    #"Pre": np.random.randint(low=5, high=10000, size=samples+fake_data_val),
                    "Label": np.random.choice([0, 1], size=samples+fake_data_val, p=[0.1, 0.9]),
                    "Flag": np.random.choice(np.concatenate([[1] * samples, [0] * fake_data_val]),
                                             samples+fake_data_val, replace=False)}
            valid = make_df(data, valid, station_i)


def create_synthetic_data_same_size(num_stations=int, samples=int, fake_patients=None):
    """
    Create and save synthetic data of given number of samples and number of stations. Including flag patients
    """
    samples_each = samples // num_stations
    left_over = samples % num_stations
    for station_i in range(num_stations):
        if station_i == range(num_stations)[-1]: # add left number over at last stations
            samples_each = samples_each + left_over
        valid = False
        while not valid:
            fake_data_val = randint(fake_patients[0], fake_patients[1])  # random number flag value in given percentage range
            data = {"Pre": np.random.random(size=samples_each),
                    #"Pre": np.random.randint(low=5, high=10000, size=samples_each),
                    "Label": np.random.choice([0, 1], size=samples_each, p=[0.2, 0.8]),
                    "Flag": np.random.choice(np.concatenate([[1] * samples_each, [0] * fake_data_val]),
                                             samples_each, replace=False)}
            valid = make_df(data, valid, station_i)


def calculate_regular_auc(stations, performance, regular_path):
    """
    Calculate AUC with sklearn as ground truth GT
    """
    lst_df = []
    for i in range(stations):
        df_i = pickle.load(open(regular_path + '/data_s' + str(i+1) + '.pkl', 'rb'))
        lst_df.append(df_i)

    concat_df = pd.concat(lst_df)
    performance['samples'].append(len(concat_df))
    print('Use data from {} stations. Total of {} subjects (including flag subjects) '.format(stations, len(concat_df)))

    #sort_df = concat_df.sort_values(by='Pre', ascending=False).reset_index()
    sort_df = concat_df.sort_values(by='Pre', ascending=False)

    filtered_df = sort_df[sort_df["Flag"] == 1]  # remove flag patients
    dfd = filtered_df.copy()
    dfd["Pre"] = filtered_df["Pre"]
    y = dfd["Label"]
    pred = dfd["Pre"]

    gt = metrics.roc_auc_score(y, pred)

    return gt, performance


def generate_keys(stations, directory, results):
    """
    Generate and save keys of given numbers of stations and train results
    """
    for i in range(stations):
        # paillier keys
        sk, pk = generate_keypair(3072)
        pickle.dump(sk, open(directory + '/keys/s' + str(i+1) + '_paillier_sk.p', 'wb'))
        pickle.dump(pk, open(directory + '/keys/s' + str(i+1) + '_paillier_pk.p', 'wb'))
        results['stations_paillier_pk'][i] = pk

        # rsa keys
        rsa_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        rsa_public_key = rsa_private_key.public_key()


        private_pem = rsa_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        with open(directory + '/keys/s' + str(i+1) + '_rsa_sk.pem', 'wb') as f:
            f.write(private_pem)

        public_pem = rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        with open(directory + '/keys/s' + str(i+1) + '_rsa_pk.pem', 'wb') as f:
            f.write(public_pem)

        results['stations_rsa_pk'][i] = public_pem

    # generate keys of aggregator
    sk, pk = generate_keypair(3072)
    sk_1 = copy.copy(sk)
    sk_2 = copy.copy(sk)
    # simulate private key separation
    del sk_1.x2
    del sk_2.x1
    pickle.dump(sk_1, open(directory + '/keys/agg_sk_1.p', 'wb'))
    pickle.dump(sk_2, open(directory + '/keys/agg_sk_2.p', 'wb'))
    pickle.dump(pk, open(directory + '/keys/agg_pk.p', 'wb'))
    results['aggregator_paillier_pk'] = pk

    rsa_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096,
        backend=default_backend()
    )
    rsa_public_key = rsa_private_key.public_key()


    private_pem = rsa_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    with open(directory + '/keys/agg_rsa_private_key.pem', 'wb') as f:
        f.write(private_pem)

    public_pem = rsa_public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open(directory + '/keys/agg_rsa_public_key.pem', 'wb') as f:
        f.write(public_pem)

    results['aggregator_rsa_pk'] = public_pem

    return results


def encrypt_table(station_df, agg_pk, r1, r2, symmetric_key, prev_results):
    """
    Encrypt dataframe of given station dataframe with paillier public key of aggregator and random values
    """
    #print('Start encrypting table with {} subjects from station {}'.format(len(station_df), station))
    #tic = time.perf_counter()
    station_df["Pre"] *= r1
    station_df["Pre"] += r2

    #print(station_df["Pre"].to_list())
    #print(station_df["Pre"]) # TODO uncomment to see obscured pre
    if prev_results['floats'][0] is True:
        station_df["Pre"] = station_df["Pre"].apply(lambda x: Fernet(symmetric_key).encrypt(struct.pack("f", x)))
    else:
        station_df["Pre"] = station_df["Pre"].apply(lambda x: Fernet(symmetric_key).encrypt(int(x).to_bytes(4, 'big')))

    #print(len(station_df["Pre"][0]))
    station_df["Label"] = station_df["Label"].apply(lambda x: encrypt(agg_pk, x))
    station_df["Flag"] = station_df["Flag"].apply(lambda x: encrypt(agg_pk, x))
    #toc = time.perf_counter()
    #print(f'Encryption time of table {toc - tic:0.4f} seconds')
    return station_df


def load_rsa_sk(path):
    """
    Return private rsa key of given file path
    """
    with open(path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    return private_key


def load_rsa_pk(path):
    """
    Return public rsa key of given file path
    """
    with open(path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )
    return public_key


def encrypt_symmetric_key(symmetric_key, directory):
    """
    Encrypt symmetric key_station with public rsa key of aggregator
    return: encrypted_symmetric_key
    """
    rsa_agg_pk = load_rsa_pk(directory + '/keys/agg_rsa_public_key.pem')
    encrypted_symmetric_key = rsa_agg_pk.encrypt(symmetric_key, padding.OAEP(
                                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                algorithm=hashes.SHA256(),
                                                label=None
                                            ))

    return encrypted_symmetric_key


def decrypt_symmetric_key(ciphertext, directory):
    """
    Decrypt of given station rsa encrypted k_station
    """
    rsa_agg_sk = load_rsa_sk(directory + '/keys/agg_rsa_private_key.pem')
    decrypted_symmetric_key = rsa_agg_sk.decrypt(
        ciphertext,
        padding.OAEP(
         mgf=padding.MGF1(algorithm=hashes.SHA256()),
         algorithm=hashes.SHA256(),
         label=None
        ))
    return decrypted_symmetric_key


def pp_auc_protocol(station_df, prev_results, directory=str, station=int):
    """
    Perform PP-AUC protocol at specific station given dataframe
    """

    agg_pk = prev_results['aggregator_paillier_pk']
    symmetric_key = Fernet.generate_key() # represents k1 k_n
    if station == 1:
        r1 = randint(1, 100)  # random value between 1 to 100
        r2 = randint(1, 100)

        enc_table = encrypt_table(station_df, agg_pk, r1, r2, symmetric_key, prev_results)
        # Save for transparency the table - not required
        enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, directory)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1)  # homomorphic encryption used
            enc_r2 = encrypt(prev_results['stations_paillier_pk'][i], r2)
            prev_results['encrypted_r1'][i] = enc_r1
            prev_results['encrypted_r2'][i] = enc_r2

    else:
        enc_r1 = prev_results['encrypted_r1'][station-1]
        sk_s_i = pickle.load(open(directory + '/keys/s' + str(station) + '_paillier_sk.p', 'rb'))
        dec_r1 = decrypt(sk_s_i, enc_r1)

        enc_r2 = prev_results['encrypted_r2'][station-1]
        dec_r2 = decrypt(sk_s_i, enc_r2)

        enc_table = encrypt_table(station_df, agg_pk, dec_r1, dec_r2, symmetric_key, prev_results)
        enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, directory)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

    prev_results['pp_auc_tables'][station-1] = enc_table

    return prev_results


def z_values(n):
    """
    Generate random values of list length n which sum is zero
    """
    l = random.sample(range(-int(n/2), int(n/2)), k=n-1)
    return l + [-sum(l)]


def dppe_auc_proxy(directory, results):
    """
    Simulation of aggregator service - globally computes privacy preserving AUC table as proxy station
    """
    agg_pk = results['aggregator_paillier_pk']
    agg_sk = pickle.load(open(directory + '/keys/agg_sk_1.p', 'rb'))

    # decrypt symmetric keys (k_stations)
    df_list = []
    for i in range(len(results['encrypted_ks'])):
        enc_k_i = results['encrypted_ks'][i]
        dec_k_i = decrypt_symmetric_key(enc_k_i, directory)

        # decrypt table values with Fernet and corresponding k_i symmetric key
        table_i = results['pp_auc_tables'][i]
        table_i["Dec_pre"] = table_i["Pre"].apply(lambda x: Fernet(dec_k_i).decrypt(x))  # returns bytes
        if results['floats'][0] is True:
            d = table_i["Dec_pre"].apply(lambda x: struct.unpack('f', x)).to_list()
            lst = [x[0] for x in d]
            table_i["Dec_pre"] = lst
        else:
            table_i["Dec_pre"] = table_i["Dec_pre"].apply(lambda x: int.from_bytes(x, "big"))
        df_list.append(table_i)

    concat_df = pd.concat(df_list)
    concat_df.pop('Pre')
    sort_df = concat_df.sort_values(by='Dec_pre', ascending=False)
    df_new_index = sort_df.reset_index()
    M = len(df_new_index)
    tp_values = []
    fp_values = []

    tp_values.insert(0, encrypt(agg_pk, 0))
    fp_values.insert(0, encrypt(agg_pk, 0))
    tmp_sum = fp_values[0]
    for i in range(1, M + 1):
        tp_values.append(add(agg_pk, tp_values[i - 1], df_new_index['Label'][i - 1]))
        sum_flags = add(agg_pk, df_new_index['Flag'][i - 1], tmp_sum)
        tmp_sum = sum_flags
        fp_values.append(add(agg_pk, sum_flags, mul_const(agg_pk, tp_values[-1], -1)))

    a = randint(1, 100)
    b = randint(1, 100)

    # Denominator
    # TP_A is summation of labels (TP)
    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)

    r_1A = randint(1, 100)
    r_2A = randint(1, 100)
    D1 = add_const(agg_pk, tp_a_mul, r_1A)
    D2 = add_const(agg_pk, fp_a_mul, r_2A)
    D3_1 = mul_const(agg_pk, tp_a_mul, r_2A)
    D3_2 = mul_const(agg_pk, fp_a_mul, r_1A)
    D3 = add(agg_pk, D3_1, add_const(agg_pk, D3_2, r_1A * r_2A))

    # partial decrypt and save to train
    results["D1"].append(proxy_decrypt(agg_sk, D1))
    results["D2"].append(proxy_decrypt(agg_sk, D2))
    results["D3"].append(proxy_decrypt(agg_sk, D3))

    # Tie condition differences between TP and FP
    # determine indexes of threshold values
    thre_ind = []
    pred = df_new_index["Dec_pre"].to_list()
    for i in range(M - 1):
        if pred[i] != pred[i + 1]:
            thre_ind.append(i)

    thre_ind = list(map(lambda x: x + 1, thre_ind))  # add one
    len_t = len(thre_ind)
    # Multiply with a and b respectively
    Z_values = z_values(len_t)

    # sum over all n_3 and only store n_3
    N_3_sum = encrypt(agg_pk, 0)
    for i in range(1, len_t+1):
        pre_ind = thre_ind[i - 1]
        if i == len_t:
            cur_ind = -1
        else:
            cur_ind = thre_ind[i]
        # Multiply with a and b respectively
        sTP_a = mul_const(agg_pk, add(agg_pk, tp_values[cur_ind], tp_values[pre_ind]), a)
        dFP_b = mul_const(agg_pk, add(agg_pk, fp_values[cur_ind], mul_const(agg_pk, fp_values[pre_ind], -1)), b)
        r1_i = randint(1, 100)
        r2_i = randint(1, 100)

        n_1 = add_const(agg_pk, sTP_a, r1_i)
        results["N1"].append(proxy_decrypt(agg_sk, n_1))

        n_2 = add_const(agg_pk, dFP_b, r2_i)
        results["N2"].append(proxy_decrypt(agg_sk, n_2))

        N_i3_1 = mul_const(agg_pk, sTP_a, r2_i)
        N_i3_2 = mul_const(agg_pk, dFP_b, r1_i)
        N_i3_a = add(agg_pk, N_i3_1, add_const(agg_pk, N_i3_2, r1_i * r2_i))
        n_3 = add_const(agg_pk, N_i3_a, Z_values[i - 1])
        n_3_tmp = add(agg_pk, N_3_sum, n_3)
        N_3_sum = n_3_tmp

    results["N3"].append(proxy_decrypt(agg_sk, N_3_sum))

    return results


def dppe_auc_station_final(directory, train_results):
    """
    Simulation of station delegated AUC parts to compute global DPPE-AUC locally
    """
    agg_sk_2 = pickle.load(open(directory + '/keys/agg_sk_2.p', 'rb'))
    agg_pk = train_results['aggregator_paillier_pk']

    # decrypt random components D1, D2, D3, Ni1, Ni2, Ni3
    D1 = station_decrypt(agg_sk_2, train_results['D1'][0])
    D2 = station_decrypt(agg_sk_2, train_results['D2'][0])
    D3 = station_decrypt(agg_sk_2, train_results['D3'][0])

    sum_n_1_mul_2 = 0
    for j in range(len(train_results['N2'])):
        n_i1 = station_decrypt(agg_sk_2, train_results['N1'][j])
        n_1_mul_n_2 = mul_const(agg_pk,train_results['N2'][j], n_i1)
        if j == 0:
                sum_n_1_mul_2 = n_1_mul_n_2
        else:
                sum_n_1_mul_2 = add(agg_pk, sum_n_1_mul_2, n_1_mul_n_2)

    E_N = add(agg_pk, sum_n_1_mul_2 , mul_const(agg_pk, train_results['N3'][0], -1))
    N = station_decrypt(agg_sk_2, E_N)

    D = (D1 * D2) - D3
    if D == 0:
        auc = 0
    else:
        auc = (N / D) / 2
    print('DPPE-AUC: {}'.format(auc))
    return auc


def experiment_1(res):
    total_time = [sum(x) for x in zip(*[res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2']])]
    df = pd.DataFrame(list(zip(res['time']['stations_1'], res['time']['proxy'],
                               res['time']['stations_2'], total_time, res['samples'], res['stations'])),
                      index=res['stations'],
                      columns=['Station_1', 'Proxy', 'Station_2', 'Total', 'Samples', 'Stations'])
    # Create a figure and an axis
    s3_data = df.loc[df['Stations'] == 3]
    s6_data = df.loc[df['Stations'] == 6]
    s9_data = df.loc[df['Stations'] == 9]
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(str(10) + ' runs with ' + str(res['samples'][0]) + ' subjects')
    ax1.set_ylabel('Time (sec)')

    ax2.set_ylabel('Time (sec)')
    ax3.set_ylabel('Time (sec)')
    ax1.set_xticklabels('3')
    ax2.set_xticklabels('6')
    ax3.set_xticklabels('9')
    ax1.set_xlabel('Stations')
    ax2.set_xlabel('Stations')
    ax3.set_xlabel('Stations')
    ax1.boxplot(s3_data['Total'])
    ax2.boxplot(s6_data['Total'])
    ax3.boxplot(s9_data['Total'])
    fig.tight_layout()
    plt.savefig('germany_sub_boxplot.png')


def experiment_2(res):
    perf = pd.DataFrame(list(zip(res['pp-auc'], res['gt-auc'])), index=res['stations'], columns=['pp', 'gt'])
    total_time = [sum(x) for x in zip(*[res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2']])]
    df = pd.DataFrame(list(zip(res['time']['stations_1'], res['time']['proxy'],
                               res['time']['stations_2'], total_time, res['samples'], res['stations'])),
                      index=res['stations'],
                      columns=['Station_1', 'Proxy', 'Station_2', 'Total', 'Samples', 'Stations'])
    c = plt.cm.Set2
    color = iter(c.colors)
    for category in df.Stations.unique():
        c = next(color)
        plt.plot('Samples', 'Total', c=c, data=df.loc[df['Stations'].isin([category])], marker='o',
                 label=str(category) + ' stations')
    plt.xlabel('Number of subjects')
    num_stations = res['stations'][0]
    plt.ylabel('Time (sec)')
    plt.title('DPPE-AUC total runtime evaluation')
    plt.legend(loc="upper left")
    plt.savefig('test.png')

    gt = perf['gt']
    pp = perf['pp']
    diff = gt - pp
    print(diff)


if __name__ == "__main__":
    """
    Run with either complete experiment 1 or 2 uncommented
    """

    DIRECTORY = './data'
    # Experiment 1
    station_list = [3, 6, 9]
    subject_list = [150]
    loops = 10

    # Experiment 2 # uncomment to run runtime measurement
    #station_list = [3]
    #subject_list = [25, 100, 200, 400, 800, 1600, 3200, 6400, 12800] # not final list, running until overflow
    #loops = 1

    performance = {'time':
                    {'stations_1': [],
                     'proxy': [],
                     'stations_2': [],
                     'total_step_1': []
                     },
                   'samples': [],
                   'stations': [],
                   'pp-auc': [],
                   'gt-auc': []
                   }

    for subjects in subject_list:
        train = Train(results='results.pkl')
        differences = []
        for stations in station_list:
            for i in range(loops):  # repeat n times, to make boxplot
                performance['stations'].append(stations)
                try:
                    shutil.rmtree(DIRECTORY + '/')
                except Exception as e:
                    print(e)
                directories = [DIRECTORY, DIRECTORY + '/keys', DIRECTORY + '/synthetic',
                               DIRECTORY + '/encrypted', DIRECTORY + '/pht_results']
                for dir in directories:
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                # Experiment 1 - increase number of stations, but same sample size
                create_synthetic_data_same_size(stations, subjects, [int(subjects*.20), int(subjects*.50)])

                # Experiment 2 - keep randomness in stations
                #create_synthetic_data(stations, subjects, [int(subjects*.30), int(subjects*.50)])

                results = train.load_results()
                results = generate_keys(stations, DIRECTORY, results)
                # Mimic train building process
                train.save_results(results)

                # compute AUC without encryption for proof of concept
                REGULAR_PATH = DIRECTORY + '/synthetic'
                auc_gt, performance = calculate_regular_auc(stations, performance, REGULAR_PATH)
                performance['gt-auc'].append(auc_gt)
                print('AUC value of GT {}'.format(auc_gt))
                times = []
                for i in range(stations):
                    stat_df = pickle.load(open(DIRECTORY + '/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))

                    prev_results = train.load_results()  # loading results simulates pull of image
                    if stat_df['Pre'].dtype == 'float64':
                        prev_results['floats'].insert(0, True)
                    else:
                        prev_results['floats'].insert(0, False)

                    t1 = time.perf_counter()
                    results = pp_auc_protocol(stat_df, prev_results, DIRECTORY, station=i+1)
                    t2 = time.perf_counter()
                    times.append(t2 - t1)

                    # remove at last station all encrypted noise values
                    if i is stations - 1:
                        results.pop('encrypted_r1')
                        results.pop('encrypted_r2')

                    train.save_results(results)  # saving results simulates push of image

                print(f'Total execution time at stations {sum(times):0.4f} seconds')
                print(f'Average execution time at stations {sum(times)/len(times):0.4f} seconds')
                performance['time']['stations_1'].append(sum(times)/len(times))
                performance['time']['total_step_1'].append(sum(times))
                results = train.load_results()

                t3 = time.perf_counter()
                results = dppe_auc_proxy(DIRECTORY, results)
                t4 = time.perf_counter()

                performance['time']['proxy'].append(t4 - t3)
                print(f'Execution time by proxy station {t4 - t3:0.4f} seconds')

                train.save_results(results) # simulate push and pull
                results = train.load_results()

                t1 = time.perf_counter()
                auc_pp = dppe_auc_station_final(DIRECTORY, results)
                t2 = time.perf_counter()
                local_dppe = t2 - t1
                performance['time']['stations_2'].append(local_dppe * stations)
                print(f'Final AUC execution time at station {t2 - t1:0.4f} seconds')

                performance['pp-auc'].append(auc_pp)

                diff = auc_gt - auc_pp
                differences.append(diff)
                print('Difference pp-AUC to GT: ', diff)
                print('\n')

            print("Avg difference {} over {} runs".format(sum(differences)/len(differences), len(differences)))

    print(performance)

    # Experiment 1
    experiment_1(performance)
    # Experiment 2
    #experiment_2(performance)