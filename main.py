import os
import time
import copy
import shutil
import pickle
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import randint
from sklearn import metrics
from paillier.paillier import *
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa


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


def return_df(df):
    return pd.DataFrame(df, columns=['Pre', 'Label', 'Flag'])


def create_synthetic_data(num_stations=int, samples=int, fake_patients=None, save=None):
    """
    Create and save synthetic data of given number of samples and number of stations. Including flag patients
    """

    dfs = []
    samples_each = samples // num_stations
    for station_i in range(num_stations):
        fakes = random.uniform(fake_patients[0], fake_patients[1])
        fake_data_val = int(samples_each * fakes)
        real_data = {
            "Pre": np.random.random(size=samples_each),
            "Label": np.random.choice([0, 1], size=samples_each, p=[0.2, 0.8]),
            "Flag": np.random.choice([1], size=samples_each)
        }
        df_real = return_df(real_data)
        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]
        prob = list(df_real['Pre'].value_counts(normalize=True, ascending=False))

        fake_data = {
            "Pre": random.choices(values, weights=prob, k=fake_data_val),
            "Label": np.random.choice([0], size=fake_data_val),
            "Flag": np.random.choice([0], size=fake_data_val)
        }
        df_fake = return_df(fake_data)

        df = [df_real, df_fake]
        merged = pd.concat(df, axis=0)
        df = merged.sample(frac=1).reset_index(drop=True)
        # plot_input_data(df, df_real, df_fake, station_i)

        df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0
        if save:
            df.to_pickle('./data/synthetic/data_s' + str(station_i + 1) + '.pkl')
        else:
            dfs.append(df)
    if not save:
        return dfs


def create_synthetic_data_same_size(num_stations=int, samples=int, fake_patients=None, save=None):
    """
    Create and save synthetic data of given number of samples and number of stations. Including flag patients
    """
    fakes = random.uniform(fake_patients[0], fake_patients[1])
    samples_each = samples // num_stations
    fake_data_val = int(samples * fakes)
    fakes_at_station = fake_data_val // num_stations
    fakes_left = fakes_at_station % num_stations
    left_over = samples % num_stations
    dfs = []
    for station_i in range(num_stations):
        if station_i == range(num_stations)[-1]:  # add left number over at last stations
            samples_each = samples_each + left_over
            fakes_at_station = fakes_at_station + fakes_left

        real_data = {
            "Pre": np.random.random(size=samples_each - fakes_at_station),
            "Label": np.random.choice([0, 1], size=samples_each - fakes_at_station, p=[0.2, 0.8]),
            "Flag": np.random.choice([1], size=samples_each - fakes_at_station)
        }
        df_real = pd.DataFrame(real_data, columns=['Pre', 'Label', 'Flag'])

        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]
        prob = list(df_real['Pre'].value_counts(normalize=True, ascending=False))

        fake_data = {
                "Pre": random.choices(values, weights=prob, k=fakes_at_station),
                "Label": np.random.choice([0], size=fakes_at_station),
                "Flag": np.random.choice([0], size=fakes_at_station)
            }
        df_fake = pd.DataFrame(fake_data, columns=['Pre', 'Label', 'Flag'])

        df = [df_real, df_fake]
        merged = pd.concat(df, axis=0)
        df = merged.sample(frac=1).reset_index(drop=True)
        # plot_input_data(df, df_real, df_fake, station_i)

        df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0
        if save:
            df.to_pickle('./data/synthetic/data_s' + str(station_i + 1) + '.pkl')
        else:
            dfs.append(df)
    if not save:
        return dfs


def plot_input_data(df, df_real, df_fake, station):
    d = {"Combined": df['Pre'], "Real": df_real['Pre'], "Flag": df_fake['Pre']}
    df = pd.DataFrame(d)
    plt.style.use('ggplot')

    plt.title('Data distribution of station {}'.format(station+1))

    plt.hist(df['Real'], edgecolor='black', bins=40, color='green', rwidth=0.6, alpha=0.5, label='Real')
    plt.hist(df['Flag'], edgecolor='black',  bins=40, color='red', rwidth=0.7, alpha=0.5, label='Flag')

    plt.legend(loc='upper left')
    plt.yscale('log')
    plt.xlabel('Prediction Values')
    plt.ylabel('Subjects')

    plt.tight_layout()
    plt.savefig('plots/dist_rand_300.png')
    exit(0)


def calculate_regular_auc(stations, performance, regular_path, save, data):
    """
    Calculate AUC with sklearn as ground truth GT
    """

    if save:
        lst_df = []
        for i in range(stations):
            df_i = pickle.load(open(regular_path + '/data_s' + str(i+1) + '.pkl', 'rb'))
            lst_df.append(df_i)
    else:
        lst_df = data
    concat_df = pd.concat(lst_df)
    flags = len(concat_df[concat_df['Flag'] == 0])
    samples = len(concat_df)
    performance['samples'].append(samples)
    performance['flags'].append(samples)
    print('Use data from {} stations. Total of {} subjects (including {} flag subjects) '.format(stations,
        len(concat_df), flags))

    sort_df = concat_df.sort_values(by='Pre', ascending=False)

    filtered_df = sort_df[sort_df["Flag"] == 1]  # remove flag patients
    dfd = filtered_df.copy()
    dfd["Pre"] = filtered_df["Pre"]
    y = dfd["Label"]
    pred = dfd["Pre"]

    gt = metrics.roc_auc_score(y, pred)

    return gt, performance


def generate_keys(stations, directory, results, save):
    """
    Generate and save keys (optional - to save disk) of given numbers of stations and train results
    return: results with PKs and sk_keys = [[s_p_sk, s_rsa_sk * stations], agg_sk1, agg_sk2, agg_rsa_sk]
    """
    sk_keys = {
        's_p_sks': [],
        's_rsa_sks': [],
        'agg_sk_1': [],
        'agg_sk_2': [],
        'agg_rsa_sk': [],
    }
    for i in range(stations):
        sk, pk = generate_keypair(3072)  # paillier keys
        if save:
            pickle.dump(sk, open(directory + '/keys/s' + str(i+1) + '_paillier_sk.p', 'wb'))
            pickle.dump(pk, open(directory + '/keys/s' + str(i + 1) + '_paillier_pk.p', 'wb'))
        else:
            sk_keys['s_p_sks'].append(sk)
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

        public_pem = rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        if save:
            with open(directory + '/keys/s' + str(i + 1) + '_rsa_sk.pem', 'wb') as f:
                f.write(private_pem)
            with open(directory + '/keys/s' + str(i + 1) + '_rsa_pk.pem', 'wb') as f:
                f.write(public_pem)
        else:
            sk_keys['s_rsa_sks'].append(private_pem)
        results['stations_rsa_pk'][i] = public_pem

    # generate keys of aggregator
    sk, pk = generate_keypair(3072)
    sk_1 = copy.copy(sk)
    sk_2 = copy.copy(sk)
    # simulate private key separation
    del sk_1.x2
    del sk_2.x1
    if save:
        pickle.dump(sk_1, open(directory + '/keys/agg_sk_1.p', 'wb'))
        pickle.dump(sk_2, open(directory + '/keys/agg_sk_2.p', 'wb'))
        pickle.dump(pk, open(directory + '/keys/agg_pk.p', 'wb'))
    else:
        sk_keys['agg_sk_1'] = sk_1
        sk_keys['agg_sk_2'] = sk_2
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

    public_pem = rsa_public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    if save:
        with open(directory + '/keys/agg_rsa_private_key.pem', 'wb') as f:
            f.write(private_pem)
        with open(directory + '/keys/agg_rsa_public_key.pem', 'wb') as f:
            f.write(public_pem)
    else:
        sk_keys['agg_rsa_sk'] = private_pem
    results['aggregator_rsa_pk'] = public_pem

    return results, sk_keys


def encrypt_table(station_df, agg_pk, r1, r2, symmetric_key):
    """
    Encrypt dataframe of given station dataframe with paillier public key of aggregator and random values
    """
    station_df["Pre"] *= r1
    station_df["Pre"] += r2

    station_df["Pre"] = station_df["Pre"].apply(lambda x: Fernet(symmetric_key).encrypt(struct.pack("f", x)))
    station_df["Label"] = station_df["Label"].apply(lambda x: encrypt(agg_pk, x))
    station_df["Flag"] = station_df["Flag"].apply(lambda x: encrypt(agg_pk, x))
    return station_df


def load_rsa_sk(path, save, keys):
    """
    Return private rsa key of given file path
    """
    if save:
        with open(path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )
    else:
        private_key = load_pem_private_key(
            keys['agg_rsa_sk'],
            password=None,
            backend=default_backend()
        )
    return private_key


def load_rsa_pk(path, save, results):
    """
    Return public rsa key of given file path
    """
    if save:
        with open(path, "rb") as key_file:
            public_key = serialization.load_pem_public_key(key_file.read(), backend=default_backend())
    else:
        public_key = load_pem_public_key(results['aggregator_rsa_pk'], backend=default_backend())
    return public_key


def encrypt_symmetric_key(symmetric_key, directory, save, results):
    """
    Encrypt symmetric key_station with public rsa key of aggregator
    return: encrypted_symmetric_key
    """
    path = directory + '/keys/agg_rsa_public_key.pem'
    rsa_agg_pk = load_rsa_pk(path, save, results)
    encrypted_symmetric_key = rsa_agg_pk.encrypt(symmetric_key, padding.OAEP(
                                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                algorithm=hashes.SHA256(),
                                                label=None
                                            ))

    return encrypted_symmetric_key


def decrypt_symmetric_key(ciphertext, directory, save, keys):
    """
    Decrypt of given station rsa encrypted k_station
    """
    path = directory + '/keys/agg_rsa_private_key.pem'
    rsa_agg_sk = load_rsa_sk(path, save, keys)

    decrypted_symmetric_key = rsa_agg_sk.decrypt(
        ciphertext,
        padding.OAEP(
         mgf=padding.MGF1(algorithm=hashes.SHA256()),
         algorithm=hashes.SHA256(),
         label=None
        ))
    return decrypted_symmetric_key


def pp_auc_protocol(station_df, prev_results, directory=str, station=int, max_value=int, save_data=None, save_keys=None,
                    keys=None):
    """
    Perform PP-AUC protocol at specific station given dataframe
    """
    agg_pk = prev_results['aggregator_paillier_pk']
    symmetric_key = Fernet.generate_key()  # represents k1 k_n
    if station == 1:
        r1 = randint(1, max_value)
        r2 = randint(1, max_value)

        enc_table = encrypt_table(station_df, agg_pk, r1, r2, symmetric_key)
        # Save for transparency the table - not required
        if save_data:
            enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, directory, save_keys, prev_results)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1)  # homomorphic encryption used
            enc_r2 = encrypt(prev_results['stations_paillier_pk'][i], r2)
            prev_results['encrypted_r1'][i] = enc_r1
            prev_results['encrypted_r2'][i] = enc_r2

    else:
        enc_r1 = prev_results['encrypted_r1'][station-1]
        if save_keys:
            sk_s_i = pickle.load(open(directory + '/keys/s' + str(station) + '_paillier_sk.p', 'rb'))
        else:
            sk_s_i = keys['s_p_sks'][station-1]

        dec_r1 = decrypt(sk_s_i, enc_r1)
        enc_r2 = prev_results['encrypted_r2'][station-1]
        dec_r2 = decrypt(sk_s_i, enc_r2)

        enc_table = encrypt_table(station_df, agg_pk, dec_r1, dec_r2, symmetric_key)
        if save_data:
            enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, directory, save_keys, prev_results)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

    prev_results['pp_auc_tables'][station-1] = enc_table

    return prev_results


def z_values(n):
    """
    Generate random values of list length n which sum is zero
    """
    l = random.sample(range(-int(n/2), int(n/2)), k=n-1)
    return l + [-sum(l)]


def dppe_auc_proxy(directory, results, max_value, save_keys, keys):
    """
    Simulation of aggregator service - globally computes privacy preserving AUC table as proxy station
    """
    agg_pk = results['aggregator_paillier_pk']

    if save_keys:
        agg_sk = pickle.load(open(directory + '/keys/agg_sk_1.p', 'rb'))
    else:
        agg_sk = keys['agg_sk_1']
    df_list = []
    for i in range(len(results['encrypted_ks'])):
        enc_k_i = results['encrypted_ks'][i]
        dec_k_i = decrypt_symmetric_key(enc_k_i, directory, save=save_keys, keys=keys)

        # decrypt table values with Fernet and corresponding k_i symmetric key
        table_i = results['pp_auc_tables'][i]
        table_i["Dec_pre"] = table_i["Pre"].apply(lambda x: Fernet(dec_k_i).decrypt(x))  # returns bytes
        d = table_i["Dec_pre"].apply(lambda x: struct.unpack('f', x)).to_list()
        lst = [x[0] for x in d]
        table_i["Dec_pre"] = lst
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

    a = randint(1, max_value)
    b = randint(1, max_value)
    # Denominator
    # TP_A is summation of labels (TP)
    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)

    r_1A = randint(1, max_value)
    r_2A = randint(1, max_value)
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
    print('Threshold values: {}'.format(len_t))
    # Multiply with a and b respectively
    Z_values = z_values(len_t)

    # sum over all n_3 and only store n_3
    N_3_sum = encrypt(agg_pk, 0)
    for i in range(1, len_t + 1):
        pre_ind = thre_ind[i - 1]
        if i == len_t:
            cur_ind = -1
        else:
            cur_ind = thre_ind[i]
        # Multiply with a and b respectively
        sTP_a = mul_const(agg_pk, add(agg_pk, tp_values[cur_ind], tp_values[pre_ind]), a)
        dFP_b = mul_const(agg_pk, add(agg_pk, fp_values[cur_ind], mul_const(agg_pk, fp_values[pre_ind], -1)), b)
        r1_i = randint(1, max_value)
        r2_i = randint(1, max_value)
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


def dppe_auc_station_final(directory, train_results, save_keys, keys):
    """
    Simulation of station delegated AUC parts to compute global DPPE-AUC locally
    """
    if save_keys:
        agg_sk_2 = pickle.load(open(directory + '/keys/agg_sk_2.p', 'rb'))
    else:
        agg_sk_2 = keys['agg_sk_2']
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


def plot_experiment_1(res):
    total_time = [sum(x) for x in zip(*[res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2']])]
    df = pd.DataFrame(list(zip(res['time']['stations_1'], res['time']['proxy'],
                               res['time']['stations_2'], total_time, res['samples'], res['stations'])),
                      index=res['stations'],
                      columns=['Station_1', 'Proxy', 'Station_2', 'Total', 'Samples', 'Stations'])

    df['Station_2'] = df['Station_2'].multiply(df['Stations']) # multiply last step by number of stations
    b_plot = df.boxplot(column='Total', by='Stations', grid=False)
    plt.title(str(10) + ' runs with ' + str(res['samples'][0]) + ' subjects')
    plt.suptitle('')  # remove prev title
    b_plot.set_ylabel('time (sec)')
    b_plot.plot()
    plt.show()
    plt.tight_layout()
    plt.savefig('plots/exp1.png')


def plot_experiment_2(res):
    total_time = [sum(x) for x in zip(*[res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2']])]
    df = pd.DataFrame(list(zip(res['time']['stations_1'], res['time']['proxy'],
                               res['time']['stations_2'], total_time, res['samples'], res['stations'])),
                      index=res['stations'],
                      columns=['Station_1', 'Proxy', 'Station_2', 'Total', 'Samples', 'Stations'])

    df['Station_2'] = df['Station_2'].multiply(df['Stations'])
    c = plt.cm.Set2
    color = iter(c.colors)
    for category in df.Stations.unique():
        c = next(color)
        plt.plot('Samples', 'Total', c=c, data=df.loc[df['Stations'].isin([category])], marker='o',
                 label=str(category) + ' stations')
    plt.xlabel('Number of subjects')
    num_stations = res['stations'][0]
    plt.ylabel('Time (sec)')
    plt.title('DPPE-AUC total runtime evaluation with ' + str(num_stations) + ' stations')
    plt.legend(loc="upper left")
    plt.savefig('plots/exp2.png')


if __name__ == "__main__":
    """
    Run with either complete experiment 1 or 2 uncommented
    """

    DIRECTORY = './data'

    SIMULATE_PUSH_PULL = False
    SAVE_DATA = False
    SAVE_KEYS = False

    MAX = 100
    FAKES = [0.1, 0.6]  # percentage range for random values

    EXPERIMENT_1 = True  # one of them must be true
    EXPERIMENT_2 = False

    if EXPERIMENT_1:
        # Experiment 1
        station_list = [3, 6, 9]
        subject_list = [1500]
        loops = 10
    elif EXPERIMENT_2:
        # Experiment 2
        station_list = [3]
        subject_list = [30, 90, 180, 360, 720, 1440, 2880, 5760, 11520]
        loops = 1

    per = {'time':
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
                per['stations'].append(stations)
                try:
                    shutil.rmtree(DIRECTORY + '/')
                except Exception as e:
                    pass
                directories = []
                if SAVE_DATA:
                    directories = [DIRECTORY, DIRECTORY + '/synthetic', DIRECTORY + '/encrypted']
                elif SAVE_KEYS:
                    directories.append(DIRECTORY)
                    directories.append(DIRECTORY + '/keys')
                elif SIMULATE_PUSH_PULL:
                    directories.append(DIRECTORY)
                    directories.append(DIRECTORY + '/pht_results')

                for dir in directories:
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                # Experiment 1 - increase number of stations, but same sample size
                if SAVE_DATA:
                    if EXPERIMENT_1:
                        create_synthetic_data_same_size(stations, subjects, [FAKES[0], FAKES[1]], SAVE_DATA)
                    elif EXPERIMENT_2:
                        create_synthetic_data(stations, subjects, [FAKES[0], FAKES[1]], SAVE_DATA)

                    data = {}
                else:
                    if EXPERIMENT_1:
                        data = create_synthetic_data_same_size(stations, subjects, [FAKES[0], FAKES[1]], SAVE_DATA)
                    elif EXPERIMENT_2:
                        data = create_synthetic_data(stations, subjects, [FAKES[0], FAKES[1]], SAVE_DATA)

                results = train.load_results()
                results, keys = generate_keys(stations, DIRECTORY, results, save=SAVE_KEYS)
                if SIMULATE_PUSH_PULL:
                    train.save_results(results)

                # compute AUC without encryption for proof of concept
                REGULAR_PATH = DIRECTORY + '/synthetic'
                auc_gt, per = calculate_regular_auc(stations, per, REGULAR_PATH, SAVE_DATA, data)
                per['gt-auc'].append(auc_gt)
                print('AUC value of GT {}'.format(auc_gt))
                times = []
                for i in range(stations):
                    if SAVE_DATA:
                        stat_df = pickle.load(open(DIRECTORY + '/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))
                    else:
                        stat_df = data[i]

                    if SIMULATE_PUSH_PULL:
                        results = train.load_results()  # loading results simulates pull of image

                    if stat_df['Pre'].dtype == 'float64':
                        results['floats'].insert(0, True)
                    else:
                        results['floats'].insert(0, False)

                    t1 = time.perf_counter()
                    results = pp_auc_protocol(stat_df, results, DIRECTORY, station=i+1, max_value=MAX,
                                              save_data=SAVE_DATA, save_keys=SAVE_KEYS, keys=keys)
                    t2 = time.perf_counter()
                    times.append(t2 - t1)
                    print('Station {} step 1 time {}'.format(i + 1, times[-1]))
                    # remove at last station all encrypted noise values
                    if i is stations - 1:
                        results.pop('encrypted_r1')
                        results.pop('encrypted_r2')

                    if SIMULATE_PUSH_PULL:
                        train.save_results(results)

                print(f'Total execution time at stations - Step 1 {sum(times):0.4f} seconds')
                print(f'Average execution time at stations - Step 1 {sum(times)/len(times):0.4f} seconds')
                per['time']['stations_1'].append(sum(times) / len(times))
                per['time']['total_step_1'].append(sum(times))

                if SIMULATE_PUSH_PULL:
                    results = train.load_results()

                t3 = time.perf_counter()
                results = dppe_auc_proxy(DIRECTORY, results, max_value=MAX, save_keys=SAVE_KEYS, keys=keys)
                t4 = time.perf_counter()

                per['time']['proxy'].append(t4 - t3)
                print(f'Execution time by proxy station {t4 - t3:0.4f} seconds')

                if SIMULATE_PUSH_PULL:
                    train.save_results(results)
                    results = train.load_results()

                for i in range(1):
                    t1 = time.perf_counter()
                    auc_pp = dppe_auc_station_final(DIRECTORY, results, SAVE_KEYS, keys)
                    t2 = time.perf_counter()
                    local_dppe = t2 - t1
                per['time']['stations_2'].append(local_dppe)
                print(f'Final AUC execution time at station {i+1} {local_dppe:0.4f} seconds')

                per['pp-auc'].append(auc_pp)

                diff = auc_gt - auc_pp
                differences.append(diff)
                print('Difference pp-AUC to GT: ', diff)
                print('\n')
            print("Avg difference {} over {} runs".format(sum(differences)/len(differences), len(differences)))
    print(per)

    if EXPERIMENT_1:
        plot_experiment_1(per)
    elif EXPERIMENT_2:
        plot_experiment_2(per)
    