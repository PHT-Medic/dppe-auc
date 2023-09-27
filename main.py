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
from FHM_approx import dppa_auc_protocol, dppa_auc_proxy, create_synthetic_data_dppa


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
            return {'approx': {'enc_rx': {},
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
        real_data = {
            "Pre": np.random.random(size=samples_each),
            "Label": np.random.choice([0, 1], size=samples_each, p=[0.2, 0.8]),
            "Flag": np.random.choice([1], size=samples_each)
        }
        df_real = return_df(real_data)
        # tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        # values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]
        # prob = list(df_real['Pre'].value_counts(normalize=True, ascending=False))
        #
        # fake_data = {
        #     "Pre": random.choices(values, weights=prob, k=fake_data_val),
        #     "Label": np.random.choice([0], size=fake_data_val),
        #     "Flag": np.random.choice([0], size=fake_data_val)
        # }

        #
        #
        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]  # unique values
        counts = list(df_real['Pre'].value_counts(ascending=False))
        max_a = counts[0] + int(counts[0] * 0.1)
        v = [max_a - counts[i] for i in range(len(counts))]  # probabilities
        s = pd.Series(np.repeat(values[i], v[i]) for i in range(len(v)))
        list_fakes = s.explode(ignore_index=True)
        fakes = len(list_fakes)
        # tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        # values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]  # unique values
        # counts = list(df_real['Pre'].value_counts(ascending=False))
        # max_a = counts[0] + int(counts[0] * 0.1)
        # v = [max_a - counts[i] for i in range(len(counts))]  # probabilities
        # s = pd.Series(np.repeat(values[i], v[i]) for i in range(len(v)))
        # list_fakes = s.explode(ignore_index=True)
        # fakes = len(list_fakes)

        fake_data = {"Pre": list_fakes,
                     "Label": np.random.choice([0], size=fakes),
                     "Flag": np.random.choice([0], size=fakes)
                     }
        df_fake = return_df(fake_data)

        df = [df_real, df_fake]
        merged = pd.concat(df, axis=0)
        df = merged.sample(frac=1).reset_index(drop=True)
        plot_input_data(df, df_real, df_fake, station_i, proxy=False)

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
    samples_each = (samples // num_stations) // 2
    fake_data_val = samples // 2
    fakes_at_station = fake_data_val // num_stations
    fakes_left = fakes_at_station % num_stations
    left_over = samples % num_stations
    dfs = []
    for station_i in range(num_stations):
        if station_i == range(num_stations)[-1]:  # add left number over at last stations
            samples_each = samples_each + left_over
            fakes_at_station = fakes_at_station + fakes_left

        real_data = {
            "Pre": np.random.random(size=samples_each),
            "Label": np.random.choice([0, 1], size=samples_each, p=[0.2, 0.8]),
            "Flag": np.random.choice([1], size=samples_each)
        }
        df_real = pd.DataFrame(real_data, columns=['Pre', 'Label', 'Flag'])
        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]  # unique values
        counts = list(df_real['Pre'].value_counts(ascending=False))
        highest = counts[0] + int(counts[0] * 0.4)
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
        # print("Size fake: {}".format(len(df_fake)))

        df = [df_real, df_fake]

        merged = pd.concat(df, axis=0)
        df = merged.sample(frac=1).reset_index(drop=True)
        plot_input_data(df, df_real, df_fake, station_i, run, proxy=False)

        df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0
        if save:
            df.to_pickle('./data/synthetic/data_s' + str(station_i + 1) + '.pkl')
        else:
            dfs.append(df)
    if not save:
        return dfs


def plot_input_data(df, df_real, df_fake, station, run, proxy=None):
    if proxy:
        plt.clf()
        plt.style.use('ggplot')
        plt.title('Run ' + str(run) + ' Data distribution at proxy')
        plt.hist(df['Dec_pre'], edgecolor='black', bins=40, color='orange', rwidth=0.6,
                 alpha=0.5, label='Obscured')
        plt.legend(loc='upper left')
        plt.yscale('log')
        plt.xlabel('Obscured prediction value')
        plt.ylabel('Subjects')
        plt.tight_layout()
        plt.show()
        # plt.savefig('plots/proxy.png')
    else:
        d = {'Combined': df['Pre'], "Real": df_real['Pre'], "Flag": df_fake['Pre']}
        df_p = pd.DataFrame(d)
        plt.clf()
        plt.style.use('ggplot')
        plt.title('Run ' + str(run) + ' Data distribution of station {}'.format(station + 1))
        plt.hist([df_p['Real'], df_p['Flag']], edgecolor='black', bins=40, color=['green', 'red'], stacked=True,
                 rwidth=0.6,
                 alpha=0.5, label=['Real', 'Flag'])
        plt.legend(loc='upper left')
        plt.yscale('log')
        plt.xlabel('Prediction Values')
        plt.ylabel('Subjects')

        plt.tight_layout()
        plt.show()
        # plt.savefig('plots/s_' + str(station+1)+'.png')


def calculate_regular_auc(stations, performance, regular_path, save, data, APPROX):
    """
    Calculate AUC with sklearn as ground truth GT
    """

    if save:
        lst_df = []
        for i in range(stations):
            df_i = pickle.load(open(regular_path + '/data_s' + str(i + 1) + '.pkl', 'rb'))
            lst_df.append(df_i)
    else:
        lst_df = data
    concat_df = pd.concat(lst_df)

    samples = len(concat_df)
    performance['samples'].append(samples)

    sort_df = concat_df.sort_values(by='Pre', ascending=False)
    if APPROX:
        performance['flags'].append(0)
        filtered_df = sort_df
        print('Use data from {} stations. Total of {} subjects (including 0 flag subjects) '.format(stations,
                                                                                                    len(filtered_df)))
    else:
        flags = len(concat_df[concat_df['Flag'] == 0])
        performance['flags'].append(samples)
        print('Use data from {} stations. Total of {} subjects (including {} flag subjects) '.format(stations,
                                                                                                     len(concat_df),
                                                                                                     flags))
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
            pickle.dump(sk, open(directory + '/keys/s' + str(i + 1) + '_paillier_sk.p', 'wb'))
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


def encrypt_table(s_df, agg_pk, r1, symmetric_key):
    """
    Encrypt dataframe of given station dataframe with paillier public key of aggregator and random values
    """
    s_df = s_df.copy()
    r2_values = s_df["Pre"]
    r2s = (r2_values * 10000) % r1

    s_df["Pre"] *= r1
    s_df["Pre"] += r2s
    s_df["Pre"] = s_df["Pre"].apply(lambda x: Fernet(symmetric_key).encrypt(struct.pack("f", x)))
    s_df["Label"] = s_df["Label"].apply(lambda x: encrypt(agg_pk, x))
    s_df["Flag"] = s_df["Flag"].apply(lambda x: encrypt(agg_pk, x))
    return s_df


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


def dppe_auc_protocol(local_df, prev_results, directory=str, station=int, max_value=int, save_data=None, save_keys=None,
                      keys=None):
    """
    Perform PP-AUC protocol at specific station given dataframe
    """
    agg_pk = prev_results['aggregator_paillier_pk']
    symmetric_key = Fernet.generate_key()  # represents k1 k_n
    if station == 1:
        r1 = randint(20000, max_value)
        print("rand r_1 {}".format(r1))
        enc_table = encrypt_table(local_df, agg_pk, r1, symmetric_key)
        if save_data:  # Save for transparency the table - not required
            enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, directory, save_keys, prev_results)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1)  # homomorphic encryption used
            prev_results['encrypted_r1'][i] = enc_r1

    else:
        enc_r1 = prev_results['encrypted_r1'][station - 1]
        if save_keys:
            sk_s_i = pickle.load(open(directory + '/keys/s' + str(station) + '_paillier_sk.p', 'rb'))
        else:
            sk_s_i = keys['s_p_sks'][station - 1]

        dec_r1 = decrypt(sk_s_i, enc_r1)

        enc_table = encrypt_table(local_df, agg_pk, dec_r1, symmetric_key)
        if save_data:
            enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, directory, save_keys, prev_results)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

    prev_results['pp_auc_tables'][station - 1] = enc_table

    return prev_results


def z_values(n):
    """
    Generate random values of list length n which sum is zero
    """
    l = random.sample(range(-int(n / 2), int(n / 2)), k=n - 1)
    return l + [-sum(l)]


def dppe_auc_proxy(directory, results, max_value, save_keys, run, keys):
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
    print("len_df: ", len(df_new_index))
    plot_input_data(df_new_index, None, None, None, run, proxy=True)
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
    print("rand a {}".format(a))
    print("rand b {}".format(b))
    # Denominator
    # TP_A is summation of labels (TP)
    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)

    r_1A = randint(1, max_value)
    r_2A = randint(1, max_value)
    print("rand r_1A {}".format(r_1A))
    print("rand r_2A {}".format(r_2A))
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
    thre_ind.insert(0, 0)
    len_t = len(thre_ind)
    print("len_tresholds: ", len_t)
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


def pp_auc_station_final(directory, train_results, save_keys, keys, APPROX):
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
        n_1_mul_n_2 = mul_const(agg_pk, train_results['N2'][j], n_i1)
        if j == 0:
            sum_n_1_mul_2 = n_1_mul_n_2
        else:
            sum_n_1_mul_2 = add(agg_pk, sum_n_1_mul_2, n_1_mul_n_2)

    E_N = add(agg_pk, sum_n_1_mul_2, mul_const(agg_pk, train_results['N3'][0], -1))
    N = station_decrypt(agg_sk_2, E_N)

    D = (D1 * D2) - D3
    if D == 0:
        auc = 0
    else:
        auc = (N / D) / 2
    if APPROX:
        print('DPPA-AUC: {}'.format(auc))
    else:
        print('DPPE-AUC: {}'.format(auc))
    return auc


def plot_experiment_1(res):

    options = ['approx', 'exact']

    for i in range(len(options)):
        res_part = res[options[i]]

        total_time = [sum(x) for x in zip(*[res_part['time']['total_step_1'], res_part['time']['proxy'], res_part['time']['stations_2']])]
        df = pd.DataFrame(list(zip(res_part['time']['stations_1'], res_part['time']['proxy'],
                                   res_part['time']['stations_2'], total_time, res_part['samples'], res_part['stations'])),
                          index=res_part['stations'],
                          columns=['Station_1', 'Proxy', 'Station_2', 'Total', 'Samples', 'Stations'])

        df['Station_2'] = df['Station_2'].multiply(df['Stations'])  # multiply last step by number of stations
        b_plot = df.boxplot(column='Total', by='Stations', grid=False)
        plt.suptitle('')  # remove prev title
        b_plot.set_ylabel('time (sec)')
        plt.title('Option ' + options[i] + ' ' + str(len(df['Station_2'])) + ' runs with ' + str(res_part['samples'][0]) + ' subjects')

        b_plot.plot()

    plt.tight_layout()
    plt.show()
    # plt.savefig('plots/exp1.png')


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

    print("Comparing both approaches in same run")

    MAX = 100000

    no_of_decision_points = 200
    FAKES = [0.1, 0.6]  # percentage range for random values

    EXPERIMENT_1 = True  # one of them must be true
    EXPERIMENT_2 = False

    if EXPERIMENT_1:
        # Experiment 1
        station_list = [3]
        subject_list = [1500]
        loops = 100
    elif EXPERIMENT_2:
        # Experiment 2
        station_list = [3]
        subject_list = [30, 90, 180]  # , 360, 720, 1440, 2880, 5760, 11520]
        loops = 1

    per = {'approx': {'time':
                          {'stations_1': [],
                           'proxy': [],
                           'stations_2': [],
                           'total_step_1': []
                           },
                      'total_times': [],  # total time for each run
                      'samples': [],
                      'flags': [],
                      'stations': [],
                      'pp-auc': [],
                      'gt-auc': [],
                      'diff': []
                      },
           'exact': {'time':
                         {'stations_1': [],
                          'proxy': [],
                          'stations_2': [],
                          'total_step_1': []
                          },
                     'total_times': [],  # total time for each run
                     'samples': [],
                     'flags': [],
                     'stations': [],
                     'pp-auc': [],
                     'gt-auc': [],
                     'diff': []
                     }}
    #per = {'approx': {'time': {'stations_1': [0.1984941113333356, 0.19385406900000626, 0.1914984443333386, 0.19211002766667207, 0.1999794306666066, 0.19325626366662618, 0.19136655533335065, 0.19200273599994944, 0.19133688866660728, 0.1913567776666696], 'proxy': [16.345045499999998, 16.33334658299998, 16.25011374999997, 16.21170812500003, 16.234686874999966, 16.238068999999996, 16.239693583000076, 16.289057875000026, 16.22380308300012, 16.239371458999813], 'stations_2': [24.382013498999953, 23.87916787500012, 23.91782737500006, 23.869032749999747, 23.901585501, 23.823067499999752, 23.837265249000325, 23.84140862700019, 23.894918877000237, 23.935468248000006], 'total_step_1': [0.5954823340000068, 0.5815622070000188, 0.5744953330000158, 0.5763300830000162, 0.5999382919998197, 0.5797687909998785, 0.5740996660000519, 0.5760082079998483, 0.5740106659998219, 0.5740703330000088]}, 'total_times': [40.92555311033328, 40.406368527000105, 40.35943956933337, 40.27285090266645, 40.336251806666574, 40.254392763666374, 40.26832538733375, 40.32246923800017, 40.31005884866696, 40.36619648466649], 'samples': [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500], 'flags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'stations': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'pp-auc': [0.49298379261614556, 0.4991126056699827, 0.49842925714285713, 0.4925396801982535, 0.5044916804342695, 0.5144220819536424, 0.5072134144981827, 0.4932730086915109, 0.5123831189934538, 0.5092118028310619], 'gt-auc': [0.49300771359594886, 0.4990229321376862, 0.49845485714285714, 0.4924843639367477, 0.5045101191881047, 0.5144424077578051, 0.5071688184270966, 0.49339579116561494, 0.5123404057755699, 0.5091802682663965], 'diff': [2.3920979803304654e-05, -8.9673532296497e-05, 2.56000000000145e-05, -5.5316261505788944e-05, 1.8438753835225974e-05, 2.0325804162779626e-05, -4.459607108608932e-05, 0.00012278247410402177, -4.271321788384963e-05, -3.153456466542526e-05]}, 'exact': {'time': {'stations_1': [0.2916688750000003, 0.28950843033333246, 0.2893862363333142, 0.2858041669999996, 0.29640830533332974, 0.2875637223333645, 0.28670026400004645, 0.2869573473332518, 0.28676723599998394, 0.2869977223333535], 'proxy': [62.774634999999996, 62.087768082999986, 61.53447670899999, 61.400370167000005, 61.27620329199999, 61.401696124999944, 61.49382404200003, 61.60768912499998, 61.36197766700002, 61.41187754199996], 'stations_2': [91.46530937400003, 88.86134525100002, 88.28130950099995, 88.0645286250001, 88.23527137500002, 88.16915687400001, 88.13623899899994, 88.29429312299999, 88.23463137599992, 88.40376862500034], 'total_step_1': [0.875006625000001, 0.8685252909999974, 0.8681587089999425, 0.8574125009999989, 0.8892249159999892, 0.8626911670000936, 0.8601007920001393, 0.8608720419997553, 0.8603017079999518, 0.8609931670000606]}, 'total_times': [154.53161324900003, 151.23862176433335, 150.10517244633326, 149.7507029590001, 149.80788297233335, 149.85841672133333, 149.91676330500002, 150.1889395953332, 149.88337627899992, 150.10264388933365], 'samples': [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500], 'flags': [1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500, 1500], 'stations': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'pp-auc': [0.4775030785260964, 0.4972908572908573, 0.494592, 0.47639455151964416, 0.5141655856189771, 0.5443164292842239, 0.5208171458171458, 0.48102334929866575, 0.5360260233125508, 0.5270907916488762], 'gt-auc': [0.47750307852609647, 0.4972908572908573, 0.494592, 0.47639455151964416, 0.5141655856189771, 0.5443164292842239, 0.5208171458171458, 0.48102334929866575, 0.5360260233125508, 0.5270907916488763], 'diff': [5.551115123125783e-17, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1102230246251565e-16]}}
    #plot_experiment_1(per)
    #exit(0)
    decision_points = np.linspace(0, 1, num=no_of_decision_points)[::-1]
    differences_approx, differences_exact = [], []
    data_approx, data_exact = [], []
    for subjects in subject_list:
        train = Train(results='results.pkl')

        for stations in station_list:
            for i in range(loops):  # repeat n times, to make boxplot
                run = i
                per['approx']['stations'].append(stations)
                per['exact']['stations'].append(stations)
                try:
                    # shutil.rmtree(DIRECTORY + '/')
                    print("\nnew run")
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
                        exact_data = create_synthetic_data_same_size(stations, subjects, [FAKES[0], FAKES[1]],
                                                                     SAVE_DATA)
                        approx_data = create_synthetic_data_dppa(stations, exact_data, SAVE_DATA)
                    elif EXPERIMENT_2:
                        data = create_synthetic_data(stations, subjects, [FAKES[0], FAKES[1]], SAVE_DATA)

                data_approx.append(approx_data.copy())
                data_exact.append(exact_data.copy())

                results = train.load_results()
                results['exact'], keys_exact = generate_keys(stations, DIRECTORY, results['exact'], save=SAVE_KEYS)
                results['approx'], keys_approx = generate_keys(stations, DIRECTORY, results['approx'], save=SAVE_KEYS)
                if SIMULATE_PUSH_PULL:
                    train.save_results(results)

                # compute AUC without encryption for proof of concept
                REGULAR_PATH = DIRECTORY + '/synthetic'
                times_exact, times_approx = [], []

                for i in range(stations):
                    if SAVE_DATA:
                        stat_df = pickle.load(open(DIRECTORY + '/synthetic/data_s' + str(i + 1) + '.pkl', 'rb'))
                    else:
                        exact_stat_df = exact_data[i]
                        approx_stat_df = approx_data[i]

                    if SIMULATE_PUSH_PULL:
                        results = train.load_results()  # loading results simulates pull of image

                    t1 = time.perf_counter()
                    results_approx = dppa_auc_protocol(approx_stat_df, decision_points, results['approx'], DIRECTORY,
                                                       station=i + 1,
                                                       max_value=MAX, save_data=SAVE_DATA, save_keys=SAVE_KEYS,
                                                       keys=keys_approx)
                    t2 = time.perf_counter()
                    times_approx.append(t2 - t1)

                    t_1 = time.perf_counter()
                    results_exact = dppe_auc_protocol(exact_stat_df, results['exact'], DIRECTORY, station=i + 1,
                                                      max_value=MAX,
                                                      save_data=SAVE_DATA, save_keys=SAVE_KEYS, keys=keys_exact)
                    t_2 = time.perf_counter()
                    times_exact.append(t_2 - t_1)
                    print('Exact Station {} step 1 time {}'.format(i + 1, times_exact[-1]))
                    print('Approx Station {} step 1 time {}'.format(i + 1, times_approx[-1]))

                    # remove at last station all encrypted noise values
                    if i is stations - 1:
                        results["approx"].pop('encrypted_r1')
                        results["exact"].pop('encrypted_r1')

                    if SIMULATE_PUSH_PULL:
                        train.save_results(results)

                print(f'Exact run {run} total execution time at stations - Step 1 {sum(times_exact):0.4f} seconds')
                print(f'Exact run {run} average execution time at stations - Step 1 {sum(times_exact) / len(times_exact):0.4f} seconds')
                print(f'Approx run {run} total execution time at stations - Step 1 {sum(times_approx):0.4f} seconds')
                print(f'Approx run {run} average execution time at stations - Step 1 {sum(times_approx) / len(times_approx):0.4f} seconds')

                per['approx']['time']['stations_1'].append(sum(times_approx) / len(times_approx))
                per['approx']['time']['total_step_1'].append(sum(times_approx))
                per['exact']['time']['stations_1'].append(sum(times_exact) / len(times_exact))
                per['exact']['time']['total_step_1'].append(sum(times_exact))
                if SIMULATE_PUSH_PULL:
                    results = train.load_results()

                auc_gt_approx, per['approx'] = calculate_regular_auc(stations, per['approx'], REGULAR_PATH, save=False,
                                                                     data=approx_data, APPROX=True)
                print('Approx GT-AUC: ', auc_gt_approx)
                auc_gt_exact, per['exact'] = calculate_regular_auc(stations, per['exact'], REGULAR_PATH, save=False,
                                                                   data=exact_data, APPROX=False)
                print('Exact GT-AUC: ', auc_gt_exact)

                t3 = time.perf_counter()
                approx_results = dppa_auc_proxy(DIRECTORY, results["approx"], max_value=MAX, save_keys=SAVE_KEYS,
                                                keys=keys_approx,
                                                no_dps=no_of_decision_points)
                t4 = time.perf_counter()
                per["approx"]['time']['proxy'].append(t4 - t3)
                t3 = time.perf_counter()
                exact_results = dppe_auc_proxy(DIRECTORY, results['exact'], max_value=MAX, save_keys=SAVE_KEYS, run=run,
                                               keys=keys_exact)
                t4 = time.perf_counter()
                per["exact"]['time']['proxy'].append(t4 - t3)

                print(f'Exact execution time by proxy station {per["exact"]["time"]["proxy"][-1]:0.4f} seconds')
                print(f'Approx execution time by proxy station {per["approx"]["time"]["proxy"][-1]:0.4f} seconds')

                if SIMULATE_PUSH_PULL:
                    train.save_results(results)
                    results = train.load_results()

                t1 = time.perf_counter()
                auc_pp_exact = pp_auc_station_final(DIRECTORY, results['exact'], SAVE_KEYS, keys_exact, APPROX=False)
                t2 = time.perf_counter()
                local_dppe = t2 - t1
                per['exact']['time']['stations_2'].append(local_dppe * stations)  # total time for last step
                print(f'Exact final AUC execution time at one station {i + 1} {local_dppe:0.4f} seconds')

                t1 = time.perf_counter()
                auc_pp_approx = pp_auc_station_final(DIRECTORY, results['approx'], SAVE_KEYS, keys_approx, APPROX=True)
                t2 = time.perf_counter()
                local_dppa = t2 - t1

                per['approx']['time']['stations_2'].append(local_dppa * stations)  # total time for last step
                print(f'Exact final AUC execution time at one station {i + 1} {local_dppe:0.4f} seconds')

                total_time_exact = per['exact']["time"]["proxy"][-1] + per['exact']["time"]["stations_2"][-1] + \
                                   per['exact']["time"]["stations_1"][-1]
                per['exact']['total_times'].append(total_time_exact)
                print(f'Exact final total exec time: {total_time_exact:0.4f} seconds')

                total_time_approx = per['approx']["time"]["proxy"][-1] + per['approx']["time"]["stations_2"][-1] + \
                                    per['approx']["time"]["stations_1"][-1]
                per['approx']['total_times'].append(total_time_approx)
                print(f'Approx final total exec time: {total_time_approx:0.4f} seconds')

                per['approx']['pp-auc'].append(auc_pp_approx)
                per['exact']['pp-auc'].append(auc_pp_exact)

                per['approx']['gt-auc'].append(auc_gt_approx)
                per['exact']['gt-auc'].append(auc_gt_exact)

                diff_exact = auc_gt_exact - auc_pp_exact
                differences_exact.append(diff_exact)
                per['exact']['diff'].append(diff_exact)

                diff_approx = auc_gt_approx - auc_pp_approx
                differences_approx.append(diff_approx)
                per['approx']['diff'].append(diff_approx)

                print('Difference DPPE-AUC (exact)  to GT: ', diff_exact)
                print('Difference DPPA-AUC (approx) to GT: ', diff_approx)
                print('\n')

                print("Exact avg difference {} over {} runs".format(sum(differences_exact) / len(differences_exact),
                                                                    len(differences_exact)))
                print("Exact avg exec time {} over {} runs".format(
                    sum(per['exact']['total_times']) / len(per['exact']['total_times']),
                    len(per['exact']['total_times'])))

                print("Approx avg difference {} over {} runs".format(sum(differences_approx) / len(differences_approx),
                                                                     len(differences_approx)))
                print("Approx avg exec time {} over {} runs".format(
                    sum(per['approx']['total_times']) / len(per['approx']['total_times']),
                    len(per['approx']['total_times'])))

    print(per)

    if EXPERIMENT_1:
        plot_experiment_1(per)
    elif EXPERIMENT_2:
        plot_experiment_2(per)
