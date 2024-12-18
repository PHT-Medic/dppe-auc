import copy
import pickle
import struct
import time
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from sklearn import metrics

from FHM_approx import dppa_auc_protocol, dppa_auc_proxy, create_synthetic_data_dppa
from paillier import *


def plot_input_data(df, df_real, df_fake, station, proxy=None):
    if proxy:
        plt.clf()
        plt.style.use('ggplot')
        plt.title('Data distribution at proxy')
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
        plt.title('Data distribution of station {}'.format(station + 1))
        plt.hist([df_p['Real'], df_p['Flag']], edgecolor='black', bins=40, color=['green', 'red'], stacked=True,
                 rwidth=0.6,
                 alpha=0.5, label=['Real', 'Flag'])
        plt.legend(loc='upper left')
        plt.yscale('log')
        plt.xlabel('Prediction Values')
        plt.ylabel('Subjects')

        plt.tight_layout()
        plt.show()

def return_df(df):
    return pd.DataFrame(df, columns=['Pre', 'Label', 'Flag'])


def calculate_regular_auc(stations, performance, data, APPROX):
    """
    Calculate AUC with sklearn as ground truth GT
    """
    concat_df = pd.concat(data)

    samples = len(concat_df)
    performance['samples'].append(samples)

    sort_df = concat_df.sort_values(by='Pre', ascending=False)
    if APPROX:
        performance['flags'].append(0)
        filtered_df = sort_df
        print('FHAUC uses data from {} stations. Total of {} subjects (including 0 flag subjects) '.format(stations,
                                                                                                    len(filtered_df)))
    else:
        flags = len(concat_df[concat_df['Flag'] == 0])
        performance['flags'].append(samples)
        print('DPPE-AUC uses data from {} stations. Total of {} subjects (including {} flag subjects) '.format(stations,
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

    # Aggregator RSA keys
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


def load_rsa_pk(path):
    """
    Return public rsa key of given file path
    """

    with open(path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read(), backend=default_backend())

    return public_key


def rsa_encrypt(text, key):
    """
    Encrypt symmetric key_station with public rsa key of aggregator
    return: encrypted_symmetric_key
    """
    public_key = serialization.load_pem_public_key(
        key, backend=default_backend()
    )
    encrypted_text = public_key.encrypt(
        text,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA512()),
            algorithm=hashes.SHA512(),
            label=None,
        ),
    )
    return encrypted_text


def encrypt_symmetric_key(symmetric_key, results):
    """
    Encrypt symmetric key_station with public rsa key of aggregator
    return: encrypted_symmetric_key
    """

    rsa_agg_pk = load_pem_public_key(results['aggregator_rsa_pk'], backend=default_backend())
    encrypted_symmetric_key = rsa_agg_pk.encrypt(symmetric_key, padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA512()),
        algorithm=hashes.SHA512(),
        label=None
    ))

    return encrypted_symmetric_key


def symm_key_rsa_decrypt(ciphertext, path):
    """
    Decrypt of given station rsa encrypted k_station
    """
    with open(path, "rb") as key_file:
        station_rsa_sk = serialization.load_pem_private_key(
                key_file.read(),
                password=None,
                backend=default_backend()
            )
    env_symm_key = station_rsa_sk.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA512()),
                algorithm=hashes.SHA512(),
                label=None,
            )
        )
    return env_symm_key


def dppe_auc_protocol(local_df, prev_results, directory=str, station=int, max_value=int, save_data=None, save_keys=None,
                      keys=None, rsa_sk_path=None):
    """
    Perform PP-AUC protocol at specific station given dataframe
    """
    agg_pk = prev_results['aggregator_paillier_pk']
    symmetric_key = Fernet.generate_key()  # represents k1 k_n
    if station == 1:
        r1 = randint(20000, max_value)
        #print("rand r_1 {}".format(r1))
        enc_table = encrypt_table(local_df, agg_pk, r1, symmetric_key)
        if save_data:  # Save for transparency the table - not required
            enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, prev_results)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1)  # homomorphic encryption used
            prev_results['encrypted_r1'][i] = enc_r1

    else:
        enc_r1 = prev_results['encrypted_r1'][station - 1]
        #if save_keys:
        #    sk_s_i = pickle.load(open(directory + '/keys/s' + str(station) + '_paillier_sk.p', 'rb'))
        #else:
        #    sk_s_i = keys['s_p_sks'][station - 1]

        enc_symm_key = prev_results['enc_symm_key'][0]
        env_symm_key = symm_key_rsa_decrypt(enc_symm_key, rsa_sk_path)

        sk_s_i = prev_results['enc_s_p_sks'][station - 1]
        dec_sk = {'n': Fernet(env_symm_key).decrypt(sk_s_i['n']).decode(),
                  'x': Fernet(env_symm_key).decrypt(sk_s_i['x']).decode()
                  }
        station_sk = PrivateKey(int(dec_sk['n']), int(dec_sk['x']))
        dec_r1 = decrypt(station_sk, enc_r1)

        enc_table = encrypt_table(local_df, agg_pk, dec_r1, symmetric_key)
        if save_data:
            enc_table.to_pickle(directory + '/encrypted/data_s' + str(station) + '.pkl')

        enc_symmetric_key = encrypt_symmetric_key(symmetric_key, prev_results)
        prev_results['encrypted_ks'].append(enc_symmetric_key)

    prev_results['pp_auc_tables'][station - 1] = enc_table

    return prev_results


def z_values(n):
    """
    Generate random values of list length n which sum is zero
    """
    l = random.sample(range(-int(n / 2), int(n / 2)), k=n - 1)
    return l + [-sum(l)]


def dppe_auc_proxy(results, max_value, sk_path):
    """
    Simulation of aggregator service - globally computes privacy preserving AUC table as proxy station
    """
    agg_pk = results['aggregator_paillier_pk']
    enc_sk_1 = results['enc_agg_sk_1']
    enc_symm_key = results['enc_symm_key'][-2]
    env_symm_key = symm_key_rsa_decrypt(enc_symm_key,sk_path)
    dec_sk_1 = {'n': Fernet(env_symm_key).decrypt(enc_sk_1['n']).decode(),
                'x1': Fernet(env_symm_key).decrypt(enc_sk_1['x1']).decode()
                }
    # sk_keys['agg_sk_1'] = enc_sk_1  # encrypt with aggreagtor RSA

    agg_sk = PrivateKeyOne(int(dec_sk_1['n']), int(dec_sk_1['x1']))
    df_list = []
    for i in range(len(results['encrypted_ks'])):
        enc_k_i = results['encrypted_ks'][i]
        dec_k_i = symm_key_rsa_decrypt(enc_k_i, sk_path)

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
    plot_input_data(df_new_index, None, None, None, proxy=True)
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
    #print("rand a {}".format(a))
    #print("rand b {}".format(b))
    # Denominator
    # TP_A is summation of labels (TP)
    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)

    r_1A = randint(1, max_value)
    r_2A = randint(1, max_value)
    #print("rand r_1A {}".format(r_1A))
    #print("rand r_2A {}".format(r_2A))
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


def pp_auc_station_final(train_results, sk_path, sk_pw, APPROX):
    """
    Simulation of station delegated AUC parts to compute global DPPE-AUC locally
    """
    #if save_keys:
    #    agg_sk_2 = pickle.load(open(directory + '/keys/agg_sk_2.p', 'rb'))
    #else:
    #    agg_sk_2 = keys['agg_sk_2']

    enc_symm_key = train_results['enc_symm_key'][-1]

    with open(sk_path, "rb") as key_file:
        agg_rsa_sk = serialization.load_pem_private_key(
            key_file.read(),
            password=sk_pw.encode(),
            backend=default_backend()
        )
    env_symm_key = agg_rsa_sk.decrypt(
        enc_symm_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA512()),
            algorithm=hashes.SHA512(),
            label=None,
        )
    )

    enc_sk_2 = train_results['enc_agg_sk_2']

    dec_sk_2 = {'n': Fernet(env_symm_key).decrypt(enc_sk_2['n']).decode(),
                'x2': Fernet(env_symm_key).decrypt(enc_sk_2['x2']).decode()
                }
    agg_sk_2 = PrivateKeyTwo(int(dec_sk_2['n']), int(dec_sk_2['x2']))
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
        print('FHAUC: {}'.format(auc))
    else:
        print('DPPE-AUC: {}'.format(auc))
    return auc


