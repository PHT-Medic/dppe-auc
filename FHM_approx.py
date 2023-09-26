import pickle
import numpy as np
import pandas as pd

from random import randint
from paillier.paillier import *
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric import padding


def create_synthetic_data_dppa(num_stations=int, df=None, save=None):  # edited
    """
    Create and save synthetic data of given number of samples and number of stations. Including flag patients
    """
    dfs = []

    for station_i in range(num_stations):
        real = df[station_i][df[station_i]['Flag'] == 1]
        real_data = {
            "Pre": real.Pre,
            "Label": real.Label,

        }
        df_real = pd.DataFrame(real_data, columns=['Pre', 'Label'])

        df_real.sort_values('Pre', ascending=False, inplace=True)

        # tmp_val = list(df_real['Pre'].sort_values(ascending=False))

        if save:
            df_real.to_pickle('./data/synthetic/data_s' + str(station_i + 1) + '.pkl')
        else:
            dfs.append(df_real)
    if not save:
        return dfs


def encrypt_table_dppa(station_df, agg_pk, r1, symmetric_key):  # edited
    """
    Encrypt dataframe of given station dataframe with paillier public key of aggregator and random values
    """

    station_df["tp"] = station_df["tp"].apply(lambda x: encrypt(agg_pk, x))
    station_df["fp"] = station_df["fp"].apply(lambda x: encrypt(agg_pk, x))

    return station_df


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


def dppa_auc_protocol(station_df, dps, prev_results, directory=str, station=int, max_value=int, save_data=None, save_keys=None, keys=None):
    """
    Perform PP-AUC protocol at specific station given dataframe
    """
    agg_pk = prev_results['aggregator_paillier_pk']
    symmetric_key = Fernet.generate_key()  # represents k1 k_n

    lbls = np.array(station_df.Label)  # Now converting labels to list
    pcs = np.array(station_df.Pre)  # Now converting pred cons to list as well

    total_1s = lbls.sum()
    total_0s = len(lbls) - total_1s

    ones = 0
    zeros = 0

    pred_con_index = 0
    last_one_visited = False
    size = int(total_1s + total_0s)

    my_data = pd.DataFrame(columns=['tp', 'fp'])

    for d in dps:
        for p in range(pred_con_index, size):
            if pcs[p] > d:
                if p == size - 1:
                    if not last_one_visited:
                        ones += lbls[p]
                        zeros += 1 - lbls[p]
                        last_one_visited = True

                    my_data.loc[len(my_data.index)] = [ones, zeros]
                    pred_con_index = p
                    break
                else:
                    ones += lbls[p]
                    zeros += 1 - lbls[p]
            else:
                my_data.loc[len(my_data.index)] = [ones, zeros]
                pred_con_index = p
                break

    if station == 1:
        r1 = randint(1, max_value)  # delete later

        enc_table = encrypt_table_dppa(my_data, agg_pk, r1, symmetric_key)
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

        enc_table = encrypt_table_dppa(my_data, agg_pk, dec_r1, symmetric_key)
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


def dppa_auc_proxy(directory, results, max_value, save_keys, keys, no_dps=int):  # edited
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
        table_i = results['pp_auc_tables'][i]
        df_list.append(table_i)

    tp_values = [encrypt(agg_pk, 0) for _ in range(no_dps)]
    fp_values = [encrypt(agg_pk, 0) for _ in range(no_dps)]

    for i in range(0, no_dps):
        for station in range(len(df_list)):
            tp_values[i] = add(agg_pk, tp_values[i], df_list[station]["tp"].iloc[i])
            fp_values[i] = add(agg_pk, fp_values[i], df_list[station]["fp"].iloc[i])

    a = randint(1, max_value)
    b = randint(1, max_value)

    # Denominator

    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)  # total number of positive labels
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)  # total number of negative labels

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

    # compute differences between each DP for TP and FP
    final_tp_values = [tp_values[0]]
    final_fp_values = [fp_values[0]]

    for i in range(1, no_dps):
        final_tp_values.append(add(agg_pk, tp_values[i], tp_values[i - 1]))
        final_fp_values.append(add(agg_pk, fp_values[i], mul_const(agg_pk, fp_values[i - 1], -1)))

    Z_values = z_values(no_dps)
    N_3_sum = encrypt(agg_pk, 0)

    for i in range(no_dps):
        sTP_a = mul_const(agg_pk, final_tp_values[i], a)
        dFP_b = mul_const(agg_pk, final_fp_values[i], b)

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

