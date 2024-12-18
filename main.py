import os
import time
import copy
import shutil
import pickle
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
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
        self.results = results

    def load_results(self):
        try:
            with open(f'./data/pht_results/{self.results}', 'rb') as results_file:
                return pickle.load(results_file)
        except FileNotFoundError:
            return self._initialize_results()

    @staticmethod
    def _initialize_results():
        return {'approx': Train._empty_result_dict(), 'exact': Train._empty_result_dict()}

    @staticmethod
    def _empty_result_dict():
        return {
            'enc_rx': {}, 'pp_auc_tables': {}, 'encrypted_ks': [],
            'encrypted_r1': {}, 'encrypted_r2': {}, 'aggregator_rsa_pk': {},
            'aggregator_paillier_pk': {}, 'stations_paillier_pk': {},
            'stations_rsa_pk': {}, 'proxy_encrypted_r_N': {}, 'D1': [], 'D2': [], 'D3': [], 'N1': [], 'N2': [], 'N3': []
        }

    def save_results(self, results):
        try:
            with open(f'./data/pht_results/{self.results}', 'wb') as results_file:
                pickle.dump(results, results_file)
        except FileNotFoundError as e:
            print(f"Error saving results: {e}")
            raise

def return_df(df):
    return pd.DataFrame(df, columns=['Pre', 'Label', 'Flag'])


def create_synthetic_data(num_stations, samples, fake_patients=None):
    """
    Create synthetic data of given number of samples and number of stations.
    """
    dfs = []
    samples_each = samples // num_stations

    for station_i in range(num_stations):
        fakes = random.uniform(fake_patients[0], fake_patients[1]) if fake_patients else 0
        np.random.seed(42)

        real_data = {
            "Pre": np.random.random(size=samples_each),
            "Label": np.random.choice([0, 1], size=samples_each, p=[0.2, 0.8]),
            "Flag": np.random.choice([1], size=samples_each)
        }
        df_real = return_df(real_data)

        # Fake Data
        tmp_val = list(df_real['Pre'].sort_values(ascending=False))
        unique_values = [tmp_val[y] for y in sorted(np.unique(tmp_val, return_index=True)[1])]
        counts = list(df_real['Pre'].value_counts(ascending=False))
        max_count = counts[0] + int(counts[0] * 0.1)
        repetitions = [max_count - counts[i] for i in range(len(counts))]
        if sum(repetitions) == 0:
            repetitions = [x + 1 for x in repetitions]

        synthetic_series = pd.Series(np.repeat(unique_values[i], repetitions[i]) for i in range(len(repetitions)))
        list_fakes = synthetic_series.explode(ignore_index=True)
        fake_sample_size = len(list_fakes)

        fake_data = {
            "Pre": list_fakes,
            "Label": np.zeros(fake_sample_size, dtype=int),  # Fake data gets Label 0
            "Flag": np.zeros(fake_sample_size, dtype=int)  # Flag 0 for fake data
        }
        df_fake = return_df(fake_data)

        # Merge DataFrames (Skip if empty)
        df_list = [df_real, df_fake]
        df_list = [df for df in df_list if
                   not df.empty and not df.isna().all().any()]  # Filter out empty or NA DataFrames
        if df_list:
            merged = pd.concat(df_list, axis=0).sample(frac=1).reset_index(drop=True)
            merged.loc[merged["Flag"] == 0, "Label"] = 0  # Ensure consistency: Label = 0 where Flag = 0
            #plot_input_data(merged, df_real, df_fake, station_i, run, proxy=False)
            dfs.append(merged)

    return dfs

def generate_skewed_predictions(size):
    size_high = int(size * 0.85)  # 85% of values close to 1
    size_low = int(size * 0.1)  # 10% of values close to 0
    size_mid = size - size_high - size_low  # ~5% for mid-range values

    # Bias generation with range enforcement
    bias_high =  np.clip(np.random.beta(a=0.8, b=1, size=size_high), 0.9, 1)
    bias_low = np.clip(np.random.beta(a=1, b=0.1, size=size_low), 0, 0.1)
    bias_mid = np.clip(np.random.beta(a=2, b=2, size=size_mid), 0.1, 0.9)  # Sparse mid-range

    return np.concatenate([bias_low, bias_mid, bias_high])


def create_synthetic_data_same_size(num_stations, samples, fake_ratio=(0.1, 0.5)):
    """
    Create synthetic data of given number of samples and number of stations with same size.
    """
    np.random.seed(42)  # Ensure reproducibility

    samples_each = samples // num_stations
    leftover_samples = samples % num_stations

    # Cap the number of synthetic samples to 10% - 50% of real data
    min_fakes = int(samples * fake_ratio[0])
    max_fakes = int(samples * fake_ratio[1])
    total_fakes = np.random.randint(min_fakes, max_fakes + 1)

    dfs = []
    for station_i in range(num_stations):
        current_samples = samples_each + (leftover_samples if station_i == num_stations - 1 else 0)
        fakes_this_station = total_fakes // num_stations

        # Step 1: Generate real data
        real_data = {
            "Pre": generate_skewed_predictions(current_samples),
            "Label": np.random.choice([0, 1], size=current_samples, p=[0.2, 0.8]),
            "Flag": [1] * current_samples
        }
        df_real = pd.DataFrame(real_data)

        # Step 2: Generate minimal synthetic data
        unique_vals = df_real['Pre'].unique()
        synthetic_pre = []
        subjects_per_value = max(1, fakes_this_station // len(unique_vals))

        for val in unique_vals:
            synthetic_pre.extend([val] * subjects_per_value)

        synthetic_pre = synthetic_pre[:fakes_this_station]
        fake_data = {
            "Pre": synthetic_pre,
            "Label": [0] * len(synthetic_pre),
            "Flag": [0] * len(synthetic_pre)
        }
        df_fake = pd.DataFrame(fake_data)

        # Step 3: Merge real and synthetic data
        merged = pd.concat([df_real, df_fake], ignore_index=True)
        merged = merged.sample(frac=1).reset_index(drop=True)

        # Ensure consistency: Label = 0 where Flag = 0
        merged.loc[merged["Flag"] == 0, "Label"] = 0
        #plot_input_data(merged, df_real, df_fake, station_i, run, proxy=False)

        dfs.append(merged)

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
    else:
        flags = len(concat_df[concat_df['Flag'] == 0])
        performance['flags'].append(samples)
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
    #rsa_agg_pk = agg_pk
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


def dppe_auc_protocol(local_df, prev_results, directory, station, max_value, save_keys, keys):
    """
    Perform DPPE-AUC protocol at specific station given dataframe.
    """
    agg_pk = prev_results['aggregator_paillier_pk']
    symmetric_key = Fernet.generate_key()  # represents k1 k_n
    if station == 1:
        random.seed(9001)
        r1 = random.randint(20000, max_value)
        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1)  # homomorphic encryption
            prev_results['encrypted_r1'][i] = enc_r1
    else:
        enc_r1 = prev_results['encrypted_r1'][station - 1]
        sk_s_i = pickle.load(open(directory + f'/keys/s{station}_paillier_sk.p', 'rb')) if save_keys else keys['s_p_sks'][station - 1]
        r1 = decrypt(sk_s_i, enc_r1)
    enc_table = encrypt_table(local_df, agg_pk, r1, symmetric_key)
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
    #plot_input_data(df_new_index, None, None, None, run, proxy=True)
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

    a = random.randint(1, max_value)
    b = random.randint(1, max_value)

    # Denominator
    # TP_A is summation of labels (TP)
    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)

    r_1A = random.randint(1, max_value)
    r_2A = random.randint(1, max_value)

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
    threshold_indexes = []
    pred = df_new_index["Dec_pre"].to_list()
    for i in range(M - 1):
        if pred[i] != pred[i + 1]:
            threshold_indexes.append(i)

    threshold_indexes = list(map(lambda x: x + 1, threshold_indexes))  # add one
    threshold_indexes.insert(0, 0)
    len_t = len(threshold_indexes)

    # Multiply with a and b respectively
    Z_values = z_values(len_t)

    # sum over all n_3 and only store n_3
    N_3_sum = encrypt(agg_pk, 0)
    for i in range(1, len_t + 1):
        pre_ind = threshold_indexes[i - 1]
        if i == len_t:
            cur_ind = -1
        else:
            cur_ind = threshold_indexes[i]
        # Multiply with a and b respectively
        sTP_a = mul_const(agg_pk, add(agg_pk, tp_values[cur_ind], tp_values[pre_ind]), a)
        dFP_b = mul_const(agg_pk, add(agg_pk, fp_values[cur_ind], mul_const(agg_pk, fp_values[pre_ind], -1)), b)
        r1_i = random.randint(1, max_value)
        r2_i = random.randint(1, max_value)

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


def pp_auc_station_final(directory, train_results, save_keys, keys, approx):
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
    if approx:
        print('FHAUC: {}'.format(auc))
    else:
        print('DPPE-AUC: {}'.format(auc))
    return auc


def plot_experiment_1(results):
    """
     Plots the runtime for all methods (e.g., 'approx', 'exact') provided in the results dictionary.

     :param results: The performance results dictionary containing methods and timing info.
     """
    data_frames = []

    for method, res in results.items():
        # Calculate total time by summing relevant columns
        total_time = [sum(x) for x in zip(res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2'])]

        # Create a DataFrame for each method's results
        df = pd.DataFrame({
            'Station_1': res['time']['stations_1'],
            'Proxy': res['time']['proxy'],
            'Station_2': res['time']['stations_2'],
            'Total': total_time,
            'Samples': res['samples'],
            'Stations': res['stations'],
            'Method': method
        })

        # Multiply 'Station_2' by the number of stations for more accurate representation
        df['Station_2'] *= df['Stations'] # TODO clarify if station_part_2 is already multiplied

        # Append to list for later concatenation
        data_frames.append(df)

    # Combine all data into one DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Plot setup
    plt.figure(figsize=(12, 8))
    sns.set_palette(["#1f77b4", "#ff7f0e"])  # Set distinct colors for different methods

    # Create the boxplot with thicker lines for better visibility
    ax = sns.boxplot(x='Stations', y='Total', hue='Method', data=combined_df, linewidth=2.5)
    plt.xlabel('Number of Input-Parties')
    plt.ylabel('Time (sec)')
    plt.title("Runtime Analysis")

    # Adjust legend position
    plt.gca().legend(title="Method", loc="upper left")
    plt.tight_layout()

    # Display the plot
    plt.show()
    # Optional: Save the plot
        #plt.savefig(f'plots/exp1_{method}.png')
        #plt.close()

def plot_experiment_2(performance):
    df_list = []
    for method in performance:
        res = performance[method]
        total_time = [sum(x) for x in zip(res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2'])]
        df = pd.DataFrame({
            'Method': method,
            'Station_1': res['time']['stations_1'],
            'Proxy': res['time']['proxy'],
            'Station_2': res['time']['stations_2'],
            'Total': total_time,
            'Samples': res['samples'],
            'Stations': res['stations']
        })

        df['Station_2'] = df['Station_2'] * df['Stations']
        df_list.append(df)

    # Combine dataframes from all methods
    df_all = pd.concat(df_list, ignore_index=True)

    c = plt.cm.Set2.colors
    markers = ['o', 's', '^', 'D']
    color_iter = iter(c)
    marker_iter = iter(markers)

    for method in df_all['Method'].unique():
        color = next(color_iter)
        marker = next(marker_iter)
        df_method = df_all[df_all['Method'] == method]
        for category in df_method['Stations'].unique():
            df_category = df_method[df_method['Stations'] == category]
            plt.plot('Samples', 'Total', data=df_category, color=color, marker=marker,
                     label=f"{method} - {category} stations")

    plt.xlabel('Number of subjects')
    plt.ylabel('Time (sec)')
    plt.title('Total Runtime Evaluation')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def initialize_performance_tracking():
    return {
        'time': {
            'stations_1': [], 'proxy': [], 'stations_2': [], 'total_step_1': []
        },
        'total_times': [], 'samples': [], 'flags': [], 'stations': [],
        'pp-auc': [], 'gt-auc': [], 'diff': []
    }


if __name__ == "__main__":
    """
    Run with either complete experiment 1 or 2 uncommented
    """
    DIRECTORY = './data'
    MAX = 100000
    no_of_decision_points = 100
    FAKES = [0.1, 0.6]  # percentage range for random values

    SIMULATE_PUSH_PULL = False
    SAVE_DATA = False
    SAVE_KEYS = False

    print("Comparing both approaches in same run")

    # Initialize performance tracking
    per = {
        'FHAUC': initialize_performance_tracking(),
        'DPPE-AUC': initialize_performance_tracking()
    }
    experiment_config = {
        1: {
            "description": "Experiment 1 - Increasing number of stations with the same sample size",
            "station_list": [3, 4, 5, 6, 7, 8, 9],
            "subject_list": [1500],
            "loops": 10,
            "experiment_id": 1
        },
        2: {
            "description": "Experiment 2 - Varying number of subjects with a fixed number of stations",
            "station_list": [3],
            "subject_list": [30, 90, 180, 360, 720, 1440, 2880, 5760, 11520],
            "loops": 10,
            "experiment_id": 2
        },
        3: {
            "description": "Test Experiment 2 - Fast run",
            "station_list": [2],
            "subject_list": [10,30,90],
            "loops": 2,
            "experiment_id": 2
        },
        4:  {
            "description": "Test Experiment 1 - Fast run",
            "station_list": [3],
            "subject_list": [15000],
            "loops": 1,
            "experiment_id": 1
        }
    }

    selected_experiment = 1

    if selected_experiment not in experiment_config:
        print("Invalid selection. Exiting.")
        exit(1)

    # Load the selected experiment configuration
    experiment = experiment_config[selected_experiment]
    print(f"Running {experiment['description']}")

    decision_points = np.linspace(0, 1, num=no_of_decision_points)[::-1]
    differences_approx, differences_exact = [], []
    data_approx, data_exact = [], []

    for subjects in experiment['subject_list']:
        train = Train(results='results.pkl')

        for stations in experiment['station_list']:
            for run in range(experiment['loops']):  # repeat n times, to make boxplot
                print(f"\nNew run {run + 1}/{experiment['loops']}: {stations} stations, {subjects} subjects")

                # Prepare directories
                directories = [DIRECTORY]
                if SAVE_DATA:
                    directories += [DIRECTORY + '/synthetic', DIRECTORY + '/encrypted']
                elif SAVE_KEYS:
                    directories.append(DIRECTORY + '/keys')
                elif SIMULATE_PUSH_PULL:
                    directories.append(DIRECTORY + '/pht_results')

                for dir in directories:
                    os.makedirs(dir, exist_ok=True)

                # Generate synthetic data based on experiment type and save setting
                exact_data_fn = create_synthetic_data_same_size if experiment['experiment_id'] == 1\
                    else create_synthetic_data
                exact_data = exact_data_fn(stations, subjects, FAKES)
                approx_data = create_synthetic_data_dppa(stations, exact_data, SAVE_DATA)

                data_approx.append(approx_data.copy())
                data_exact.append(exact_data.copy())

                results = train.load_results()
                results['exact'], keys_exact = generate_keys(stations, DIRECTORY, results['exact'], save=SAVE_KEYS)
                results['approx'], keys_approx = generate_keys(stations, DIRECTORY, results['approx'], save=SAVE_KEYS)

                if SIMULATE_PUSH_PULL:
                    train.save_results(results)

                per['DPPE-AUC']['stations'].append(stations)
                per['FHAUC']['stations'].append(stations)

                # Compute AUC without encryption for proof of concept
                REGULAR_PATH = DIRECTORY + '/synthetic'
                times_exact, times_approx = [], []

                for i in range(stations):
                    stat_df = pickle.load(open(f"{DIRECTORY}/synthetic/data_s{i + 1}.pkl", 'rb')) if SAVE_DATA else None
                    exact_stat_df = exact_data[i] if not SAVE_DATA else stat_df
                    approx_stat_df = approx_data[i] if not SAVE_DATA else stat_df

                    if SIMULATE_PUSH_PULL:
                        results = train.load_results()  # Simulate pull of image

                    # DPPA and DPPE protocols
                    t1 = time.perf_counter()
                    results_approx = dppa_auc_protocol(approx_stat_df, decision_points, results['approx'], DIRECTORY,
                                                       station=i + 1, max_value=MAX, save_data=SAVE_DATA,
                                                       save_keys=SAVE_KEYS, keys=keys_approx)
                    t2 = time.perf_counter()
                    times_approx.append(t2 - t1)

                    t_1 = time.perf_counter()
                    results_exact = dppe_auc_protocol(exact_stat_df, results['exact'], DIRECTORY, station=i + 1,
                                                      max_value=MAX, save_keys=SAVE_KEYS, keys=keys_exact)
                    t_2 = time.perf_counter()
                    times_exact.append(t_2 - t_1)

                    print(f'DPPE-AUC Station {i + 1} step 1 time: {times_exact[-1]:.4f} seconds')
                    print(f'FHAUC Station {i + 1} step 1 time: {times_approx[-1]:.4f} seconds')

                    if i == stations - 1:  # Remove at the last station all encrypted noise values
                        results["approx"].pop('encrypted_r1')
                        results["exact"].pop('encrypted_r1')

                    if SIMULATE_PUSH_PULL:
                        train.save_results(results)

                # Output execution times
                print(f'Exact run {run + 1} total execution time at stations - Step 1: {sum(times_exact):.4f} seconds')
                print(
                    f'Approx run {run + 1} total execution time at stations - Step 1: {sum(times_approx):.4f} seconds')

                per['FHAUC']['time']['stations_1'].append(sum(times_approx) / len(times_approx))
                per['DPPE-AUC']['time']['stations_1'].append(sum(times_exact) / len(times_exact))

                per['FHAUC']['time']['total_step_1'].append(sum(times_approx))
                per['DPPE-AUC']['time']['total_step_1'].append(sum(times_exact))

                # Compute ground truth AUC
                auc_gt_approx, per['FHAUC'] = calculate_regular_auc(stations, per['FHAUC'], REGULAR_PATH, save=False,
                                                                     data=approx_data, APPROX=True)
                auc_gt_exact, per['DPPE-AUC'] = calculate_regular_auc(stations, per['DPPE-AUC'], REGULAR_PATH, save=False,
                                                                   data=exact_data, APPROX=False)

                print(f'Approx GT-AUC: {auc_gt_approx}')
                print(f'Exact GT-AUC: {auc_gt_exact}')

                # Proxy execution and final AUC
                t3 = time.perf_counter()
                approx_results = dppa_auc_proxy(DIRECTORY, results["approx"], max_value=MAX, save_keys=SAVE_KEYS,
                                                keys=keys_approx, no_dps=no_of_decision_points)
                t4 = time.perf_counter()
                per['FHAUC']['time']['proxy'].append(t4 - t3)

                t3 = time.perf_counter()
                exact_results = dppe_auc_proxy(DIRECTORY, results['exact'], max_value=MAX, save_keys=SAVE_KEYS,
                                               run=run, keys=keys_exact)
                t4 = time.perf_counter()
                per['DPPE-AUC']['time']['proxy'].append(t4 - t3)

                print(f'Exact execution time by proxy: {per['DPPE-AUC']["time"]["proxy"][-1]:.4f} seconds')
                print(f'Approx execution time by proxy: {per['FHAUC']["time"]["proxy"][-1]:.4f} seconds')

                if SIMULATE_PUSH_PULL:
                    train.save_results(results)
                    results = train.load_results()

                # Final AUC calculation
                t1 = time.perf_counter()
                auc_pp_exact = pp_auc_station_final(DIRECTORY, results['exact'], SAVE_KEYS, keys_exact, approx=False) # todo gpt_
                t2 = time.perf_counter()
                per['DPPE-AUC']['time']['stations_2'].append((t2 - t1) * stations)

                t1 = time.perf_counter()
                auc_pp_approx = pp_auc_station_final(DIRECTORY, results['approx'], SAVE_KEYS, keys_approx, approx=True)
                t2 = time.perf_counter()
                per['FHAUC']['time']['stations_2'].append((t2 - t1) * stations)

                # Record total times and differences
                total_time_exact = per['DPPE-AUC']["time"]["proxy"][-1] + per['DPPE-AUC']["time"]["stations_2"][-1] + \
                                   per['DPPE-AUC']["time"]["stations_1"][-1]
                per['DPPE-AUC']['total_times'].append(total_time_exact)

                total_time_approx = per['FHAUC']["time"]["proxy"][-1] + per['FHAUC']["time"]["stations_2"][-1] + \
                                    per['FHAUC']["time"]["stations_1"][-1]
                per['FHAUC']['total_times'].append(total_time_approx)

                per['FHAUC']['pp-auc'].append(auc_pp_approx)
                per['DPPE-AUC']['pp-auc'].append(auc_pp_exact)

                per['FHAUC']['gt-auc'].append(auc_gt_approx)
                per['DPPE-AUC']['gt-auc'].append(auc_gt_exact)

                diff_exact = auc_gt_exact - auc_pp_exact
                diff_approx = auc_gt_approx - auc_pp_approx

                per['DPPE-AUC']['diff'].append(diff_exact)
                per['FHAUC']['diff'].append(diff_approx)

                print(f'Difference DPPE-AUC (exact) to GT: {diff_exact}')
                print(f'Difference FHAUC (approx) to GT: {diff_approx}')
                print(f'')
                print(f'Exact avg difference over {len(per['DPPE-AUC']["diff"])} runs: {sum(per['DPPE-AUC']["diff"]) / len(per['DPPE-AUC']["diff"])}')
                print(f'Approx avg difference over {len(per['FHAUC']["diff"])} runs: {sum(per['FHAUC']["diff"]) / len(per['FHAUC']["diff"])}')
                print(f'')
                print(f'Exact avg exec time: {sum(per['DPPE-AUC']["total_times"]) / len(per['DPPE-AUC']["total_times"])} seconds')
                print(f'Approx avg exec time: {sum(per['FHAUC']["total_times"]) / len(per['FHAUC']["total_times"])} seconds')
    print(per)

    if experiment['experiment_id'] == 1:
        plot_experiment_1(per)
    elif experiment['experiment_id'] == 2:
        plot_experiment_2(per)
