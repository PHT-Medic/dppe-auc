import pandas as pd
import numpy as np
import pickle
import time
from random import randint
from paillier.paillier import generate_keypair, encrypt, decrypt, e_add, add, e_mul_const, proxy_decrypt
import os
import shutil
from typing import Union
from sklearn import metrics
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import random

import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


class Train:
    def __init__(self, results=None):
        """
        :param results:
        """
        self.results = results

    def load_results(self):
        """
        If a result file exists, loads the results. Otherwise will return empty results.
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
                    'encrypted_r1': {}, # index is for station for the following three lists
                    'encrypted_r2': {},
                    'encrypted_ks': [],
                    'aggregator_rsa_pk': {},
                    'aggregator_paillier_pk': {},
                    'stations_paillier_pk': {},
                    'stations_rsa_pk': {},
                    'test_r1': {},
                    'test_r2': {}
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
        except Exception as e:
            print(e)
            raise FileNotFoundError("Result file cannot be saved")


def create_protocol_data():
    data_1 = {"Pre": [99, 5, 12, 67, 4],
              "Label": [1, 1, 0, 0, 1],
              "Flag": [1, 1, 0, 1, 1]}

    df1 = pd.DataFrame(data_1, columns=['Pre', 'Label', 'Flag'])

    df1.to_pickle('./data/synthetic/protocol_data_s1.pkl')

    data_2 = {"Pre": [11, 27, 17, 98, 44],
              "Label": [0, 1, 0, 1, 0],
              "Flag": [0, 1, 1, 1, 0]}

    df2 = pd.DataFrame(data_2, columns=['Pre', 'Label', 'Flag'])

    df2.to_pickle('./data/synthetic/protocol_data_s2.pkl')

    data_3 = {"Pre": [77, 88, 41, 39, 66],
              "Label": [1, 0, 0, 0, 0],
              "Flag": [1, 0, 0, 1, 0]}
    df3 = pd.DataFrame(data_3, columns=['Pre', 'Label', 'Flag'])
    df3.to_pickle('./data/synthetic/protocol_data_s3.pkl')


def generate_keys(stations, results):
    # generate keys of stations
    for i in range(stations):
        # paillier keys
        sk, pk = generate_keypair(128)
        pickle.dump(sk, open('./data/keys/s' + str(i+1) + '_paillier_sk.p', 'wb'))
        pickle.dump(pk, open('./data/keys/s' + str(i+1) + '_paillier_pk.p', 'wb'))
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

        with open('./data/keys/s' + str(i+1) + '_rsa_sk.pem', 'wb') as f:
            f.write(private_pem)

        public_pem = rsa_public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        with open('./data/keys/s' + str(i+1) + '_rsa_pk.pem', 'wb') as f:
            f.write(public_pem)

        results['stations_rsa_pk'][i] = public_pem

    # generate keys of aggregator
    sk, pk = generate_keypair(128)
    pickle.dump(sk, open('./data/keys/agg_sk.p', 'wb'))
    pickle.dump(pk, open('./data/keys/agg_pk.p', 'wb'))
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

    with open('./data/keys/agg_rsa_private_key.pem', 'wb') as f:
        f.write(private_pem)

    public_pem = rsa_public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open('./data/keys/agg_rsa_public_key.pem', 'wb') as f:
        f.write(public_pem)

    results['aggregator_rsa_pk'] = public_pem
    print('Keys created for {} stations and aggregator station'.format(stations))
    print('Added rsa and paillier pks of stations and aggregator to results')

    return results


def encrypt_table(station_df, agg_pk, r1, r2, symm_key, station):
    print('Start encrypting table with {} subjects from station {}'.format(len(station_df), station))
    tic = time.perf_counter()
    # Just trivial implementation - improve with vectorizing and
    station_df["Pre"] *= r1
    station_df["Pre"] += r2
    station_df["Pre"] = station_df["Pre"].apply(lambda x: Fernet(symm_key).encrypt(int(x).to_bytes(2, 'big')))
    # Step 1
    station_df["Label"] = station_df["Label"].apply(lambda x: encrypt(agg_pk, x))
    station_df["Flag"] = station_df["Flag"].apply(lambda x: encrypt(agg_pk, x))
    toc = time.perf_counter()
    print(f'Encryption time {toc - tic:0.4f} seconds')
    return station_df


def load_rsa_sk(path):
    with open(path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )
    return private_key


def load_rsa_pk(path):
    with open(path, "rb") as key_file:
        public_key = serialization.load_pem_public_key(
            key_file.read(),
            backend=default_backend()
        )
    return public_key


def encrypt_symm_key(station, symm_key):
    print('Symmetric key of k_{} is: {}'.format(station, symm_key))
    rsa_agg_pk = load_rsa_pk('./data/keys/agg_rsa_public_key.pem')
    encrypted_symm_key = rsa_agg_pk.encrypt(symm_key,
                                            padding.OAEP(
                                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                algorithm=hashes.SHA256(),
                                                label=None
                                            ))

    return encrypted_symm_key


def decrypt_symm_key(station, ciphertext):
    print('Symmetric key of k_{} encrypted is: {}'.format(station, ciphertext))
    rsa_agg_sk = load_rsa_sk('./data/keys/agg_rsa_private_key.pem')
    decrypted_symm_key = rsa_agg_sk.decrypt(
        ciphertext,
        padding.OAEP(
         mgf=padding.MGF1(algorithm=hashes.SHA256()),
         algorithm=hashes.SHA256(),
         label=None
        ))
    print('Symmetric key of k_{} decrypted is: {}'.format(station, decrypted_symm_key))
    return decrypted_symm_key


def pp_auc_protocol(station_df, agg_paillier_pk, station=int):
    prev_results = train.load_results()  # loading results simulates pull of image

    if station == 1:
        # Step 2
        r1 = randint(1, 100) # random value between 1 to 100
        r2 = randint(1, 100)
        prev_results['test_r1'][0] = r1
        prev_results['test_r2'][0] = r2
        symm_key = Fernet.generate_key()  # represents k1
        # rand_noise = r1 + rx
        enc_table = encrypt_table(station_df, agg_paillier_pk, r1, r2, symm_key, station)

        # Save for transparency the table - not required
        enc_table.to_pickle('./data/encrypted/data_s' + str(station) + '.pkl')

        # Step 3 - 1
        encrypted_symm_key = encrypt_symm_key(station, symm_key)
        # Step 3 - 2 (partial Decrypt enc(k) with x1 of pk
        # used RSA encryption of symm key (Fernet) - Document and ask Mete
        #partial_priv_key = pickle.load(open('./data/keys/agg_sk.p', 'rb'))
        #partial_decrypted = proxy_decrypt(partial_priv_key, encrypted_symm_key)

        prev_results['encrypted_ks'].append(encrypted_symm_key)

        # encrypt r1 and r2 for stations rsa_pk
        #  prev_results['enc_rx'][station-1] = ra_enc
        # TODO figure partially decrypted Sp of ra out
        # Step 4
        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1) # Homomoprphic encryption used
            enc_r2 = encrypt(prev_results['stations_paillier_pk'][i], r2)
            # Step 5
            prev_results['encrypted_r1'][i] = enc_r1
            prev_results['encrypted_r2'][i] = enc_r2
            print('Store with pk of stations encrypted r1 value {} as {}'.format(r1, enc_r1))
            print('Store with pk of stations encrypted r2 value {} as {}'.format(r2, enc_r2))

    else:
        # Step 7
        enc_r1 = prev_results['encrypted_r1'][station-1]
        sk_s_i = pickle.load(open('./data/keys/s' + str(station) + '_paillier_sk.p', 'rb'))
        pk_s_i = pickle.load(open('./data/keys/s' + str(station) + '_paillier_pk.p', 'rb'))
        # Step 8
        dec_r1 = decrypt(sk_s_i, enc_r1)
        print('Decrypted at station {} encrypted r1 {} to {}'.format(station, enc_r1, dec_r1))
        enc_r2 = prev_results['encrypted_r2'][station-1]
        dec_r2 = decrypt(sk_s_i, enc_r2)
        print('Decrypted at station {} encrypted r1 {} to {}'.format(station, enc_r2, dec_r2))

        # Step 9 / 10
        symm_key = Fernet.generate_key()  # represents k_2 to k_n
        enc_table = encrypt_table(station_df, agg_paillier_pk, dec_r1, dec_r2, symm_key, station)

        # Step 11
        encrypted_symm_key = encrypt_symm_key(station, symm_key)
        # Step 12
        prev_results['encrypted_ks'].append(encrypted_symm_key)

    #rx_enc = encrypt(agg_pk, rx)
    #print('Store encrypted rx value {} in train results'.format(rx))
    #prev_results['enc_rx'][station-1] = rx_enc
    prev_results['pp_auc_tables'][station-1] = enc_table

    return prev_results


def create_fake_data(stations=int, samples=int, fake_patients=None):
    for i in range(stations):
        fake_data_val = randint(fake_patients[0], fake_patients[1])
        data = {"Pre": np.random.randint(low=5, high=100, size=samples+fake_data_val),
                "Label": np.random.choice([0,1], size=samples+fake_data_val, p=[0.1, 0.9]),
                "Flag": np.random.choice(np.concatenate([[1] * samples, [0] * fake_data_val]), samples+fake_data_val, replace=False)}

        df = pd.DataFrame(data, columns=['Pre', 'Label', 'Flag'])

        df.to_pickle('./data/synthetic/data_s' + str(i+1) + '.pkl')


def sum_over_enc_series(encrypted_series, agg_pk):
    if len(encrypted_series) == 1:
        return encrypted_series[0]
    else:
        res = encrypt(agg_pk, 0)
        for cipher in encrypted_series:
            res = add(agg_pk, res, cipher)

        return res


def generate_random(M):
    while True:
        pick = random.sample(range(-100, 100), M)
        if sum(pick) == 0:
            break
    return pick


def generate_random_fast(M):
    import numpy as np, numpy.random
    pick = np.random.dirichlet(np.ones(M-1), size=1)
    pick *= 100
    pick = np.insert(pick.round()[0], 1, -sum(pick.round()[0]))
    return pick


def proxy_station():
    # Step 21
    results = train.load_results()
    agg_pk = pickle.load(open('./data/keys/agg_pk.p', 'rb'))
    agg_sk = pickle.load(open('./data/keys/agg_sk.p', 'rb'))

    # decrypt symm key (k_stations)
    with open('./data/keys/agg_rsa_private_key.pem', 'rb') as key_file:
        rsa_agg_sk = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend()
        )

    df_list = []
    for i in range(len(results['encrypted_ks'])):
        enc_k_i = results['encrypted_ks'][i]
        # Step 22
        dec_k_i = decrypt_symm_key(i, enc_k_i)
        print('Decrypted k value {} of station {}'.format(dec_k_i, i+1))

        # Step 23 decrypt table values with Fernet and corresponding k_i symmetric key
        table_i = results['pp_auc_tables'][i]
        table_i["Dec_pre"] = table_i["Pre"].apply(lambda x: Fernet(dec_k_i).decrypt(x)) # returns bytes
        table_i["Dec_pre"] = table_i["Dec_pre"].apply(lambda x: int.from_bytes(x, "big"))
        df_list.append(table_i)
    #dec_symm = decrypt(priv, pub, enc_symm)
    concat_df = pd.concat(df_list)
    concat_df.pop('Pre')
    print('\n')
    print('Concatenated (and sorted by paillier encrypted Pre) predictions of all station:')
    # Step 24
    sort_df = concat_df.sort_values(by='Dec_pre', ascending=False)
    print(sort_df)
    df_new_index = sort_df.reset_index()

    # calculate TP / FN / TN and FP with paillier summation over rows
    # Step 25
    TP_values = []
    FP_values = []

    # Step 29
    r1_A = randint(1, 100)  # random value between 1 to 100
    r2_A = randint(1, 100)

    #  generate M random numbers which sum up to 0
    # Step 31
    M = len(df_new_index)
    z_values = generate_random(M)
    z_fast_values = generate_random_fast(M).astype(int)

    D_1_vals = []
    D_2_vals = []
    D_3_vals = []

    N_1_vals = []
    N_2_vals = []
    N_3_vals = []

    for i in range(M):
        TP_enc = sum_over_enc_series(df_new_index['Label'][:i+1], agg_pk)
        print(decrypt(agg_sk, TP_enc))

        TP_values.append(TP_enc)


        neg_TP = e_mul_const(agg_pk, TP_enc, -1)  # subtraction of enc_tp_val

        FP_enc = e_add(agg_pk, sum_over_enc_series(df_new_index['Flag'][:i + 1], agg_pk), neg_TP)
        FP_values.append(FP_enc)

        r1_i = randint(1, 100)
        r2_i = randint(1, 100)

        D_1 = e_add(agg_pk, TP_enc, encrypt(agg_pk, r1_A))
        D_1_vals.append(D_1)

        D_2 = e_add(agg_pk, FP_enc, encrypt(agg_pk, r2_A))
        D_2_vals.append(D_2)

        D_31 = e_mul_const(agg_pk, TP_enc, r2_A)
        D_32 = e_mul_const(agg_pk, FP_enc, r1_A)
        D_33 = r1_A * r2_A

        D_3 = e_add(agg_pk, e_add(agg_pk, D_31, D_32), encrypt(agg_pk, D_33))
        D_3_vals.append(D_3)

        # Step 30
        N_i_1 = e_add(agg_pk, TP_enc, encrypt(agg_pk, r1_i))
        N_1_vals.append(N_i_1)

        N_i_2 = e_add(agg_pk, FP_enc, encrypt(agg_pk, r2_i))
        N_2_vals.append(N_i_2)

        N_i_31 = e_mul_const(agg_pk, TP_enc, r2_i)
        N_i_32 = e_mul_const(agg_pk, FP_enc, r1_i)
        N_i_33 = e_mul_const(agg_pk, encrypt(agg_pk, r1_i), r2_i)
        N_i_3 = e_add(agg_pk, e_add(agg_pk, N_i_31, N_i_32), N_i_33)
        z_i = z_fast_values[i]

        if z_i < 0:
            enc_z_i = encrypt(agg_pk, abs(int(z_i)))
            z_i_neg = e_mul_const(agg_pk, enc_z_i, agg_pk.n -1)
            N_i_3_noise = e_add(agg_pk, N_i_3, enc_z_i)
        else:
            enc_z_i = encrypt(agg_pk, int(z_i))
            N_i_3_noise = e_add(agg_pk, N_i_3, enc_z_i)

        N_3_vals.append(N_i_3_noise)

    # TODO test for validation of values! REMOVE in FINAL step
    TP_dec = [decrypt(agg_sk, x) for x in TP_values]
    FP_dec = [decrypt(agg_sk, x) for x in FP_values]
    dec_label = [decrypt(agg_sk, x) for x in df_new_index["Label"]]
    dec_flag = [decrypt(agg_sk, x) for x in df_new_index["Flag"]]
    full_series_enc_label = df_new_index['Label']
    full_series_enc_flag = df_new_index['Flag']
    TP_A = sum_over_enc_series(full_series_enc_flag, agg_pk)  # summation over all flags
    FP_A = sum_over_enc_series(full_series_enc_label, agg_pk)  # summation over all labels

    #  denominator -> D currently only with multiplication of decrypted value
    # D = e_mul_const(agg_pk, TP_A, decrypt(agg_sk, FP_A))
    D = e_mul_const(agg_pk, TP_A, FP_A)
    # nominator of sum for each TP * FP value
    # N = sum TP * FP

    # Randomized encoding
    r1_A = randint(1, 100) # random value between 1 to 100
    r2_A = randint(1, 100)
    # calc D_1, D_2, D_3 components of D

    # iterate over each M and regenerate rand r1_i and r2_i
    for i in range(len(df_new_index)):
        series_enc_label = df_new_index['Label'][:i+1]
        series_enc_flag = df_new_index['Flag'][:i+1]



    # add z values to each N_3
    # and catch if z < 0 -> negative multipy by -1 and add to subtract
    df_new_index["N_3"].apply(lambda x: add_noise(x,))

    test_r1 = results['test_r1'][0]
    test_r2 = results['test_r2'][0]
    recon_df = df_new_index
    recon_df['Dec_pre'] -= test_r2
    recon_df['Dec_pre'] /= test_r1
    recon_df['Label'] = dec_label
    recon_df['Flag'] = dec_flag
    print(recon_df)

    return recon_df


if __name__ == "__main__":
    stations = 3  # TODO adjust
    subjects = 10  # TODO adjust
    recreate = False # Set first True then false for running
    protocol = True

    train = Train(results='results.pkl')
    try:
        os.remove('./data/pht_results/results.pkl')
    except:
        pass

    # Initialization
    if recreate:
        try:
            shutil.rmtree('./data')
            print('Removed previous results')
        except Exception as e:
            print('No previous files and results to remove')

        directories = ['./data', './data/keys', './data/synthetic', './data/encrypted', './data/pht_results']
        for dir in directories:
            if not os.path.exists(dir):
                os.makedirs(dir)
        if protocol:
            create_protocol_data()
        else:
            create_fake_data(stations, subjects, [int(subjects*.30), int(subjects*.50)])

        print("Created data and exits")
        exit(0)

    results = train.load_results()
    results = generate_keys(stations, results)
    # Train Building process
    train.save_results(results)

    agg_paillier_pk = results['aggregator_paillier_pk']

    for i in range(stations):
        if protocol:
            stat_df = pickle.load(open('./data/synthetic/protocol_data_s' + str(i+1) + '.pkl', 'rb'))
        else:
            stat_df = pickle.load(open('./data/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))
        print('\n')
        results = pp_auc_protocol(stat_df, agg_paillier_pk, station=i+1)
        # remove at last station all encrypted noise values
        if i is stations - 1:
            print('Remove paillier_pks and all encrypted r1 and r2 values')
            results.pop('aggregator_paillier_pk')
            results.pop('encrypted_r1')
            results.pop('encrypted_r2')

        train.save_results(results)  # saving results simulates push of image
        print('Stored train results')
    print('\n ------ \n PROXY STATION')

    #  TODO compute pp-AUC with aggregator
    pp_auc = proxy_station() # TODO return list with TP, (TP+FN), TN and (TN+FP) for each threshold value

    # TODO itererate over stations and calcucale TP/(TP+FN) and TN/(TN+FP) for each threshold value

    # TODO calculate exact PP_AUC for comparison of regular_AUC
