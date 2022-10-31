import pandas as pd
import numpy as np
import pickle
import time
from random import randint
from paillier.paillier import *
import os
import shutil
from typing import Union
from sklearn import metrics
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import random
#from phe import paillier
import logging
import copy


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
                    'encrypted_ks': [],
                    'encrypted_r1': {}, # index is used by station i
                    'encrypted_r2': {},
                    'aggregator_rsa_pk': {},
                    'aggregator_paillier_pk': {},
                    'stations_paillier_pk': {},
                    'stations_rsa_pk': {},
                    'proxy_encrypted_r_N': {}, # index 0 = r1_iN; 1 = r2_iN
                    'test_r1' : {},
                    'test_r2': {},
                    'D1': [],
                    'D2': [],
                    'D3': [],
                    'N1': {},
                    'N2': {},
                    'N3': {},
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
              #"Label": [1, 0, 0, 0, 0], # AUC 0
              "Label": [1, 1, 1, 0, 0], # AUC 1
              "Flag": [1, 0, 0, 1, 0]}
    df3 = pd.DataFrame(data_3, columns=['Pre', 'Label', 'Flag'])
    df3.to_pickle('./data/synthetic/protocol_data_s3.pkl')


def calculate_regular_auc(stations, protocol):
    lst_df = []
    for i in range(stations):
        if protocol:
            df_i = pickle.load(open('./data/synthetic/protocol_data_s' + str(i+1) + '.pkl', 'rb'))
        else:
            df_i = pickle.load(open('./data/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))
        lst_df.append(df_i)

    # drop flag patients
    concat_df = pd.concat(lst_df)
    filtered_df = concat_df[concat_df["Flag"] == 0] # remove flag patients
    sort_df = filtered_df.sort_values(by='Pre', ascending=False)

    # iterate over sorted list
    auc = 0.0
    height = 0.0
    sort_df["Pre"] = sort_df["Pre"] / 100
    y = sort_df["Label"]
    pred = sort_df["Pre"]
    fpr, tpr, thesholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of ground truth')
    plt.legend(loc="lower right")
    plt.show()

    # return exact auc to be benchmarked with
    return auc


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
    sk_1 = copy.copy(sk)
    sk_2 = copy.copy(sk)
    # simulate private key separation
    del sk_1.x2
    del sk_2.x1
    pickle.dump(sk_1, open('./data/keys/agg_sk_1.p', 'wb'))
    pickle.dump(sk_2, open('./data/keys/agg_sk_2.p', 'wb'))
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
    logging.info('Keys created for {} stations and aggregator station'.format(stations))
    logging.info('Added rsa and paillier pks of stations and aggregator to results')

    return results


def encrypt_table(station_df, agg_pk, r1, r2, symm_key, station):
    logging.info('Start encrypting table with {} subjects from station {}'.format(len(station_df), station))
    tic = time.perf_counter()
    # Just trivial implementation - improve with vectorizing and
    station_df["Pre"] *= r1
    station_df["Pre"] += r2
    station_df["Pre"] = station_df["Pre"].apply(lambda x: Fernet(symm_key).encrypt(int(x).to_bytes(2, 'big')))
    # Step 1
    station_df["Label"] = station_df["Label"].apply(lambda x: encrypt(agg_pk, x))
    station_df["Flag"] = station_df["Flag"].apply(lambda x: encrypt(agg_pk, x))
    toc = time.perf_counter()
    logging.info(f'Encryption time {toc - tic:0.4f} seconds')
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
    logging.info('Symmetric key of k_{} is: {}'.format(station, symm_key))
    rsa_agg_pk = load_rsa_pk('./data/keys/agg_rsa_public_key.pem')
    encrypted_symm_key = rsa_agg_pk.encrypt(symm_key,
                                            padding.OAEP(
                                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                algorithm=hashes.SHA256(),
                                                label=None
                                            ))

    return encrypted_symm_key


def decrypt_symm_key(station, ciphertext):
    logging.info('Symmetric key of k_{} encrypted is: {}'.format(station, ciphertext))
    rsa_agg_sk = load_rsa_sk('./data/keys/agg_rsa_private_key.pem')
    decrypted_symm_key = rsa_agg_sk.decrypt(
        ciphertext,
        padding.OAEP(
         mgf=padding.MGF1(algorithm=hashes.SHA256()),
         algorithm=hashes.SHA256(),
         label=None
        ))
    logging.info('Symmetric key of k_{} decrypted is: {}'.format(station, decrypted_symm_key))
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

        # Step 4
        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1) # Homomoprphic encryption used
            enc_r2 = encrypt(prev_results['stations_paillier_pk'][i], r2)
            # Step 5
            prev_results['encrypted_r1'][i] = enc_r1
            prev_results['encrypted_r2'][i] = enc_r2
            logging.info('Store with pk of stations encrypted r1 value {} as {}'.format(r1, enc_r1))
            logging.info('Store with pk of stations encrypted r2 value {} as {}'.format(r2, enc_r2))

    else:
        # Step 7
        enc_r1 = prev_results['encrypted_r1'][station-1]
        sk_s_i = pickle.load(open('./data/keys/s' + str(station) + '_paillier_sk.p', 'rb'))
        pk_s_i = pickle.load(open('./data/keys/s' + str(station) + '_paillier_pk.p', 'rb'))
        # Step 8
        dec_r1 = decrypt(sk_s_i, enc_r1)
        logging.info('Decrypted at station {} encrypted r1 {} to {}'.format(station, enc_r1, dec_r1))
        enc_r2 = prev_results['encrypted_r2'][station-1]
        dec_r2 = decrypt(sk_s_i, enc_r2)
        logging.info('Decrypted at station {} encrypted r1 {} to {}'.format(station, enc_r2, dec_r2))

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
        #return add_const(agg_pk, encrypted_series[0], -1)
        return encrypted_series[0]
    else:
        res = encrypt(agg_pk, 0)
        for cipher in encrypted_series:
            res = add(agg_pk, res, cipher)

        #return add_const(agg_pk, res, -1)
        return res



def generate_random_fast(M):
    import numpy as np, numpy.random
    pick = np.random.dirichlet(np.ones(M-1), size=1)
    pick *= 100
    pick = np.insert(pick.round()[0], 1, -sum(pick.round()[0]))
    return pick


def compute_TP_FP_values(dataframe, agg_pk, length):
    TP_values = []
    FP_values = []
    agg_sk = pickle.load(open('./data/keys/agg_sk_1.p', 'rb'))

    for i in range(length):
        TP_enc = sum_over_enc_series(dataframe['Label'][:i+1], agg_pk)
        TP_values.append(TP_enc)
        #print("TP: {}".format(decrypt(agg_sk, TP_enc)))
    sum = 0
    for x in range(length):
        val = decrypt(agg_sk, TP_values[x])
        sum += val
        #print("List accessed TP: {}".format(val))
    logging.info('Expected TP sum: {}'.format(sum))

    for i in range(length):
        # subtraction of enc_tp_val
        pre_FP_enc = sum_over_enc_series(dataframe['Flag'][:i + 1], agg_pk)
        #print("Pre FP: {}".format(decrypt(agg_sk, pre_FP_enc)))
        # TODO Fix FP - is wrong in computation
        #print("Pre TP: {}".format(decrypt(agg_sk, TP_values[i])))
        FP_enc = e_add(agg_pk, pre_FP_enc, mul_const(agg_pk, TP_values[i], -1)) # subtraction of TP_A from sum_flags
        FP_values.append(FP_enc)
        #print("FP: {}".format(decrypt(agg_sk, FP_enc)))

    # Uncomment to see behaviour
    sum = 0
    for x in range(length):
         val = decrypt(agg_sk, FP_values[x])
         sum += val
         #print("List accessed FP: {}".format(val))
    logging.info('Expected FP sum: {}'.format(sum))
    return TP_values, FP_values


def calc_denominator(TP_A_mul, FP_A_mul, agg_pk):
    # Denominator
    r_1A = randint(1, 100)
    r_2A = randint(1, 100)

    D1 = add_const(agg_pk, TP_A_mul, r_1A)
    D2 = add_const(agg_pk, FP_A_mul, r_2A)

    D3_1 = mul_const(agg_pk, TP_A_mul, r_2A)
    D3_2 = mul_const(agg_pk, FP_A_mul, r_1A)
    D3 = add(agg_pk, D3_1, add_const(agg_pk, D3_2, r_1A * r_2A))

    return D1, D2, D3


def calc_nominator(TP_a, FP_a, agg_pk, length):
    # Nominator
    N_i1 = []
    N_i2 = []
    N_i3 = []

    # Step 31
    #  generate M random numbers which sum up to 0
    z_values = generate_random_fast(length).astype(int)
    for i in range(length):
        r1_i = randint(1, 100)
        r2_i = randint(1, 100)

        TP_i = TP_a[i]
        FP_i = FP_a[i]

        N_i1.append(add_const(agg_pk, TP_i, r1_i))
        N_i2.append(add_const(agg_pk, FP_i, r2_i))

        N_i3_1 = mul_const(agg_pk, TP_i, r2_i)
        N_i3_2 = mul_const(agg_pk, FP_i, r1_i)
        N_i3_a = add(agg_pk, N_i3_1, add_const(agg_pk, N_i3_2, r1_i * r2_i))
        # Add z values to N_i3_a
        z_i = z_values[i]
        if z_i < 0:
            # If noise is negative, multiply by -1 and add encrypted neg z value
            enc_z_i = encrypt(agg_pk, abs(int(z_i)))
            n_i_3_noise = e_add(agg_pk, N_i3_a, mul_const(agg_pk, enc_z_i, -1))
        else:
            n_i_3_noise = add_const(agg_pk, N_i3_a, z_i)

        N_i3.append(n_i_3_noise)

    return N_i1, N_i2, N_i3


def proxy_station():
    # Step 21
    results = train.load_results()
    agg_pk = pickle.load(open('./data/keys/agg_pk.p', 'rb'))
    agg_sk = pickle.load(open('./data/keys/agg_sk_1.p', 'rb'))

    # decrypt symmetric keys (k_stations)
    df_list = []
    for i in range(len(results['encrypted_ks'])):
        enc_k_i = results['encrypted_ks'][i]
        # Step 22
        dec_k_i = decrypt_symm_key(i, enc_k_i)
        logging.info('Decrypted k value {} of station {}'.format(dec_k_i, i + 1))

        # Step 23 decrypt table values with Fernet and corresponding k_i symmetric key
        table_i = results['pp_auc_tables'][i]
        table_i["Dec_pre"] = table_i["Pre"].apply(lambda x: Fernet(dec_k_i).decrypt(x))  # returns bytes
        table_i["Dec_pre"] = table_i["Dec_pre"].apply(lambda x: int.from_bytes(x, "big"))
        df_list.append(table_i)

    concat_df = pd.concat(df_list)
    concat_df.pop('Pre')
    logging.info('\n')
    #print('Concatenated (and sorted by paillier encrypted Pre) predictions of all station:')
    # Step 24
    sort_df = concat_df.sort_values(by='Dec_pre', ascending=False)
    #print(sort_df)
    df_new_index = sort_df.reset_index()

    # calculate TP / FN / TN and FP with paillier summation over rows
    # Step 25
    M = len(df_new_index) # TODO remove after denominator a
    TP_values, FP_values = compute_TP_FP_values(df_new_index, agg_pk, M)

    # TP_A is summation of labels (TP)
    # FP_A is sum Flags (FP) - TP_A
    TP_A = sum_over_enc_series(TP_values, agg_pk)
    logging.info('TP_A: {}'.format(decrypt(agg_sk, TP_A)))

    FP_A = e_add(agg_pk,  sum_over_enc_series(FP_values, agg_pk), mul_const(agg_pk,TP_A, -1))
    logging.info('FP_A: {}'.format(decrypt(agg_sk, FP_A)))

    # Step 26
    a = randint(1, 100)
    b = randint(1, 100)

    # Step 27
    tp_a_multiplied = mul_const(agg_pk, TP_A, a)
    FP_A_mul = mul_const(agg_pk, FP_A, b)

    # Step 28
    TP_is = []
    FP_is = []
    # Multiply with a and b respectively
    for i in range(M):
        TP_i = TP_values[i]
        TP_is.append(mul_const(agg_pk, TP_i, a))

        FP_i = FP_values[i]
        FP_is.append(mul_const(agg_pk, FP_i, b))


    D1, D2, D3 = calc_denominator(tp_a_multiplied, FP_A_mul, agg_pk)

    # Step 30
    N_i1, N_i2, N_i3 = calc_nominator(TP_is, FP_is, agg_pk, M)

    # partial decrypt and save to train
    results["D1"].append(proxy_decrypt(agg_sk, D1))
    results["D2"].append(proxy_decrypt(agg_sk, D2))
    results["D3"].append(proxy_decrypt(agg_sk, D3))
    results["N1"] = [proxy_decrypt(agg_sk, x) for x in N_i1]
    results["N2"] = [proxy_decrypt(agg_sk, x) for x in N_i2]
    results["N3"] = [proxy_decrypt(agg_sk, x) for x in N_i3]

    train.save_results(results)

def stations_auc(station):
    results = train.load_results()
    agg_sk = pickle.load(open('./data/keys/agg_sk_2.p', 'rb')) # todo split sk in sk_1 and sk_2

    logging.info('Station {}:\n'.format(station+1))
    # decrypt random components
    D1 = decrypt2(agg_sk, results['D1'][0])
    D2 = decrypt2(agg_sk, results['D2'][0])
    D3 = decrypt2(agg_sk, results['D3'][0])
    logging.info('D1 {}'.format(D1))
    logging.info('D2 {}'.format(D2))
    logging.info('D3 {}'.format(D3))

    iN1 = []
    iN2 = []
    iN3 = []

    for i in range(len(results['N1'])):
        iN1.append(decrypt2(agg_sk, results['N1'][i]))
        iN2.append(decrypt2(agg_sk, results['N2'][i]))
        iN3.append(decrypt2(agg_sk, results['N3'][i]))

    iNs = {"iN1": iN1,
             "iN2": iN2,
             "iN3": iN3}

    logging.info(iNs)
    N = 0
    for i in range(len(results['N1'])):
        N += ((iN1[i] * iN2[i]) + iN3[i])

    AUC = N / ((D1 * D2) + D3)
    logging.info('PP-AUC: {0:.3f}'.format(AUC))
    print('Station {}'.format(station+1))
    print('PP-AUC: {0:.3f}'.format(AUC))
    return AUC


if __name__ == "__main__":
    # Configuration
    logging.basicConfig(filename='pp-auc.log', level=logging.INFO)
    logging.info('Start PP-AUC execution')

    stations = 3  # TODO adjust
    subjects = 50  # TODO adjust

    #recreate, protocol = True, True
    recreate, protocol = False, True # Set first True then false for running

    train = Train(results='results.pkl')
    try:
        os.remove('./data/pht_results/results.pkl')
    except:
        pass

    # Initialization
    if recreate:
        try:
            shutil.rmtree('./data')
            logging.info('Removed previous results')
        except Exception as e:
            logging.info('No previous files and results to remove')

        directories = ['./data', './data/keys', './data/synthetic', './data/encrypted', './data/pht_results']
        for dir in directories:
            if not os.path.exists(dir):
                os.makedirs(dir)
        if protocol:
            create_protocol_data()
        else:
            create_fake_data(stations, subjects, [int(subjects*.30), int(subjects*.50)])

        logging.info('Created data and exits')
        exit(0)

    results = train.load_results()
    results = generate_keys(stations, results)
    # Train Building process
    train.save_results(results)

    agg_paillier_pk = results['aggregator_paillier_pk']

    # compute AUC without encryption for proof of principal of pp_auc
    auc_gt = calculate_regular_auc(stations, protocol)
    logging.info('AUC value of ground truth {}'.format(auc_gt))
    print('AUC value of ground truth {}'.format(auc_gt))

    for i in range(stations):
        if protocol:
            stat_df = pickle.load(open('./data/synthetic/protocol_data_s' + str(i+1) + '.pkl', 'rb'))
        else:
            stat_df = pickle.load(open('./data/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))
        logging.info('\n')
        results = pp_auc_protocol(stat_df, agg_paillier_pk, station=i+1)
        # remove at last station all encrypted noise values
        if i is stations - 1:
            logging.info('Remove paillier_pks and all encrypted r1 and r2 values')
            results.pop('aggregator_paillier_pk')
            results.pop('encrypted_r1')
            results.pop('encrypted_r2')

        train.save_results(results)  # saving results simulates push of image
        logging.info('Stored train results')
    logging.info('\n ------ \n PROXY STATION')

    # pp_auc = proxy_station() # TODO return list with TP, (TP+FN), TN and (TN+FP) for each threshold value
    proxy_station()

    # TODO itererate over stations and calcucale TP/(TP+FN) and TN/(TN+FP) for each threshold value
    # stations_auc(0)
    for i in range(stations):
        AUC = stations_auc(i)
    print(AUC == auc_gt)