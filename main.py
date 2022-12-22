import copy
import logging
import os
import pickle
import shutil
import time
from random import randint
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric import rsa
from matplotlib.pyplot import cm
from sklearn import metrics
import seaborn as sns
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


def create_protocol_data():
    """
    Create data used in protocol
    """
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


def create_synthetic_data(num_stations=int, samples=int, fake_patients=None):
    """
    Create and save synthetic data of given number of samples and number of stations. Including flag patients
    """
    for station_i in range(num_stations):
        valid = False
        while not valid:
            fake_data_val = randint(fake_patients[0], fake_patients[1]) # random number flag value in given percentage range
            data = {"Pre": np.random.randint(low=5, high=100, size=samples+fake_data_val),
                    "Label": np.random.choice([0, 1], size=samples+fake_data_val, p=[0.1, 0.9]),
                    "Flag": np.random.choice(np.concatenate([[1] * samples, [0] * fake_data_val]),
                                             samples+fake_data_val, replace=False)}

            df = pd.DataFrame(data, columns=['Pre', 'Label', 'Flag'])
            df.loc[df["Flag"] == 0, "Label"] = 0  # when Flag is 0 Label must also be 0

            if not np.all(df["Label"] == 1):
                valid = True

            df.to_pickle('./data/synthetic/data_s' + str(station_i + 1) + '.pkl')


def calculate_regular_auc(stations, protocol, performance):
    """
    Calculate AUC with sklearn as ground truth GT
    """
    lst_df = []
    for i in range(stations):
        if protocol:
            df_i = pickle.load(open('./data/synthetic/protocol_data_s' + str(i+1) + '.pkl', 'rb'))
        else:
            df_i = pickle.load(open('./data/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))
        lst_df.append(df_i)

    concat_df = pd.concat(lst_df)
    #print('All unique Pre? ', concat_df["Pre"].is_unique)
    performance['samples'].append(len(concat_df))
    print('Use data from {} stations. Total of {} subjects (including flag subjects) '.format(stations, len(concat_df)))

    #sort_df = concat_df.sort_values(by='Pre', ascending=False).reset_index()
    sort_df = concat_df.sort_values(by='Pre', ascending=False)

    debugging = False

    if debugging:
        print("Data Predi: {}".format(sort_df["Pre"].to_list()))
        print("Data Label: {}".format(sort_df["Label"].to_list()))
        print("Data Flags: {}".format(sort_df["Flag"].to_list()))
    filtered_df = sort_df[sort_df["Flag"] == 1]  # remove flag patients
    # iterate over sorted list
    # auc = 0.0
    # height = 0.0
    dfd = filtered_df.copy()
    dfd["Pre"] = filtered_df["Pre"] / 100
    y = dfd["Label"]
    pred = dfd["Pre"]

    #fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    #auc = metrics.auc(fpr, tpr)
    auc = metrics.roc_auc_score(y, pred)


    # plt.figure()
    # lw = 1
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC of ground truth')
    # plt.legend(loc="lower right")
    # plt.show()

    # return exact auc to be benchmarked with
    return auc, performance

def generate_keys(stations, results):
    """
    Generate and save keys of given numbers of stations and train results
    """
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

def encrypt_table(station_df, agg_pk, r1, r2, symmetric_key, station):
    """
    Encrypt dataframe of given station with paillier public key of aggregator and random values
    """
    logging.info('Start encrypting table with {} subjects from station {}'.format(len(station_df), station))
    #print('Start encrypting table with {} subjects from station {}'.format(len(station_df), station))
    tic = time.perf_counter()
    # Just trivial implementation - improve with vectorizing and
    station_df["Pre"] *= r1
    station_df["Pre"] += r2
    station_df["Pre"] = station_df["Pre"].apply(lambda x: Fernet(symmetric_key).encrypt(int(x).to_bytes(2, 'big')))
    # Step 1
    station_df["Label"] = station_df["Label"].apply(lambda x: encrypt(agg_pk, x))
    station_df["Flag"] = station_df["Flag"].apply(lambda x: encrypt(agg_pk, x))
    toc = time.perf_counter()
    #print(f'Encryption time of table {toc - tic:0.4f} seconds')
    logging.info(f'Encryption time {toc - tic:0.4f} seconds')
    #print(f'Encryption time {toc - tic:0.4f} seconds')

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

def encrypt_symmetric_key(station, symmetric_key):
    """
    Encrypt symmetric key_station with public rsa key of aggregator
    return: encrypted_symmetric_key
    """
    logging.info('Symmetric key of k_{} is: {}'.format(station, symmetric_key))
    rsa_agg_pk = load_rsa_pk('./data/keys/agg_rsa_public_key.pem')
    encrypted_symmetric_key = rsa_agg_pk.encrypt(symmetric_key, padding.OAEP(
                                                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                                                algorithm=hashes.SHA256(),
                                                label=None
                                            ))

    return encrypted_symmetric_key

def decrypt_symmetric_key(station, ciphertext):
    """
    Decrypt of given station rsa encrypted k_station
    """
    logging.info('Symmetric key of k_{} encrypted is: {}'.format(station, ciphertext))
    rsa_agg_sk = load_rsa_sk('./data/keys/agg_rsa_private_key.pem')
    decrypted_symmetric_key = rsa_agg_sk.decrypt(
        ciphertext,
        padding.OAEP(
         mgf=padding.MGF1(algorithm=hashes.SHA256()),
         algorithm=hashes.SHA256(),
         label=None
        ))
    logging.info('Symmetric key of k_{} decrypted is: {}'.format(station, decrypted_symmetric_key))
    return decrypted_symmetric_key

def pp_auc_protocol(station_df, station=int):
    """
    Perform PP-AUC protocol at specific station given dataframe
    """
    prev_results = train.load_results()  # loading results simulates pull of image
    agg_pk = prev_results['aggregator_paillier_pk']
    if station == 1:
        # Step 2
        r1 = randint(1, 100)  # random value between 1 to 100
        r2 = randint(1, 100)

        symmetric_key = Fernet.generate_key()  # represents k1
        enc_table = encrypt_table(station_df, agg_pk, r1, r2, symmetric_key, station)

        # Save for transparency the table - not required
        enc_table.to_pickle('./data/encrypted/data_s' + str(station) + '.pkl')

        # Step 3 - 1
        enc_symmetric_key = encrypt_symmetric_key(station, symmetric_key)
        # Step 3 - 2 (partial Decrypt enc(k) with x1 of pk
        # used RSA encryption of symmetric key (Fernet) - Document and ask Mete
        #partial_private_key = pickle.load(open('./data/keys/agg_sk.p', 'rb'))
        #partial_decrypted = proxy_decrypt(partial_private_key, symmetric_key)

        prev_results['encrypted_ks'].append(enc_symmetric_key)

        # Step 4
        for i in range(len(prev_results['stations_rsa_pk'])):
            enc_r1 = encrypt(prev_results['stations_paillier_pk'][i], r1)  # homomorphic encryption used
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
        # Step 8
        dec_r1 = decrypt(sk_s_i, enc_r1)
        logging.info('Decrypted at station {} encrypted r1 {} to {}'.format(station, enc_r1, dec_r1))
        enc_r2 = prev_results['encrypted_r2'][station-1]
        dec_r2 = decrypt(sk_s_i, enc_r2)
        logging.info('Decrypted at station {} encrypted r1 {} to {}'.format(station, enc_r2, dec_r2))

        # Step 9 / 10
        symmetric_key = Fernet.generate_key()  # represents k_2 to k_n
        enc_table = encrypt_table(station_df, agg_pk, dec_r1, dec_r2, symmetric_key, station)

        # Step 11
        enc_symmetric_key = encrypt_symmetric_key(station, symmetric_key)
        # Step 12
        prev_results['encrypted_ks'].append(enc_symmetric_key)

    prev_results['pp_auc_tables'][station-1] = enc_table

    return prev_results

def sum_over_enc_series(encrypted_series, agg_pk):
    """
    Compute encrypted sum over given series
    """
    if len(encrypted_series) == 1:
        return encrypted_series[0]
    else:
        res = encrypt(agg_pk, 0)
        for cipher in encrypted_series:
            res = add(agg_pk, res, cipher)
        return res

def z_values(n):
    """
    Generate random values of list length n which sum is zero
    """
    l = random.sample(range(-int(n/2), int(n/2)), k=n-1)
    return l + [-sum(l)]

def proxy_station():
    """
    Simulation of aggregator service - globally computes privacy preserving AUC table
    """
    # Step 21
    results = train.load_results()
    agg_pk = results['aggregator_paillier_pk']
    agg_sk = pickle.load(open('./data/keys/agg_sk_1.p', 'rb'))

    # decrypt symmetric keys (k_stations)
    df_list = []
    for i in range(len(results['encrypted_ks'])):
        enc_k_i = results['encrypted_ks'][i]
        # Step 22
        dec_k_i = decrypt_symmetric_key(i, enc_k_i)
        logging.info('Decrypted k value {} of station {}'.format(dec_k_i, i + 1))

        # Step 23 decrypt table values with Fernet and corresponding k_i symmetric key
        table_i = results['pp_auc_tables'][i]
        table_i["Dec_pre"] = table_i["Pre"].apply(lambda x: Fernet(dec_k_i).decrypt(x))  # returns bytes
        table_i["Dec_pre"] = table_i["Dec_pre"].apply(lambda x: int.from_bytes(x, "big"))
        df_list.append(table_i)

    concat_df = pd.concat(df_list)
    concat_df.pop('Pre')
    logging.info('\n')
    sort_df = concat_df.sort_values(by='Dec_pre', ascending=False)
    df_new_index = sort_df.reset_index()

    M = len(df_new_index)
    tp_values = []
    fp_values = []

    tp_values.insert(0, encrypt(agg_pk, 0))
    fp_values.insert(0, encrypt(agg_pk, 0))
    tmp_sum = fp_values[0]
    for i in range(1, M + 1):
        tp_values.append(e_add(agg_pk, tp_values[i - 1], df_new_index['Label'][i - 1]))
        value_add = df_new_index['Flag'][i - 1]
        sum_flags = add(agg_pk, value_add, tmp_sum)
        tmp_sum = sum_flags
        fp_values.append(e_add(agg_pk, sum_flags, mul_const(agg_pk, tp_values[-1], -1)))

    a = randint(1, 100)
    b = randint(1, 100)

    # Denominator
    # TP_A is summation of labels (TP)
    tp_a_mul = mul_const(agg_pk, tp_values[-1], a)
    fp_a_mul = mul_const(agg_pk, fp_values[-1], b)
    #
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
    pred = sort_df["Dec_pre"].to_list()
    for i in range(M - 1):
        if pred[i] != pred[i + 1]:
            thre_ind.append(i)

    thre_ind = list(map(lambda x: x + 1, thre_ind)) # add one

    # Multiply with a and b respectively
    Z_values = z_values(len(thre_ind)-1)
    for i in range(1, len(thre_ind)):
        pre_ind = thre_ind[i - 1]
        cur_ind = thre_ind[i]
        dFP_t = mul_const(agg_pk, e_add(agg_pk, fp_values[cur_ind], mul_const(agg_pk, fp_values[pre_ind], -1)), b)
        sTP_t = mul_const(agg_pk, e_add(agg_pk, tp_values[cur_ind],  tp_values[pre_ind]), a)
        r1_i = randint(1, 100)
        r2_i = randint(1, 100)
        results["N1"].append(proxy_decrypt(agg_sk, add_const(agg_pk, sTP_t, r1_i)))
        results["N2"].append(proxy_decrypt(agg_sk, add_const(agg_pk, dFP_t, r2_i)))
        N_i3_1 = mul_const(agg_pk, sTP_t, r2_i)
        N_i3_2 = mul_const(agg_pk, dFP_t, r1_i)
        N_i3_a = add(agg_pk, N_i3_1, add_const(agg_pk, N_i3_2, r1_i * r2_i))
        results["N3"].append(proxy_decrypt(agg_sk, add_const(agg_pk, N_i3_a, Z_values[i-1])))

    train.save_results(results)


def stations_auc(station):
    """
    Simulation of station delegated AUC parts to compute global AUC locally
    """
    train_results = train.load_results()
    agg_sk_2 = pickle.load(open('./data/keys/agg_sk_2.p', 'rb')) # todo split sk in sk_1 and sk_2
    agg_pk = train_results['aggregator_paillier_pk']
    logging.info('Station {}:\n'.format(station+1))

    # decrypt random components D1, D2, D3, Ni1, Ni2, Ni3
    D1 = station_decrypt(agg_sk_2, train_results['D1'][0])
    D2 = station_decrypt(agg_sk_2, train_results['D2'][0])
    D3 = station_decrypt(agg_sk_2, train_results['D3'][0])

    N = 0

    for j in range(len(train_results['N1'])):
        n_i1 = station_decrypt(agg_sk_2, train_results['N1'][j])
        n_i2 = station_decrypt(agg_sk_2, train_results['N2'][j])
        n_i3 = station_decrypt(agg_sk_2, train_results['N3'][j])

        if n_i3 != 0 and int(math.log10(n_i3)) + 1 >= 10:
            n_i3 = -(agg_pk.n - n_i3)
        else:
            pass

        N += ((n_i1 * n_i2) - n_i3)
    D = ((D1 * D2) - D3)

    if D == 0:
        auc = 0
    else:
        auc = (N / D) / 2
    logging.info('PP-AUC: {0:.3f}'.format(auc))
    print('PP-AUC: {}'.format(auc))
    return auc


def plot_results(res):
    perf = pd.DataFrame(list(zip(res['pp-auc'], res['gt-auc'])), index=res['stations'], columns=['pp', 'gt'])
    total_time = [sum(x) for x in zip(*[res['time']['total_step_1'], res['time']['proxy'], res['time']['stations_2']])]
    df = pd.DataFrame(list(zip(res['time']['stations_1'], res['time']['proxy'],
                               res['time']['stations_2'], total_time, res['samples'], res['stations'])),
                      index=res['stations'],
                      columns=['Station_1', 'Proxy', 'Station_2', 'Total', 'Samples', 'Stations'])
    color = iter(cm.rainbow(np.linspace(0, 1, len(df.Stations.unique()))))

    #pd.pivot_table(df.reset_index(), index='Samples', columns='Stations', values='Total').plot(subplots=True, layout=(1, 3))
    # for category in df.Stations.unique():
    #     c = next(color)
    #     plt.plot('Samples', 'Station_1', c=c, data=df.loc[df['Stations'].isin([category])], marker='x',
    #              linestyle=':', label=str(category) + ' S - Step 1')
    #     plt.plot('Samples', 'Proxy', c=c, data=df.loc[df['Stations'].isin([category])], marker='o',
    #              linestyle=' ', label=str(category) + ' S Proxy')
    #     plt.plot('Samples', 'Station_2', data=df.loc[df['Stations'].isin([category])], marker='x',
    #              c=c, linestyle='-.', label=str(category) + "  S - Step 2")
    #     plt.plot('Samples', 'Total', c=c, data=df.loc[df['Stations'].isin([category])], marker='x',
    #              label=str(category) + ' S Total')
    for category in df.Stations.unique():
        plt.plot('Samples', 'Total', c="blue", data=df.loc[df['Stations'].isin([category])], marker='o',
                 label=str(category) + 'Total runtime')
    plt.xlabel('Number of subjects')
    num_stations = res['stations'][0]
    plt.ylabel('Time (sec)')
    plt.title('DPPE-AUC total runtime w/ ' + str(num_stations) + ' stations')
    #plt.legend(loc="upper left")
    plt.show()
    #plt.savefig('runtime-s-' + str(num_stations) + '.png')

    # Enter raw dataF
    gt = perf['gt']
    #print(gt)
    pp = perf['pp']
    diff = gt - pp
    print(diff)

    # Calculate the average
    gt_mean = np.mean(diff)
    pp_mean = np.mean(pp)

    # Calculate the standard deviation
    gt_std = np.std(diff)
    pp_std = np.std(pp)

    # Create lists for the plot
    materials = ['GT-AUC']
    x_pos = np.arange(len(materials))
    CTEs = [gt_mean]
    error = [gt_std]

    # fig, ax = plt.subplots()
    # sns.boxplot(x=np.zeros(len(diff)), y=diff)
    # ax.set_ylabel('difference')
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(materials)
    # ax.set_title('AUC difference GT to PP')
    # ax.yaxis.grid(True)

    # Save the figure and show
    #plt.tight_layout()
    #plt.savefig('bar_plot_with_error_bars.png')
    #plt.show()


if __name__ == "__main__":
    # Configuration
    logging.basicConfig(filename='pp-auc.log', level=logging.INFO)
    logging.info('Start PP-AUC execution')

    #stations = 3  # TODO adjust

    protocol = False # if protocol true, then: subject_list = [20]
    #subject_list = [15]

    station_list = [3]# ,12]
    subject_list = [25, 50, 100, 200, 400, 800, 1600] # 320, 640, 1280, 2560, 5120, 10240]
    #per = {'time': {'stations_1': [0.0238299579990174, 0.02750213200003297, 0.04754705566544241, 0.04858811799931573, 0.09245213866718889, 0.08697813883676038], 'proxy': [0.381446125000366, 0.7966919999889797, 0.7213030839920975, 1.201679542005877, 1.0368845000048168, 1.7390484170027776], 'stations_2': [0.13467783300438896, 0.2640320829959819, 0.27925649999815505, 0.38997579099668656, 0.3639489589986624, 0.478104624999105], 'total_step_1': [0.0714898739970522, 0.1650127920001978, 0.14264116699632723, 0.29152870799589437, 0.2773564160015667, 0.5218688330205623]}, 'samples': [30, 68, 67, 133, 138, 267], 'stations': [3, 6, 3, 6, 3, 6], 'pp-auc': [0.7727272727272727, 0.6595744680851063, 0.09259259259259259, 0.4654895666131621, 0.5810936051899908, 0.5806722689075631], 'gt-auc': [0.7727272727272727, 0.6595744680851063, 0.09259259259259262, 0.4654895666131621, 0.5810936051899908, 0.580672268907563]}
    #plot_results(per)
    #exit(0)
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
            performance['stations'].append(stations)
            try:
                shutil.rmtree('./data/')
                logging.info('Backuped and deleted previous results')
            except Exception as e:
                logging.info('No previous files and results to remove')

            directories = ['./data', './data/keys', './data/synthetic', './data/encrypted', './data/pht_results']
            for dir in directories:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            if protocol:
                create_protocol_data()
            else:
                create_synthetic_data(stations, subjects, [int(subjects*.30), int(subjects*.50)])
                pass
            results = train.load_results()
            results = generate_keys(stations, results)
            # Train Building process
            train.save_results(results)

            # compute AUC without encryption for proof of principal of pp_auc
            auc_gt, performance = calculate_regular_auc(stations, protocol, performance)
            performance['gt-auc'].append(auc_gt)
            logging.info('AUC value of ground truth {}'.format(auc_gt))
            print('AUC value of GT {}'.format(auc_gt))
            times = []
            for i in range(stations):
                if protocol:
                    stat_df = pickle.load(open('./data/synthetic/protocol_data_s' + str(i+1) + '.pkl', 'rb'))
                else:
                    stat_df = pickle.load(open('./data/synthetic/data_s' + str(i+1) + '.pkl', 'rb'))
                logging.info('\n')

                t1 = time.perf_counter()
                results = pp_auc_protocol(stat_df, station=i+1)
                t2 = time.perf_counter()
                times.append(t2 - t1)

                # remove at last station all encrypted noise values
                if i is stations - 1:
                    logging.info('Remove paillier_pks and all encrypted r1 and r2 values')
                    results.pop('encrypted_r1')
                    results.pop('encrypted_r2')

                train.save_results(results)  # saving results simulates push of image
                logging.info('Stored train results')

            print(f'Total execution time at stations {sum(times):0.4f} seconds')
            print(f'Average execution time at stations {sum(times)/len(times):0.4f} seconds')
            performance['time']['stations_1'].append(sum(times)/len(times))
            performance['time']['total_step_1'].append(sum(times))
            logging.info('\n ------ \n PROXY STATION')

            t3 = time.perf_counter()
            proxy_station()
            t4 = time.perf_counter()
            performance['time']['proxy'].append(t4 - t3)
            print(f'Execution time by proxy station {t4 - t3:0.4f} seconds')

            t1 = time.perf_counter()
            auc_pp = stations_auc(0)
            t2 = time.perf_counter()
            performance['time']['stations_2'].append(t2 - t1)
            print(f'Final AUC execution time at station {t2 - t1:0.4f} seconds')

            performance['pp-auc'].append(auc_pp)

            # for i in range(stations):
            #    AUC = stations_auc(i)
            print('Equal GT? {}'.format(auc_gt == auc_pp))
            diff = auc_gt - auc_pp
            differences.append(diff)
            print('Difference pp-AUC to GT: ', diff)
            print('\n')

        print("Avg difference {} over {} runs".format(sum(differences)/len(differences), len(differences)))
        # back up data and remove directory
        try:
            shutil.move('./data/', './bk-data-s' + subjects + '-n-' + stations)
            shutil.rmtree('./data/')
            print('Saved prev data')
            logging.info('Saved and deleted previous results')
        except Exception as e:
            logging.info('No previous files and results to remove')

    print(performance)
    plot_results(performance)