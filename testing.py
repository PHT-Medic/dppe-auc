import main
from paillier.paillier import *
from cryptography.fernet import Fernet
from main import *

if __name__ == "__main__":
    """
    Test function to test partial en/ decryption
    """
    directories = ['./data', './data/keys', './data/synthetic', './data/encrypted', './data/pht_results']
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
    create_protocol_data()
    print("k1 as fernet key")
    priv_p, pub_p = generate_keypair(512)

    # Station side creates k1
    symm_key = Fernet.generate_key()  # try to use 16 bytes instead of 32
    print("k1 Symm key:\t\t{}".format(symm_key)) # symm key in bytes
    #int_key = bytes_to_int(symm_key, signed=True)  # convert to int, to encrypt with pk
    #int_key = int.from_bytes(symm_key, 'big')
    #print("Int symm key:\t{}".format(int_key))

    en_key = main.encrypt_symm_key(1, symm_key)
    #en_key = encrypt(pub_p, int_key)  # Paillier encrypt int key

    print("Encrypted symm key:\t{}".format(en_key))

    part_de2 = main.decrypt_symm_key(1, en_key)
    #part_de = proxy_decrypt(priv_p, en_key) # used by stations

    # Proxy station needs to receive k1 as bytes to decrypt Pre values with Fernet
    #part_de2 = decrypt2(priv_p, part_de)  # used by proxy station -> decrypted int key as float

    print("Decrypted symm key:\t{}".format(part_de2))

    # Aim is to receive bytes of symm_key to use with fernet

    #int_k1 = int(part_de2)  # convert float to int TODO int_k1 is NOT equal int_key! converstion issue
    #dec_symm = int_to_bytes(int_k1, signed=True) # convert int to bytes of symm key
    print("Is decrypted (int) symm key equals the int k1 key: {}\n"
          "Int symm key:\t\t    {} \nInt decrypted symm key:\t{}".format(symm_key == part_de2,
                                                                     symm_key, part_de2))



    #print("Converted decrypted symm key: {}".format(dec_symm))

    # By mete
    print("\n")
    print("Partial encryption")

    priv, pub = generate_keypair(512)
    number = 2202148856699959262278163878129583615742140072329252762777117269347956711482983433116476160208458081521712
    print("Symm key (large number):", number)
    cipher1 = encrypt(pub, number)
    cipher2 = encrypt(pub,5)
    cipher3 = add(pub, cipher1, cipher2)
    number = number + 5  # plaintext operation
    cipher4 = mul_const(pub, cipher3, 3)
    number = number * 3 # plaintext operation
    cipher5 = add_const(pub,cipher4,9)
    number = number + 9 # plaintext operation
    plain1 = decrypt(priv,cipher5)
    print("HE regular decrypted: ",plain1)
    p_cipher5 = proxy_decrypt(priv,cipher5)
    print("Partial - step 1 -  HE decrypted: ",p_cipher5)
    plain2 = decrypt2(priv,p_cipher5)
    print("Partial - step 2 -  HE decrypted: ",plain2)
    print("GT: {} Outcome: {} \n equal?: {}".format(number, plain2, plain2 == number))



