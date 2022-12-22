from paillier.paillier import *


if __name__ == "__main__":
    priv, pub = generate_keypair(512)
    number = 3

    cipher_1 = encrypt(pub,number)
    cipher_2 = mul_const(pub, cipher_1, -1)
    cipher_1_neg = add_const(pub, cipher_1,-6)
    cipher_1_neg_added = add_const(pub, cipher_1_neg, 6)

    cipher_0_mul = mul_const(pub, cipher_1, 0)
    print("Dec number: {}".format(decrypt(priv, cipher_0_mul)))

    print("Number {}".format(number))
    #print("Enc number {}".format(cipher_1))
    
    print("Dec number: {}".format(decrypt(priv, cipher_1_neg_added)))
    #print("Neg Dec number: {}".format(decrypt(priv, cipher_2)))

    number2 = 4
    cipher_3 = encrypt(pub,number2)
    cipher_result = add(pub, cipher_2, cipher_3)

    print(number2-number)
    plain_test = decrypt(priv, cipher_result)
    print(plain_test)