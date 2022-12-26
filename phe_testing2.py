from paillier2.paillier import *
import time


if __name__ == "__main__":
    priv, pub = generate_keypair(3072)
    print("testtt")
    number = 3

    tic = time.perf_counter()
    cipher_1 = encrypt(pub,number)
    toc = time.perf_counter()
    print(f"Encrypted in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    cipher_2 = mul_const(pub, cipher_1, 3)
    toc = time.perf_counter()
    print(f"MUL with Constant in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    cipher_3 = add_const(pub, cipher_2, -5)
    toc = time.perf_counter()
    print(f"ADD with Constant in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    cipher_result = add(pub, cipher_2, cipher_3)
    toc = time.perf_counter()
    print(f"ADD in {toc - tic:0.4f} seconds")


    tic = time.perf_counter()
    plain_test = decrypt(priv, cipher_result)
    toc = time.perf_counter()
    print(f"DEC in {toc - tic:0.4f} seconds")
    print(plain_test)


    tic = time.perf_counter()
    cipher_3_1 = proxy_decrypt(priv, cipher_3)
    toc = time.perf_counter()
    print(f"Pro DEC in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    plain_test = station_decrypt(priv, cipher_3_1)
    toc = time.perf_counter()
    print(f"Sta DEC in {toc - tic:0.4f} seconds")
    print(plain_test)
