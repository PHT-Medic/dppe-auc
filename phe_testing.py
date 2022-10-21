#from phe import paillier
from paillier.paillier import *
#public_key, private_key = paillier.generate_paillier_keypair()
#sec_num = public_key.encrypt(10)
#neg_num = sec_num * (-1)

#private_key.decrypt(sec_num)
#private_key.decrypt(neg_num)


priv, pub = generate_keypair(512)
number = 3
cipher_1 = encrypt(pub,number)
cipher_2 = mul_const(pub, cipher_1, -1)

number2 = 4
cipher_3 = encrypt(pub,number2)
cipher_result = add(pub, cipher_2, cipher_3)

print(number2-number)
plain_test = decrypt(priv, cipher_result)
print(plain_test)