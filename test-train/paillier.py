import random
from paillier_util import powmod, mulmod, invert, isInvertible, getprimeover
import gmpy2  # Für optimierte mathematische Operationen

class PrivateKey:
    def __init__(self, n, x):
        self.n = n
        self.x = x
        self.x1 = random.SystemRandom().randrange(1, x - 1)
        self.x2 = x - self.x1
        self.nsqr = n * n  # Precomputed for efficiency

    def __repr__(self):
        return f'<PrivateKey: {self.x} {self.n}>'

class PrivateKeyOne:
    def __init__(self, n, x1):
        self.n = n
        self.x1 = x1
        self.nsqr = n * n  # Precomputed

    def __repr__(self):
        return f'<PrivateKey_1: {self.x1} {self.n}>'

class PrivateKeyTwo:
    def __init__(self, n, x2):
        self.n = n
        self.x2 = x2
        self.nsqr = n * n  # Precomputed

    def __repr__(self):
        return f'<PrivateKey_2: {self.x2} {self.n}>'

class PublicKey:
    def __init__(self, n, g, x):
        self.n = n
        self.g = g
        self.nsqr = n * n  # Precomputed
        self.h = gmpy2.powmod(g, x, self.nsqr)  # Optimized modular exponentiation
        self.r = 128 * 128  # Equivalent to pow(128, 2), simplified

    def __repr__(self):
        return f'<PublicKey: {self.n} {self.g} {self.h}>'

def random_element(n):
    """
    Generate a random element from 1 to n that is invertible modulo n.
    """
    rand_gen = random.SystemRandom()
    g = rand_gen.randrange(1, n)
    while not isInvertible(g, n):
        g = rand_gen.randrange(1, n)
    return g

def choose_g(n):
    """
    Choose a generator g for the Paillier cryptosystem.
    """
    a = random_element(n * n)
    g = gmpy2.powmod(-a, 2 * n, n * n)  # Optimized with gmpy2
    return g

def generate_keypair(bits):
    """
    Generate a key pair for the Paillier cryptosystem.
    """
    p = getprimeover(bits // 2)
    q = getprimeover(bits // 2)
    n = p * q
    x = random.SystemRandom().randrange(1, (n * n) >> 1)
    g = choose_g(n)
    return PrivateKey(n, x), PublicKey(n, g, x)

def encrypt(pub, plain):
    """
    Encrypt an integer using the public key.
    """
    rand_gen = random.SystemRandom()
    r = rand_gen.randrange(1, pub.r)
    c1 = gmpy2.powmod(pub.g, r, pub.nsqr)
    c2 = (gmpy2.powmod(pub.h, r, pub.nsqr) * (1 + (plain * pub.n) % pub.nsqr)) % pub.nsqr
    return [c1, c2]

def add(pub, a, b):
    """
    Add two encrypted integers.
    """
    return [mulmod(a[0], b[0], pub.nsqr), mulmod(a[1], b[1], pub.nsqr)]

def mul_const(pub, a, n):
    """
    Multiply an encrypted integer by a constant.
    """
    return [gmpy2.powmod(a[0], n, pub.nsqr), gmpy2.powmod(a[1], n, pub.nsqr)]

def add_const(pub, a, n):
    """
    Add a constant to an encrypted integer.
    """
    b = encrypt(pub, n)
    return [mulmod(a[0], b[0], pub.nsqr), mulmod(a[1], b[1], pub.nsqr)]

def decrypt(priv, cipher):
    """
    Decrypt an encrypted integer using the private key.
    """
    cinv = invert(gmpy2.powmod(cipher[0], priv.x, priv.nsqr), priv.nsqr)
    u = mulmod(cipher[1], cinv, priv.nsqr) - 1
    plain = u // priv.n
    return plain

def proxy_decrypt(priv, cipher):
    """
    Perform a proxy decryption step with part of the private key.
    """
    cinv = invert(gmpy2.powmod(cipher[0], priv.x1, priv.nsqr), priv.nsqr)
    cipher[1] = mulmod(cipher[1], cinv, priv.nsqr)
    return cipher

def station_decrypt(priv, cipher):
    """
    Perform the final decryption step at the station.
    """
    cinv = invert(gmpy2.powmod(cipher[0], priv.x2, priv.nsqr), priv.nsqr)
    u = mulmod(cipher[1], cinv, priv.nsqr) - 1
    plain = u // priv.n
    return plain

def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    """
    Convert an integer to a byte array.
    """
    length = (i.bit_length() + 7) // 8 or 1
    return i.to_bytes(length, byteorder='big', signed=signed)

def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    """
    Convert a byte array to an integer.
    """
    return int.from_bytes(b, byteorder='big', signed=signed)