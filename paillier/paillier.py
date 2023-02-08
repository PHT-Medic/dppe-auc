import random
from .util import invert, isInvertible, powmod, mulmod, getprimeover


class PrivateKey(object):
    def __init__(self, n, x):
        self.n = n
        self.x = x
        self.x1 = random.randrange(1, x-1, 1)
        self.x2 = x - self.x1
        self.nsqr = pow(n, 2)

    def __repr__(self):
        return '<PrivateKey: %s %s>' % (self.x, self.n)


class PublicKey(object):
    def __init__(self, n, g, x):
        self.n = n
        self.g = g
        self.nsqr = pow(n, 2)
        self.h = powmod(g, x, self.nsqr)
        self.r = pow(128, 2)

    def __repr__(self):
        return '<PublicKey: %s %s %s>' % (self.n, self.g, self.h)


def random_element(n):
    g = random.SystemRandom().randrange(1, n)
    while True:
        if isInvertible(g, n):
            break
        else:
            g = random.SystemRandom().randrange(1, n)

    return g


def choose_g(n):
    a = random_element(pow(n, 2))
    g = powmod((-1*a), (2*n), pow(n, 2))
    return g


def generate_keypair(bits):
    p = getprimeover(bits // 2)
    q = getprimeover(bits // 2)
    n = p * q
    x = random.SystemRandom().randrange(1, pow(n, 2) >> 1)
    g = choose_g(n)
    return PrivateKey(n, x), PublicKey(n, g, x)


def encrypt(pub, plain):
    r = random.SystemRandom().randrange(1, pub.r)
    c1 = powmod(pub.g, r, pub.nsqr)
    c2 = (powmod(pub.h, r, pub.nsqr) * (1+((plain * pub.n) % pub.nsqr) % pub.nsqr)) % pub.nsqr
    return [c1, c2]


def add(pub, a, b):
    """Add one encrypted integer to another"""
    return [mulmod(a[0], b[0], pub.nsqr), mulmod(a[1], b[1], pub.nsqr)]


def mul_const(pub, a, n):
    """Multiplies an encrypted integer by a constant"""
    return [powmod(a[0], n, pub.nsqr), powmod(a[1], n, pub.nsqr)]


def add_const(pub, a, n):
    """Add one encrypted integer to a constant"""
    b = encrypt(pub, n)
    return [mulmod(a[0], b[0], pub.nsqr), mulmod(a[1], b[1], pub.nsqr)]


def decrypt(priv, cipher):
    cinv = invert(powmod(cipher[0], priv.x, priv.nsqr), priv.nsqr)
    u = mulmod(cipher[1], cinv, priv.nsqr) - 1
    plain = u//priv.n
    return plain


def proxy_decrypt(priv, cipher):
    cinv = invert(powmod(cipher[0], priv.x1, priv.nsqr), priv.nsqr)
    cipher[1] = mulmod(cipher[1], cinv, priv.nsqr)
    return cipher


def station_decrypt(priv, cipher):
    cinv = invert(powmod(cipher[0], priv.x2, priv.nsqr), priv.nsqr)
    u = mulmod(cipher[1], cinv, priv.nsqr) - 1
    plain = u//priv.n
    return plain


def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    length = ((i + ((i * signed) < 0)).bit_length() + 7 + signed) // 8
    return i.to_bytes(length, byteorder='big', signed=signed)


def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    return int.from_bytes(b, byteorder='big', signed=signed)
