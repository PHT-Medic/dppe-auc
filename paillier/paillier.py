import math
from .primes import generate_prime
import random

def invmod(a, p, maxiter=1000000):
    """The multiplicitive inverse of a in the integers modulo p:
         a * b == 1 mod p
       Returns b.
       (http://code.activestate.com/recipes/576737-inverse-modulo-p/)"""
    if a == 0:
        raise ValueError('0 has no inverse mod %d' % p)
    r = a
    d = 1
    for i in range(min(p, maxiter)):
        d = ((p // r + 1) * d) % p
        r = (d * a) % p
        if r == 1:
            break
    else:
        raise ValueError('%d has no inverse mod %d' % (a, p))

    return d


def modpow(base, exponent, modulus):
    """Modular exponent:
         c = b ^ e mod m
       Returns c.
       (http://www.programmish.com/?p=34)"""
    result = 1
    while exponent > 0:
        if exponent & 1 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result


class PrivateKey(object):
    def __init__(self, n, x):
        self.n = n
        self.x = x
        self.x1 = random.randrange(1, x-1, 1)
        self.x2 = x - self.x1
        self.nsqr = pow(n,2)

    def __repr__(self):
        return '<PrivateKey: %s %s>' % (self.x, self.n)


class PublicKey(object):
    def __init__(self, n, g, x):
        self.n = n
        self.g = g
        self.nsqr = pow(n,2)
        self.h = pow(g,x,self.nsqr)

    def __repr__(self):
        return '<PublicKey: %s %s %s>' % (self.n, self.g, self.h)


def isinvmod(a, p, maxiter= 1000000):
    """The multiplicitive inverse of a in the integers modulo p:
         a * b == 1 mod p
       Returns b.
       (http://code.activestate.com/recipes/576737-inverse-modulo-p/)"""
    if a == 0:
        return False
    #raise ValueError('0 has no inverse mod %d' % p)
    r = a
    d = 1
    for i in range(min(p, maxiter)):
        d = ((p // r + 1) * d) % p
        r = (d * a) % p
        if r == 1:
            break
    else:
        return False
        #raise ValueError('%d has no inverse mod %d' % (a, p))
    return True


def randomElement(n):
    g = random.randrange(1, n, 1)
    while True:
        if isinvmod(g,n):
            break
        else:
            g = random.randrange(1, n, 1)

    return g

def chooseG(n):
    a = randomElement(pow(n,2))
    g = pow((-1*a),(2*n),pow(n,2))
    return g

def generate_keypair(bits):
    p = generate_prime(bits)
    q = generate_prime(bits)
    n = p * q
    x = random.randrange(1, pow(n,2)>>1, 1)
    g = chooseG(n)
    return PrivateKey(n, x), PublicKey(n,g,x)


def encrypt(pub, plain):
    r = random.randrange(1, pub.n/4, 1)
    c1 = pow(pub.g,r,pub.nsqr)
    c2 = (pow(pub.h,r,pub.nsqr) * (1+((plain * pub.n) % pub.nsqr) % pub.nsqr)) % pub.nsqr
    return [c1,c2]


def add(pub, a, b):
    """Add one encrypted integer to another"""
    a[0] = a[0] * b[0] % pub.nsqr
    a[1] = a[1] * b[1] % pub.nsqr
    return a


def e_add(pub, a, b):
    """Add one encrypted integer to another"""
    a[0] = a[0] * b[0] % pub.nsqr
    a[1] = a[1] * b[1] % pub.nsqr
    return a


def e_mul_const(pub, a, n):
    """Multiplies an ancrypted integer by a constant"""
    a[0] = pow(a[0],n,pub.nsqr)
    a[1] = pow(a[1],n,pub.nsqr)
    return a


def mul_const(pub, a, n):
    """Multiplies an ancrypted integer by a constant"""
    a[0] = pow(a[0],n,pub.nsqr)
    a[1] = pow(a[1],n,pub.nsqr)
    return a


def add_const(pub, a, n):
    """Add one encrypted integer to a constant"""
    b = encrypt(pub, n)
    a[0] = a[0] * b[0] % pub.nsqr
    a[1] = a[1] * b[1] % pub.nsqr
    return a


def decrypt(priv, cipher):
    cinv = invmod(pow(cipher[0],priv.x, priv.nsqr),priv.nsqr)
    u = ((cipher[1] * cinv % priv.nsqr) - 1) % priv.nsqr
    plain = u//priv.n
    return plain


def proxy_decrypt(priv, cipher):
    cinv = invmod(pow(cipher[0],priv.x1, priv.nsqr),priv.nsqr)
    cipher[1] = cipher[1] * cinv % priv.nsqr
    return cipher


def decrypt2(priv, cipher):
    cinv = invmod(pow(cipher[0],priv.x2, priv.nsqr),priv.nsqr)
    u = ((cipher[1] * cinv % pow(priv.n,2)) - 1) % pow(priv.n,2)
    plain = u//priv.n
    return plain


def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    length = ((i + ((i * signed) < 0)).bit_length() + 7 + signed) // 8
    return i.to_bytes(length, byteorder='big', signed=signed)


def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    return int.from_bytes(b, byteorder='big', signed=signed)
