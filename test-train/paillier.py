from paillier_util import *


def miller_rabin(n, k):
    """Run the Miller-Rabin test on n with at most k iterations

    Arguments:
        n (int): number whose primality is to be tested
        k (int): maximum number of iterations to run

    Returns:
        bool: If n is prime, then True is returned. Otherwise, False is
        returned, except with probability less than 4**-k.

    See <https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test>
    """
    assert n > 3

    # find r and d such that n-1 = 2^r × d
    d = n-1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    assert n-1 == d * 2**r
    assert d % 2 == 1

    for _ in range(k):  # each iteration divides risk of false prime by 4
        a = random.randint(2, n-2)  # choose a random witness

        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue  # go to next witness

        for _ in range(1, r):
            x = x*x % n
            if x == n-1:
                break   # go to next witness
        else:
            return False
    return True


def is_prime(n, mr_rounds=25):
    """Test whether n is probably prime

    See <https://en.wikipedia.org/wiki/Primality_test#Probabilistic_tests>

    Arguments:
        n (int): the number to be tested
        mr_rounds (int, optional): number of Miller-Rabin iterations to run;
            defaults to 25 iterations, which is what the GMP library uses

    Returns:
        bool: when this function returns False, `n` is composite (not prime);
        when it returns True, `n` is prime with overwhelming probability
    """
    # as an optimization we quickly detect small primes using the list above
    if n <= first_primes[-1]:
        return n in first_primes
    # for small dividors (relatively frequent), euclidean division is best
    for p in first_primes:
        if n % p == 0:
            return False
    # the actual generic test; give a false prime with probability 2⁻⁵⁰
    return miller_rabin(n, mr_rounds)


class PrivateKey(object):
    def __init__(self, n, x):
        self.n = n
        self.x = x
        self.x1 = random.randrange(1, x-1, 1)
        self.x2 = x - self.x1
        self.nsqr = pow(n, 2)

    def __repr__(self):
        return '<PrivateKey: %s %s>' % (self.x, self.n)


class PrivateKeyOne(object):
    def __init__(self, n, x1):
        self.n = n
        self.x1 = x1
        self.nsqr = pow(n, 2)

    def __repr__(self):
        return '<PrivateKey_1: %s %s>' % (self.x, self.x1)


class PrivateKeyTwo(object):
    def __init__(self, n, x2):
        self.n = n
        self.x2 = x2
        self.nsqr = pow(n, 2)

    def __repr__(self):
        return '<PrivateKey_2: %s %s>' % (self.x, self.x1)


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
