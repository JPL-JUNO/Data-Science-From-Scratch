"""
@Description: 题外话：tqdm
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-08-18 17:47:12
"""

import tqdm


def primes_to_n(n: int) -> list[int]:
    primes = [2]

    with tqdm.trange(3, n) as t:
        for i in t:
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)
            t.set_description(f"{len(primes)} primes")
    return primes


my_primes = primes_to_n(100000)
