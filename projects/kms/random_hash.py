import hashlib
import random

# Generate random data
random_data = str(random.getrandbits(128)).encode("utf-8")

# Generate MD5 hash
md5_hash = hashlib.md5(random_data).hexdigest()[:8]

print(md5_hash)
