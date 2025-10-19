from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
import pandas as pd
import pickle
import os

password = ''
salt = b""
encoded_data = pickle.load(open('data.pkl','rb'))

def encrypt_aes(key, plaintext):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext.encode()) + padder.finalize()

    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()
    return ciphertext, iv

def decrypt_aes(key: bytes, ciphertext: bytes, iv: bytes) -> str:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return plaintext.decode()

def generate_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return kdf.derive(passphrase.encode())

key = generate_key(password, salt)
try:
    for i in range(encoded_data.shape[0]):
        name = encoded_data.iloc[i]['name']
        iv_name = encoded_data.iloc[i]['iv_name']
        pin = encoded_data.iloc[i]['pno']
        ivpnos = encoded_data.iloc[i]['ivpnos']
        final_name = decrypt_aes(key, name, iv_name)
        final_pin = decrypt_aes(key, pin, ivpnos)
        print(final_name,' : ',final_pin)
except:
    print('Wrong Password and Seed') 
