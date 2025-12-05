import requests

from encryption_module import (
    create_ckks_context,
    serialize_public_context,
    encrypt_answers,
    serialize_ciphertext,
    deserialize_ciphertext,
    decrypt_vector,
)

BACKEND_URL = "http://127.0.0.1:5000/process"


def run_test_client():
    """
    Simple end-to-end test of the HE backend:
    1. Create CKKS context and encrypt example answers.
    2. Send encrypted vector + public context to the Flask backend.
    3. Receive encrypted score and decrypt it locally.
    """

    # 1) CLIENT SIDE: create context and encrypt answers
    context = create_ckks_context()

    # Example user answers (length must match STRESS_WEIGHTS on the server)
    answers = [1, 3, 5, 2, 4]
    enc_vec = encrypt_answers(context, answers)

    context_hex = serialize_public_context(context)
    ct_hex = serialize_ciphertext(enc_vec)

    payload = {
        "context_hex": context_hex,
        "ciphertext": ct_hex,
    }

    # 2) Send to your backend
    resp = requests.post(BACKEND_URL, json=payload, timeout=10)
    print("Status code:", resp.status_code)
    print("Response JSON:", resp.json())

    resp_json = resp.json()
    enc_score_hex = resp_json["encrypted_score"]

    # 3) CLIENT SIDE: deserialize + decrypt the returned score
    enc_score = deserialize_ciphertext(context, enc_score_hex)
    score_plain = decrypt_vector(context, enc_score)

    print("Decrypted score (approx):", score_plain)


if __name__ == "__main__":
    run_test_client()
