from typing import List, Sequence

import tenseal as ts

# ---------------------------------------------------------------------------
# CKKS default parameters
# ---------------------------------------------------------------------------

DEFAULT_POLY_MOD_DEGREE = 8192
DEFAULT_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]  # Common safe default
DEFAULT_GLOBAL_SCALE = 2**40


# ---------------------------------------------------------------------------
# 1. CONTEXT / KEY GENERATION  (CLIENT SIDE)
# ---------------------------------------------------------------------------


def create_ckks_context(
    poly_mod_degree: int = DEFAULT_POLY_MOD_DEGREE,
    coeff_mod_bit_sizes: Sequence[int] = DEFAULT_COEFF_MOD_BIT_SIZES,
    global_scale: float = DEFAULT_GLOBAL_SCALE,
) -> ts.Context:
    """
    Create and configure a TenSEAL CKKS context.

    Intended to run on the CLIENT. It generates:
    - encryption parameters
    - secret key (kept locally)
    - Galois & relinearization keys (for homomorphic operations)

    Returns
    -------
    ts.Context
        TenSEAL CKKS context with a secret key.
        This object should NOT be sent to the server directly.
        Use `serialize_public_context()` first.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )

    # Scale used for CKKS approximate arithmetic
    context.global_scale = global_scale

    # Keys needed for rotations and multiplications
    context.generate_galois_keys()
    context.generate_relin_keys()

    return context


def serialize_public_context(context: ts.Context) -> str:
    """
    Serialize the TenSEAL context WITHOUT the secret key as a hex string.

    This is what you send to the SERVER so it can perform homomorphic
    operations on ciphertexts.

    Parameters
    ----------
    context : ts.Context
        Client-side context (includes secret key locally).

    Returns
    -------
    str
        Hex string of the public context (no secret key).
    """
    public_bytes = context.serialize(save_secret_key=False)
    return public_bytes.hex()


def deserialize_public_context(context_hex: str) -> ts.Context:
    """
    Rebuild a public TenSEAL context from a hex string.

    Used on the SERVER (and also locally for testing).

    Parameters
    ----------
    context_hex : str
        Hex string of a serialized public context.

    Returns
    -------
    ts.Context
        TenSEAL context without a secret key.
    """
    context_bytes = bytes.fromhex(context_hex)
    return ts.context_from(context_bytes)


# ---------------------------------------------------------------------------
# 2. ENCRYPT / DECRYPT HELPERS  (CLIENT SIDE ENCRYPT, CLIENT SIDE DECRYPT)
# ---------------------------------------------------------------------------


def encode_answers_to_floats(answers: Sequence[float | int]) -> List[float]:
    """
    Convert user answers (e.g., Likert-scale 1–5 integers) into floats.

    You can modify this later if you need normalization or feature scaling.

    Parameters
    ----------
    answers : sequence of float or int
        List of user responses.

    Returns
    -------
    list of float
        Answers converted to float.
    """
    return [float(a) for a in answers]


def encrypt_answers(
    context: ts.Context,
    answers: Sequence[float | int],
) -> ts.ckks_vector:
    """
    Encrypt a list of user answers as a CKKS ciphertext vector.

    Parameters
    ----------
    context : ts.Context
        CKKS context that includes the secret key (CLIENT side).
    answers : sequence of float or int
        User answers, e.g., [1, 3, 5, 2, 4].

    Returns
    -------
    ts.ckks_vector
        Encrypted vector of answers.
    """
    float_values = encode_answers_to_floats(answers)
    enc_vec = ts.ckks_vector(context, float_values)
    return enc_vec


def decrypt_vector(
    context: ts.Context,
    enc_vec: ts.ckks_vector,
) -> List[float]:
    """
    Decrypt a CKKS ciphertext vector back to a list of floats.

    Parameters
    ----------
    context : ts.Context
        CKKS context that has the secret key (CLIENT side).
    enc_vec : ts.ckks_vector
        Encrypted vector.

    Returns
    -------
    list of float
        Decrypted approximate values.
    """
    return enc_vec.decrypt()


# ---------------------------------------------------------------------------
# 3. SERIALIZATION OF CIPHERTEXT (FOR SENDING IN JSON)
# ---------------------------------------------------------------------------


def serialize_ciphertext(enc_vec: ts.ckks_vector) -> str:
    """
    Serialize an encrypted CKKS vector into a hex string.

    This hex string can be safely sent in a JSON body over HTTP.

    Parameters
    ----------
    enc_vec : ts.ckks_vector
        Encrypted vector to serialize.

    Returns
    -------
    str
        Hex string representing the encrypted vector.
    """
    ct_bytes = enc_vec.serialize()
    return ct_bytes.hex()


def deserialize_ciphertext(
    context: ts.Context,
    ct_hex: str,
) -> ts.ckks_vector:
    """
    Rebuild a CKKS encrypted vector from a hex string.

    Used mainly on the SERVER side (with a public context),
    but can also be used on the CLIENT for testing.

    Parameters
    ----------
    context : ts.Context
        TenSEAL context (usually public-only on the server).
    ct_hex : str
        Hex string of the ciphertext.

    Returns
    -------
    ts.ckks_vector
        Reconstructed encrypted CKKS vector.
    """
    ct_bytes = bytes.fromhex(ct_hex)
    return ts.ckks_vector_from(context, ct_bytes)


# ---------------------------------------------------------------------------
# 4. SIMPLE LOCAL TEST (OPTIONAL)
#    Run:  python encryption_module.py
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """
    Simple sanity check:
    - Create context
    - Encrypt a small vector
    - Decrypt it back and print the values
    """
    print("[Self-test] Creating CKKS context...")
    context = create_ckks_context()

    original = [1.0, 2.0, 3.0, 4.0]
    print(f"[Self-test] Original vector: {original}")

    enc = encrypt_answers(context, original)
    print("[Self-test] Encrypted vector created.")

    decrypted = decrypt_vector(context, enc)
    print(f"[Self-test] Decrypted vector: {decrypted}")

    ok = all(abs(o - d) < 1e-3 for o, d in zip(original, decrypted))
    print(f"[Self-test] Approx equal? {ok}")


if __name__ == "__main__":
    _self_test()
