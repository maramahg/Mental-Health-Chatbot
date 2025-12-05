from flask import Flask, request, jsonify

from encryption_module import (
    deserialize_public_context,
    deserialize_ciphertext,
    serialize_ciphertext,
)

app = Flask(__name__)

# --------------------------------------------------------------------
# Stress scoring configuration
# --------------------------------------------------------------------
# Example weights for stress scoring.
# IMPORTANT: Length must match the number of answers the client sends
# (e.g., 5 answers → 5 weights, 10 answers → 10 weights).
STRESS_WEIGHTS = [0.2, 0.3, 0.1, 0.25, 0.15]


@app.route("/")
def home():
    """Simple health-check endpoint."""
    return "Server is running (HE backend)"


@app.route("/process", methods=["POST"])
def process():
    """
    Main endpoint called by the chatbot's encrypted stress check.

    Expected JSON body:
    {
        "context_hex": "<public_context_hex>",
        "ciphertext": "<encrypted_answers_hex>"
    }

    Returns:
    {
        "encrypted_score": "<encrypted_score_hex>"
    }
    """
    data = request.get_json(force=True)

    context_hex = data.get("context_hex")
    ct_hex = data.get("ciphertext")

    if context_hex is None or ct_hex is None:
        return (
            jsonify(
                {
                    "error": (
                        "Missing required fields: 'context_hex' "
                        "and/or 'ciphertext'."
                    )
                }
            ),
            400,
        )

    # 1) Rebuild the public CKKS context (server NEVER has the secret key)
    try:
        context = deserialize_public_context(context_hex)
    except Exception as e:
        return jsonify({"error": f"Failed to deserialize context: {e}"}), 400

    # 2) Rebuild the encrypted answers vector from hex
    try:
        enc_answers = deserialize_ciphertext(context, ct_hex)
    except Exception as e:
        return jsonify({"error": f"Failed to deserialize ciphertext: {e}"}), 400

    # 3) Homomorphic weighted sum:
    #    - enc_answers is encrypted
    #    - STRESS_WEIGHTS is plaintext
    #    - result (enc_score) is still encrypted
    try:
        # Element-wise multiply encrypted answers by plain weights
        enc_weighted = enc_answers * STRESS_WEIGHTS

        # Sum all elements into a single encrypted score
        enc_score = enc_weighted.sum()
    except Exception as e:
        return jsonify({"error": f"Homomorphic computation failed: {e}"}), 500

    # 4) Serialize encrypted score back to hex so the CLIENT can decrypt it
    try:
        score_hex = serialize_ciphertext(enc_score)
    except Exception as e:
        return jsonify({"error": f"Failed to serialize encrypted score: {e}"}), 500

    return jsonify({"encrypted_score": score_hex})


if __name__ == "__main__":
    # debug=True enables auto-reload and detailed error messages during development.
    app.run(host="127.0.0.1", port=5000, debug=True)
