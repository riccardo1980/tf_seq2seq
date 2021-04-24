import tensorflow as tf
# load char2ind and ind2char from utils

def generate_text(model,
                  start_string,
                  temperature: float = 1.0,
                  num_generate: int = 1000) -> str:

    """
    Evaluation step (generating text using the learned model)

    :param model:
        TensorFlow learned model

    :param start_string:
        Initial string to feed to model

    :param temperature:
        Low temperatures results in more predictable text.
        Higher temperatures results in more surprising text.

    :param num_generate:
        number of characters to generate

    """

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2ind[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(ind2char[predicted_id])

    return (start_string + ''.join(text_generated))