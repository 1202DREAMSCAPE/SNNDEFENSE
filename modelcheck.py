from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from keras.config import enable_unsafe_deserialization


enable_unsafe_deserialization()

enhanced_model_path = "enhanced_weights/enhanced_CEDAR.keras"

@register_keras_serializable(package="Custom")
def triplet_loss(margin=1):
    def loss(y_true, y_pred):
        anchor = y_pred[:, 0]
        positive = y_pred[:, 1]
        negative = y_pred[:, 2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

try:
    model = load_model(enhanced_model_path, custom_objects={"triplet_loss": triplet_loss})
    print("Model loaded successfully!")
    print("Model configuration:", model.get_config())
except Exception as e:
    print(f"Error loading model: {e}")