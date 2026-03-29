import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import globalSettings
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocabSize, dModel, maxContextLength):
        super().__init__()
        self.dModel = dModel
        self.maxContextLength = maxContextLength
        self.embedding = layers.Embedding(vocabSize, dModel)
        self.posEncoding = self.positionalEncoding(maxContextLength, dModel)

    def get_angles(self, pos, i, dModel):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(dModel, tf.float32))
        return pos * angles

    def positionalEncoding(self, maxContextLength, dModel):
        angleRads = self.get_angles(
            pos=tf.range(maxContextLength, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(dModel, dtype=tf.float32)[tf.newaxis, :],
            dModel=dModel)
        
        sines = tf.math.sin(angleRads[:, 0::2])
        cosines = tf.math.cos(angleRads[:, 1::2])
        
        posEncoding = tf.concat([sines, cosines], axis=-1)
        return tf.cast(posEncoding, tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))
        x = x + self.posEncoding[tf.newaxis, :length, :]
        return x

class CausalSelfAttention(layers.Layer):
    def __init__(self, dModel, numHeads, dropoutRate=0.1):
        super().__init__()
        self.numHeads = numHeads
        self.headDim = dModel // numHeads
        
        self.mha = layers.MultiHeadAttention(
            num_heads=numHeads, key_dim=self.headDim, dropout=dropoutRate
        )
        self.dropout = layers.Dropout(dropoutRate)

    def call(self, x, training=False):
        attnOutput = self.mha(query=x, value=x, key=x, use_causal_mask=True, training=training)
        return self.dropout(attnOutput, training=training)

class TransformerBlock(layers.Layer):
    def __init__(self, dModel, numHeads, dff, dropoutRate=0.1):
        super().__init__()
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attnLayer = CausalSelfAttention(dModel, numHeads, dropoutRate)
        
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='gelu'),
            layers.Dense(dModel),
            layers.Dropout(dropoutRate)
        ])

    def call(self, x, training=False):
        x = x + self.attnLayer(self.layernorm1(x), training=training)
        x = x + self.ffn(self.layernorm2(x), training=training)
        return x

class TfModel(keras.Model):
    def __init__(self, vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate=0.1):
        super().__init__()
        self.dModel = dModel
        self.vocabSize = vocabSize
        
        self.posEmbedding = PositionalEmbedding(vocabSize, dModel, maxContextLength)
        self.dropout = layers.Dropout(dropoutRate)
        
        self.transformerBlocks = [
            TransformerBlock(dModel, numHeads, dff, dropoutRate)
            for _ in range(numLayers)
        ]
        
        self.finalLayernorm = layers.LayerNormalization(epsilon=1e-6)
        
        self.outputLayer = None

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):
        x = self.posEmbedding(x)
        x = self.dropout(x, training=training)
        
        for block in self.transformerBlocks:
            x = block(x, training=training)
            
        x = self.finalLayernorm(x)
        
        embeddingMatrix = self.posEmbedding.embedding.embeddings
        logits = tf.matmul(x, embeddingMatrix, transpose_b=True)
        
        return logits

def BuildTfModel(vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate=0.1):
    model = TfModel(vocabSize, dModel, numHeads, dff, maxContextLength, numLayers, dropoutRate)
        
    dummyInput = tf.zeros((1, maxContextLength), dtype=tf.int32)
    _ = model(dummyInput)
    
    return model

if __name__ == "__main__":
    VOCAB_SIZE = globalSettings.VOCAB_SIZE
    D_MODEL = globalSettings.D_MODEL
    NUM_HEADS = globalSettings.NUM_HEADS
    DFF = globalSettings.DFF
    MAX_CONTEXT_LENGTH = globalSettings.MAX_CONTEXT_LENGTH
    NUM_LAYERS = globalSettings.NUM_LAYERS
    
    try:
        model = BuildTfModel(VOCAB_SIZE, D_MODEL, NUM_HEADS, DFF, MAX_CONTEXT_LENGTH, NUM_LAYERS)
        model.summary()
        
        dummyInput = tf.random.uniform((2, MAX_CONTEXT_LENGTH), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
        output = model(dummyInput)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Could not build model due to local environment issue: {e}")
